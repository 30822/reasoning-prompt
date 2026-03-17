"""
Simulation: 16 experiments per model (experiments.yaml), simulator->AI->simulator->AI.
Writes to output/simulation/by_model. Uses OpenAI API only (utils.call_llm_openai).
"""

import json
import asyncio
import random
import yaml
from pathlib import Path
from tqdm import tqdm
from src.utils import call_llm_openai
import re
from typing import Dict, List, Optional
import argparse


def _get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def _get_output_root() -> Path:
    return _get_project_root() / "output"

def _load_prompts(prompts_file: Optional[Path] = None) -> dict:
    if prompts_file is None:
        prompts_file = _get_project_root() / "resources" / "prompt" / "simulator_prompts.yaml"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = _load_prompts()

def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

def _load_experiments_yaml(experiments_file: Optional[Path] = None) -> dict:
    if experiments_file is None:
        experiments_file = _get_project_root() / "resources" / "experiments.yaml"
    if not experiments_file.exists():
        raise FileNotFoundError(f"experiments.yaml not found: {experiments_file}")
    with open(experiments_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

EXPERIMENTS_YAML = _load_experiments_yaml()

def _get_model_short_id(model_full_name: str, experiments_cfg: Optional[dict] = None) -> str:
    cfg = experiments_cfg if experiments_cfg is not None else EXPERIMENTS_YAML
    models_map = cfg.get("models", {})
    if not models_map:
        raise ValueError("No 'models' mapping in resources/experiments.yaml")
    inv = {full: short for short, full in models_map.items()}
    short = inv.get(model_full_name)
    if not short:
        raise ValueError(
            f"Unknown model '{model_full_name}'. "
            f"Available: {list(inv.keys())}"
        )
    return short

def _load_experiments_for_model(model_full_name: str, experiments_cfg: Optional[dict] = None) -> List[dict]:
    cfg = experiments_cfg if experiments_cfg is not None else EXPERIMENTS_YAML
    short = _get_model_short_id(model_full_name, cfg)
    exps = cfg.get("experiments", [])
    if not exps:
        raise ValueError("No 'experiments' section in experiments config")
    filtered = [e for e in exps if e.get("model") == short]
    if not filtered:
        raise ValueError(f"No experiments for model '{short}'")
    norm: List[dict] = []
    for e in filtered:
        t1 = e.get("t1")
        t2 = e.get("t2")
        exp_id = e.get("id")
        if not (t1 and t2 and exp_id):
            raise ValueError(f"Bad experiment entry: {e}")
        norm.append({
            "experiment_id": exp_id,
            "label": exp_id,
            "turn1_key": f"T1_{t1}",
            "turn2_key": f"T2_{t2}",
        })
    return norm


def format_options(options_dict: dict) -> str:
    return "".join([f"({k}) {v}\n" for k, v in options_dict.items()])

def _render_ai_user_prompt_turn1(exp: dict, scenario_text: str, image_caption: str, sim_msg_1: str, prompts: Optional[dict] = None) -> str:
    prompts = prompts if prompts is not None else PROMPTS
    t1_key = exp["turn1_key"]
    template = prompts["user_messages"]["ai_turn1_user_templates"][t1_key]
    addon_text = prompts.get("variant_addons", {}).get(t1_key, "").rstrip()
    return template.format(
        scenario_text=scenario_text,
        image_caption=image_caption,
        sim_msg_1=sim_msg_1,
        addon_text=addon_text,
    )

def _render_ai_user_prompt_turn2(
    exp: dict,
    scenario_text: str,
    image_caption: str,
    sim_msg_1: str,
    ai_msg_1: str,
    sim_msg_2: str,
    prompts: Optional[dict] = None,
) -> str:
    prompts = prompts if prompts is not None else PROMPTS
    t2_key = exp["turn2_key"]
    template = prompts["user_messages"]["ai_turn2_user_templates"][t2_key]
    addon_text = prompts.get("variant_addons", {}).get(t2_key, "").rstrip()
    return template.format(
        scenario_text=scenario_text,
        image_caption=image_caption,
        sim_msg_1=sim_msg_1,
        ai_msg_1=ai_msg_1,
        sim_msg_2=sim_msg_2,
        addon_text=addon_text,
    )


async def run_single_simulation(
    target_model: str,
    experiment: dict,
    case: dict,
    mode: str,
    semaphore: asyncio.Semaphore,
    simulator_model: str,
    max_retries: int,
    prompts: Optional[dict] = None,
) -> Optional[dict]:
    prompts = prompts if prompts is not None else PROMPTS
    async with semaphore:
        try:
            if mode == "error":
                if not case.get("distractors"):
                    return None
                target_belief = random.choice(case["distractors"])
            elif mode == "correct":
                target_belief = case["options"][case["correct_option"]]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            simulator_sys_prompt_turn_1 = prompts["simulator_base_system_prompt_turn_1"].format(
                target_belief=target_belief
            )
            simulator_sys_prompt_turn_2 = prompts["simulator_base_system_prompt_turn_2"].format(
                target_belief=target_belief
            )
            target_sys_prompt = prompts["target_ai_system_prompt"]

            scenario_text = case["scenario"]
            image_caption = case["caption"]
            history: List[Dict[str, str]] = []

            sim_msg_1 = await call_llm_openai(
                model_name=simulator_model,
                messages=[
                    {"role": "system", "content": simulator_sys_prompt_turn_1},
                    {
                        "role": "user",
                        "content": prompts["user_messages"]["simulator_turn1_user"].format(
                            scenario_text=scenario_text,
                            target_belief=target_belief,
                            image_caption=image_caption,
                        ),
                    },
                ],
                temperature=0.0,
                retries=max_retries,
            )
            history.append({"role": "Doctor", "content": sim_msg_1})

            ai_turn1_user = _render_ai_user_prompt_turn1(
                experiment,
                scenario_text=scenario_text,
                image_caption=image_caption,
                sim_msg_1=sim_msg_1,
                prompts=prompts,
            )
            ai_msg_1 = await call_llm_openai(
                model_name=target_model,
                messages=[
                    {"role": "system", "content": target_sys_prompt},
                    {"role": "user", "content": ai_turn1_user},
                ],
                temperature=0.0,
                retries=max_retries,
            )
            history.append({"role": "AI", "content": ai_msg_1})

            sim_msg_2 = await call_llm_openai(
                model_name=simulator_model,
                messages=[
                    {"role": "system", "content": simulator_sys_prompt_turn_2},
                    {
                        "role": "user",
                        "content": prompts["user_messages"]["simulator_turn2_user"].format(
                            scenario_text=scenario_text,
                            image_caption=image_caption,
                            options_text=format_options(case["options"]),
                            sim_msg_1=sim_msg_1,
                            ai_msg_1=ai_msg_1,
                            target_belief=target_belief,
                        ),
                    },
                ],
                temperature=0.0,
                retries=max_retries,
            )
            history.append({"role": "Doctor", "content": sim_msg_2})

            ai_turn2_user = _render_ai_user_prompt_turn2(
                experiment,
                scenario_text=scenario_text,
                image_caption=image_caption,
                sim_msg_1=sim_msg_1,
                ai_msg_1=ai_msg_1,
                sim_msg_2=sim_msg_2,
                prompts=prompts,
            )
            ai_msg_2 = await call_llm_openai(
                model_name=target_model,
                messages=[
                    {"role": "system", "content": target_sys_prompt},
                    {"role": "user", "content": ai_turn2_user},
                ],
                temperature=0.0,
                retries=max_retries,
            )
            history.append({"role": "AI", "content": ai_msg_2})

            return {
                "case_id": case.get("case_id"),
                "mode": mode,
                "experiment_id": experiment["experiment_id"],
                "experiment_label": experiment.get("label", experiment["experiment_id"]),
                "dialogue": history,
                "ground_truth": case["correct_option"],
                "target_belief": target_belief,
            }

        except Exception as e:
            print(f"[ERR] {target_model} case={case.get('case_id')} exp={experiment.get('experiment_id')} mode={mode} | {e}")
            return None


async def simulate_model_all_experiments(
    model_name: str,
    dataset: list,
    experiments: list,
    semaphore: asyncio.Semaphore,
    simulator_model: str,
    max_retries: int,
    prompts: Optional[dict] = None,
) -> dict:
    prompts = prompts if prompts is not None else PROMPTS
    tasks = []
    for exp in experiments:
        for case in dataset:
            tasks.append(run_single_simulation(model_name, exp, case, "error", semaphore, simulator_model, max_retries, prompts=prompts))
            tasks.append(run_single_simulation(model_name, exp, case, "correct", semaphore, simulator_model, max_retries, prompts=prompts))

    logs = []
    with tqdm(total=len(tasks), desc=f"[{model_name}]", leave=False) as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if res:
                logs.append(res)
            pbar.update(1)
    return {
        "model_name": model_name,
        "logs": logs,
    }


def _by_model_dir() -> Path:
    return _get_output_root() / "simulation" / "by_model"

def _model_output_path(model_name: str) -> Path:
    return _by_model_dir() / f"{_sanitize_filename(model_name)}.json"

async def run_simulation(
    target_models: list[str],
    input_file: str,
    mode: str,
    simulator_model: str,
    max_concurrent: int,
    max_retries: int,
    output_file: str | None = None,
    prompts_file: str | Path | None = None,
    experiments_file: str | Path | None = None,
):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not target_models:
        raise ValueError("No target model provided. Use --model <provider/model>.")

    project_root = _get_project_root()
    prompts_path = None
    if prompts_file:
        p = Path(prompts_file)
        prompts_path = p if p.is_absolute() else project_root / p
    experiments_path = None
    if experiments_file:
        p = Path(experiments_file)
        experiments_path = p if p.is_absolute() else project_root / p
    prompts = _load_prompts(prompts_path) if prompts_path else PROMPTS
    experiments_cfg = _load_experiments_yaml(experiments_path) if experiments_path else None

    output_root = _get_output_root()
    sim_root = output_root / "simulation"
    by_model_dir = sim_root / "by_model"
    sim_root.mkdir(parents=True, exist_ok=True)
    by_model_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    for idx, model in enumerate(target_models, 1):
        out_path = _model_output_path(model)

        if out_path.exists():
            print(f"[{idx}/{len(target_models)}] {model} -> SKIP (exists): {out_path}")
            continue

        experiments = _load_experiments_for_model(model, experiments_cfg)
        print(f"[{idx}/{len(target_models)}] {model} -> RUN (experiments={len(experiments)})")

        result = await simulate_model_all_experiments(
            model_name=model,
            dataset=cases,
            experiments=experiments,
            semaphore=semaphore,
            simulator_model=simulator_model,
            max_retries=max_retries,
            prompts=prompts,
        )

        model_result = {
            "model_name": model,
            "logs": result.get("logs", []),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_result, f, ensure_ascii=False, indent=2)

        print(f"   Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--mode", choices=["correct", "error"], default="correct")
    parser.add_argument("--simulator_model", required=True)
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--prompts_file", default=None, help="e.g. resources/prompt/simulator_prompts_robustness.yaml")
    parser.add_argument("--experiments_file", default=None, help="e.g. resources/experiments_robustness.yaml")

    args = parser.parse_args()

    asyncio.run(
        run_simulation(
            target_models=args.model,
            input_file=args.input_file,
            mode=args.mode,
            simulator_model=args.simulator_model,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            output_file=args.output_file,
            prompts_file=args.prompts_file,
            experiments_file=args.experiments_file,
        )
    )
