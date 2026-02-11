import json
import asyncio
import random
import yaml
from pathlib import Path
from tqdm import tqdm
from src.openrouter_client import call_llm
import re
import time
from json import JSONDecodeError
from typing import Any, Dict, List, Optional
import argparse


# ======================
# Path / Prompt Utilities
# ======================

def _get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def _get_output_root() -> Path:
    return _get_project_root() / "output"

def _load_prompts() -> dict:
    project_root = _get_project_root()
    prompts_file = project_root / "resources" / "prompt" / "simulator_prompts.yaml"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = _load_prompts()

def _sanitize_filename(s: str) -> str:
    # openai/o1 -> openai__o1
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", s)

# ======================
# experiments.yaml loader
# ======================

def _load_experiments_yaml() -> dict:
    project_root = _get_project_root()
    exp_file = project_root / "resources" / "experiments.yaml"
    if not exp_file.exists():
        raise FileNotFoundError(f"experiments.yaml not found: {exp_file}")
    with open(exp_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

EXPERIMENTS_YAML = _load_experiments_yaml()

def _get_model_short_id(model_full_name: str) -> str:
    """
    experiments.yaml 'models' mapping is the single source of truth:
      models:
        o1: openai/o1
        r1: deepseek/deepseek-r1
        ...
    We invert it here.
    """
    models_map = EXPERIMENTS_YAML.get("models", {})
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

def _load_experiments_for_model(model_full_name: str) -> List[dict]:
    """
    Load only the 16 experiments for the given model, and normalize into
    simulation format:
      {
        "experiment_id": <id>,
        "label": <id>,
        "turn1_key": "T1_<t1>",
        "turn2_key": "T2_<t2>",
      }
    """
    short = _get_model_short_id(model_full_name)

    exps = EXPERIMENTS_YAML.get("experiments", [])
    if not exps:
        raise ValueError("No 'experiments' section in resources/experiments.yaml")

    filtered = [e for e in exps if e.get("model") == short]
    if len(filtered) != 16:
        raise ValueError(f"Expected 16 experiments for model '{short}', got {len(filtered)}")

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

def _render_ai_user_prompt_turn1(exp: dict, scenario_text: str, image_caption: str, sim_msg_1: str) -> str:
    t1_key = exp["turn1_key"]
    template = PROMPTS["user_messages"]["ai_turn1_user_templates"][t1_key]
    addon_text = PROMPTS.get("variant_addons", {}).get(t1_key, "").rstrip()
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
) -> str:
    t2_key = exp["turn2_key"]
    template = PROMPTS["user_messages"]["ai_turn2_user_templates"][t2_key]
    addon_text = PROMPTS.get("variant_addons", {}).get(t2_key, "").rstrip()
    return template.format(
        scenario_text=scenario_text,
        image_caption=image_caption,
        sim_msg_1=sim_msg_1,
        ai_msg_1=ai_msg_1,
        sim_msg_2=sim_msg_2,
        addon_text=addon_text,
    )


# ======================
# Retry
# ======================

def _is_retryable_error(err: Exception) -> bool:
    # JSON decode errors from broken provider responses should be retried
    if isinstance(err, JSONDecodeError):
        return True

    msg = (str(err) or "").lower()

    transient_markers = [
        "rate limit", "429",
        "timeout", "timed out",
        "overloaded",
        "temporarily unavailable",
        "service unavailable", "503",
        "502", "bad gateway",
        "gateway timeout", "504",
        "connection reset", "connection aborted",
        "network", "server error", "internal error",
        # NEW: JSON parse failure patterns
        "expecting value",            # JSONDecodeError message pattern
        "jsondecodeerror",
        "invalid json",
    ]
    return any(m in msg for m in transient_markers)


# ======================
# Simulation Core
# ======================

async def run_single_simulation(
    target_model: str,
    experiment: dict,
    case: dict,
    mode: str,
    semaphore: asyncio.Semaphore,
    simulator_model: str,
    max_retries: int,
) -> Optional[dict]:
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

            simulator_sys_prompt_turn_1 = PROMPTS["simulator_base_system_prompt_turn_1"].format(
                target_belief=target_belief
            )
            simulator_sys_prompt_turn_2 = PROMPTS["simulator_base_system_prompt_turn_2"].format(
                target_belief=target_belief
            )
            target_sys_prompt = PROMPTS["target_ai_system_prompt"]

            scenario_text = case["scenario"]
            image_caption = case["caption"]
            history: List[Dict[str, str]] = []

            # [Turn 1] Simulator
            sim_msg_1 = await call_llm(
                model_name=simulator_model,
                messages=[
                    {"role": "system", "content": simulator_sys_prompt_turn_1},
                    {
                        "role": "user",
                        "content": PROMPTS["user_messages"]["simulator_turn1_user"].format(
                            scenario_text=scenario_text,
                            target_belief=target_belief,
                            image_caption=image_caption,
                        ),
                    },
                ],
                temperature=0.0,
                max_retries=max_retries,
            )
            history.append({"role": "Doctor", "content": sim_msg_1})

            # [Turn 1] Target AI
            ai_turn1_user = _render_ai_user_prompt_turn1(
                experiment,
                scenario_text=scenario_text,
                image_caption=image_caption,
                sim_msg_1=sim_msg_1,
            )
            ai_msg_1 = await call_llm(
                model_name=target_model,
                messages=[
                    {"role": "system", "content": target_sys_prompt},
                    {"role": "user", "content": ai_turn1_user},
                ],
                temperature=0.0,
                max_retries=max_retries,
            )
            history.append({"role": "AI", "content": ai_msg_1})

            # [Turn 2] Simulator (Rebuttal)
            sim_msg_2 = await call_llm(
                model_name=simulator_model,
                messages=[
                    {"role": "system", "content": simulator_sys_prompt_turn_2},
                    {
                        "role": "user",
                        "content": PROMPTS["user_messages"]["simulator_turn2_user"].format(
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
                max_retries=max_retries,
            )
            history.append({"role": "Doctor", "content": sim_msg_2})

            # [Turn 2] Target AI
            ai_turn2_user = _render_ai_user_prompt_turn2(
                experiment,
                scenario_text=scenario_text,
                image_caption=image_caption,
                sim_msg_1=sim_msg_1,
                ai_msg_1=ai_msg_1,
                sim_msg_2=sim_msg_2,
            )
            ai_msg_2 = await call_llm(
                model_name=target_model,
                messages=[
                    {"role": "system", "content": target_sys_prompt},
                    {"role": "user", "content": ai_turn2_user},
                ],
                temperature=0.0,
                max_retries=max_retries,
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
) -> dict:
    tasks = []
    for exp in experiments:
        for case in dataset:
            tasks.append(run_single_simulation(model_name, exp, case, "error", semaphore, simulator_model, max_retries))
            tasks.append(run_single_simulation(model_name, exp, case, "correct", semaphore, simulator_model, max_retries))

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


# ======================
# Runner
# ======================

def _by_model_dir() -> Path:
    # output/simulation/by_model
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
):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # you want --model only: target_models must be 1 item typically
    if not target_models:
        raise ValueError("No target model provided. Use --model <provider/model>.")

    output_root = _get_output_root()
    sim_root = output_root / "simulation"
    by_model_dir = sim_root / "by_model"
    sim_root.mkdir(parents=True, exist_ok=True)
    by_model_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    combined: dict[str, dict] = {}
    for idx, model in enumerate(target_models, 1):
        out_path = _model_output_path(model)

        if out_path.exists():
            print(f"[{idx}/{len(target_models)}] {model} -> SKIP (exists): {out_path}")
            continue

        experiments = _load_experiments_for_model(model)
        print(f"[{idx}/{len(target_models)}] {model} -> RUN (experiments={len(experiments)})")

        result = await simulate_model_all_experiments(
            model_name=model,
            dataset=cases,
            experiments=experiments,
            semaphore=semaphore,
            simulator_model=simulator_model,
            max_retries=max_retries,
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
    parser.add_argument("--model", action="append", required=True,
                        help="Target model full name (e.g., openai/o1). Can be repeated.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--mode", choices=["correct", "error"], default="correct")
    parser.add_argument("--simulator_model", required=True)
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)

    # compatibility with run_openrouter.sh (so we can write output/simulation/<output_file>)
    parser.add_argument("--output_file", default=None)

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
        )
    )
