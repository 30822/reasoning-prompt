"""Two-turn clinician–AI simulation under the 4x4 prompt matrix.

Each case is run in both `correct` and `incorrect` belief conditions.
"""
import json
import asyncio
import random
import yaml
from pathlib import Path
from tqdm import tqdm
from src.openrouter_client import get_llm_caller
import re
from typing import Dict, List, Optional
import argparse

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
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

def _load_experiments_yaml() -> dict:
    project_root = _get_project_root()
    exp_file = project_root / "resources" / "experiments.yaml"
    if not exp_file.exists():
        raise FileNotFoundError(f"experiments.yaml not found: {exp_file}")
    with open(exp_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

EXPERIMENTS_YAML = _load_experiments_yaml()

def _get_model_short_id(model_full_name: str) -> str:
    """Invert experiments.yaml models map: 'openai/o3' -> 'o3'."""
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
    """Load the 16 prompt cells (P0–P15) for one target model."""
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


async def run_single_simulation(
    target_model: str,
    experiment: dict,
    case: dict,
    mode: str,
    semaphore: asyncio.Semaphore,
    simulator_model: str,
    max_retries: int,
) -> Optional[dict]:
    """One 2-turn dialogue. mode is `correct` or `incorrect` (simulator belief)."""
    async with semaphore:
        try:
            if mode == "incorrect":
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

            def _call_llm(model_name: str, messages, **kwargs):
                return get_llm_caller(model_name)(model_name=model_name, messages=messages, **kwargs)

            sim_msg_1 = await _call_llm(
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
                retries=max_retries,
            )
            history.append({"role": "Doctor", "content": sim_msg_1})

            ai_turn1_user = _render_ai_user_prompt_turn1(
                experiment,
                scenario_text=scenario_text,
                image_caption=image_caption,
                sim_msg_1=sim_msg_1,
            )
            ai_msg_1 = await _call_llm(
                model_name=target_model,
                messages=[
                    {"role": "system", "content": target_sys_prompt},
                    {"role": "user", "content": ai_turn1_user},
                ],
                temperature=0.0,
                retries=max_retries,
            )
            history.append({"role": "AI", "content": ai_msg_1})

            sim_msg_2 = await _call_llm(
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
            )
            ai_msg_2 = await _call_llm(
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


def _log_key(log: dict) -> tuple:
    return (
        str(log.get("case_id", "")),
        str(log.get("experiment_id", log.get("experiment_label", ""))),
        str(log.get("mode", "")),
    )


async def simulate_model_all_experiments(
    model_name: str,
    dataset: list,
    experiments: list,
    semaphore: asyncio.Semaphore,
    simulator_model: str,
    max_retries: int,
    checkpoint_path: Optional[Path] = None,
    done_keys: Optional[set] = None,
    existing_logs: Optional[list] = None,
    save_every: int = 20,
) -> dict:
    done_keys = done_keys or set()
    logs: list = list(existing_logs or [])
    save_lock = asyncio.Lock()

    work_items: list[tuple[dict, dict, str]] = []
    for exp in experiments:
        for case in dataset:
            for mode in ("incorrect", "correct"):
                key = (str(case.get("case_id", "")), str(exp["experiment_id"]), mode)
                if key in done_keys:
                    continue
                if mode == "incorrect" and not case.get("distractors"):
                    continue
                work_items.append((exp, case, mode))

    if not work_items:
        return {"model_name": model_name, "logs": logs}

    async def run_and_save(item: tuple):
        exp, case, mode = item
        res = await run_single_simulation(
            model_name, exp, case, mode, semaphore, simulator_model, max_retries
        )
        if res:
            async with save_lock:
                logs.append(res)
                done_keys.add(_log_key(res))
                if checkpoint_path and len(logs) % save_every == 0:
                    payload = {"model_name": model_name, "logs": logs}
                    checkpoint_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
        return res

    tasks = [run_and_save(item) for item in work_items]
    with tqdm(total=len(tasks), desc=f"[{model_name}]", leave=False) as pbar:
        for coro in asyncio.as_completed(tasks):
            await coro
            pbar.update(1)

    if checkpoint_path and logs:
        payload = {"model_name": model_name, "logs": logs}
        checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {"model_name": model_name, "logs": logs}


def _by_model_dir() -> Path:
    return _get_output_root() / "simulation" / "by_model"

def _model_output_path(model_name: str) -> Path:
    return _by_model_dir() / f"{_sanitize_filename(model_name)}.json"


def _checkpoint_path(model_name: str) -> Path:
    return _by_model_dir() / f"{_sanitize_filename(model_name)}_checkpoint.json"


async def run_simulation(
    target_models: list[str],
    input_file: str,
    mode: str,
    simulator_model: str,
    max_concurrent: int,
    max_retries: int,
    output_file: str | None = None,
    resume: bool = True,
):
    """Run all 16 cells x 2 belief conditions for each target model."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not target_models:
        raise ValueError("No target model provided. Use --model <provider/model>.")

    output_root = _get_output_root()
    sim_root = output_root / "simulation"
    by_model_dir = sim_root / "by_model"
    sim_root.mkdir(parents=True, exist_ok=True)
    by_model_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    for idx, model in enumerate(target_models, 1):
        out_path = _model_output_path(model)
        ckpt_path = _checkpoint_path(model)

        if out_path.exists() and resume:
            print(f"[{idx}/{len(target_models)}] {model} -> SKIP (complete): {out_path}")
            if ckpt_path.exists():
                ckpt_path.unlink()
            continue

        done_keys: set = set()
        existing_logs: list = []
        if resume and ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                existing_logs = ckpt.get("logs", [])
                done_keys = {_log_key(log) for log in existing_logs}
                print(f"[{idx}/{len(target_models)}] {model} -> RESUME ({len(done_keys)} done)")
            except Exception as e:
                print(f"[{idx}/{len(target_models)}] {model} -> checkpoint load failed: {e}, starting fresh")

        if not resume and ckpt_path.exists():
            ckpt_path.unlink()

        experiments = _load_experiments_for_model(model)
        run_msg = f"RUN ({len(done_keys)} done)" if existing_logs else "RUN"
        print(f"[{idx}/{len(target_models)}] {model} -> {run_msg}")

        result = await simulate_model_all_experiments(
            model_name=model,
            dataset=cases,
            experiments=experiments,
            semaphore=semaphore,
            simulator_model=simulator_model,
            max_retries=max_retries,
            checkpoint_path=ckpt_path,
            done_keys=done_keys,
            existing_logs=existing_logs,
            save_every=20,
        )

        model_result = {
            "model_name": model,
            "logs": result.get("logs", []),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_result, f, ensure_ascii=False, indent=2)

        if ckpt_path.exists():
            ckpt_path.unlink()

        print(f"   Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="Target model full name (e.g., openai/o3). Can be repeated.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--mode", choices=["correct", "incorrect"], default="correct")
    parser.add_argument("--simulator_model", required=True)
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")

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
            resume=not args.no_resume,
        )
    )
