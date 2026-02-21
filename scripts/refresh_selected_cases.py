#!/usr/bin/env python3
"""
Refresh selected logs in-place and recalculate downstream outputs.

Pipeline:
  simulation/by_model/<model>.json  (selected logs only rerun)
    -> team/by_model/<model>.json   (selected logs only re-judged)
    -> evaluation/annotated_results.json (selected logs only re-annotated)
    -> medcobe/final_medcobe_evaluation.csv (full recompute)

Example:
  python scripts/refresh_selected_cases.py ^
    --model deepseek/deepseek-r1-0528 ^
    --case-id 5103 ^
    --mode correct ^
    --experiment B__B_CL_SR
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from src import evaluation as eval_mod
from src import simulation_openrouter as sim_mod
from src import team_performance_openrouter as team_mod
from src.medcobe import calculate_medcobe_scores


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _key(log: dict) -> tuple[str, str, str]:
    return (
        str(log.get("case_id")),
        str(log.get("mode")),
        str(log.get("experiment_id") or log.get("experiment_label")),
    )


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _split_csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for v in values:
        parts = [x.strip() for x in str(v).split(",") if x.strip()]
        out.extend(parts)
    return out


def _resolve_experiment_filter(
    all_experiment_ids: list[str],
    wanted_tokens: list[str],
) -> set[str]:
    if not wanted_tokens:
        return set(all_experiment_ids)

    resolved: set[str] = set()
    for token in wanted_tokens:
        token = token.strip()
        if not token:
            continue
        if token in all_experiment_ids:
            resolved.add(token)
            continue
        # allow shorthand like B_COT__B or r1_0528-B_COT__B
        for exp_id in all_experiment_ids:
            if exp_id.endswith(token):
                resolved.add(exp_id)

    return resolved


def _recompute_team_metrics(logs: list[dict]) -> tuple[float, float, float]:
    total_logs = len(logs)
    correct_count = sum(1 for log in logs if log.get("is_team_correct") is True)
    team_accuracy = (correct_count / total_logs) if total_logs else 0.0

    error_mode_logs = [log for log in logs if log.get("mode") == "error"]
    correct_mode_logs = [log for log in logs if log.get("mode") == "correct"]

    correction_power = (
        sum(1 for log in error_mode_logs if log.get("is_team_correct") is True) / len(error_mode_logs)
        if error_mode_logs else 0.0
    )
    confirmation_power = (
        sum(1 for log in correct_mode_logs if log.get("is_team_correct") is True) / len(correct_mode_logs)
        if correct_mode_logs else 0.0
    )
    return team_accuracy, correction_power, confirmation_power


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh selected case/mode/experiment logs and recalculate MedCOBE."
    )
    parser.add_argument("--model", required=True, help="Target model name (e.g. deepseek/deepseek-r1-0528)")
    parser.add_argument("--case-id", action="append", default=[], help="Case ID (repeatable or comma-separated)")
    parser.add_argument("--mode", action="append", choices=["error", "correct"], default=[],
                        help="Mode filter (repeatable): error/correct")
    parser.add_argument("--experiment", action="append", default=[],
                        help="Experiment filter (full id or suffix like B_COT__B)")

    parser.add_argument("--dataset-file", default="resources/data/experiments/reproducibility_check_v1.json")
    parser.add_argument("--simulator-model", default="openai/gpt-4o")
    parser.add_argument("--judge-model", default="openai/gpt-5.2")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=6)

    parser.add_argument("--use-openai-judge", dest="use_openai_judge", action="store_true")
    parser.add_argument("--use-openrouter-judge", dest="use_openai_judge", action="store_false")
    parser.add_argument("--skip-dialogue-level-judge", dest="skip_dialogue_level_judge", action="store_true")
    parser.add_argument("--run-dialogue-level-judge", dest="skip_dialogue_level_judge", action="store_false")
    parser.set_defaults(use_openai_judge=True, skip_dialogue_level_judge=True)

    parser.add_argument("--sim-by-model", default="output/simulation/by_model")
    parser.add_argument("--sim-combined", default="output/simulation/simulator_results.json")
    parser.add_argument("--team-by-model", default="output/team/by_model")
    parser.add_argument("--team-combined", default="output/team/team_results.json")
    parser.add_argument("--evaluation-file", default="output/evaluation/annotated_results.json")
    parser.add_argument("--solo-file", default="output/solo/solo_performance_results.json")
    parser.add_argument("--medcobe-output", default="output/medcobe/final_medcobe_evaluation.csv")

    args = parser.parse_args()

    root = _project_root()
    model_name = args.model
    model_stem = _sanitize_filename(model_name)

    case_ids = set(_split_csv_values(args.case_id))
    modes = set(_split_csv_values(args.mode))
    experiment_tokens = _split_csv_values(args.experiment)

    dataset_path = (root / args.dataset_file).resolve()
    sim_by_model_path = (root / args.sim_by_model / f"{model_stem}.json").resolve()
    team_by_model_path = (root / args.team_by_model / f"{model_stem}.json").resolve()
    sim_combined_path = (root / args.sim_combined).resolve()
    team_combined_path = (root / args.team_combined).resolve()
    evaluation_path = (root / args.evaluation_file).resolve()
    solo_path = (root / args.solo_file).resolve()
    medcobe_out_path = (root / args.medcobe_output).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not sim_by_model_path.exists():
        raise FileNotFoundError(f"Simulation by_model file not found: {sim_by_model_path}")

    dataset_list = _load_json(dataset_path)
    dataset_dict = {str(item.get("case_id")): item for item in dataset_list}

    experiments = sim_mod._load_experiments_for_model(model_name)
    exp_map = {e["experiment_id"]: e for e in experiments}
    selected_experiments = _resolve_experiment_filter(list(exp_map.keys()), experiment_tokens)

    sim_model_data = _load_json(sim_by_model_path)
    sim_logs = sim_model_data.get("logs", [])

    sim_idx_map = {_key(log): i for i, log in enumerate(sim_logs)}

    selected_items: list[tuple[int | None, tuple[str, str, str]]] = []
    for i, log in enumerate(sim_logs):
        k = _key(log)
        case_id, mode, exp_id = k
        if case_ids and case_id not in case_ids:
            continue
        if modes and mode not in modes:
            continue
        if selected_experiments and exp_id not in selected_experiments:
            continue
        selected_items.append((i, k))

    # If no existing match, allow creating missing logs from filters.
    if not selected_items:
        if not case_ids or not modes or not selected_experiments:
            print("No matching logs found. To create missing logs, specify --case-id, --mode, --experiment together.")
            return
        for cid in sorted(case_ids):
            for mode in sorted(modes):
                for exp_id in sorted(selected_experiments):
                    k = (cid, mode, exp_id)
                    existing_i = sim_idx_map.get(k)
                    selected_items.append((existing_i, k))
        print(f"No existing match; creating/updating {len(selected_items)} logs from filters.")

    print(f"Selected {len(selected_items)} logs for refresh (model={model_name})")
    sim_semaphore = asyncio.Semaphore(args.max_concurrent)

    async def _rerun_simulation_key(existing_idx: int | None, key: tuple[str, str, str]):
        case_data = dataset_dict.get(key[0])
        exp = exp_map.get(key[2])
        if not case_data:
            return existing_idx, key, None, f"case_id missing in dataset: {key[0]}"
        if not exp:
            return existing_idx, key, None, f"experiment missing in experiments.yaml: {key[2]}"
        new_log = await sim_mod.run_single_simulation(
            target_model=model_name,
            experiment=exp,
            case=case_data,
            mode=key[1],
            semaphore=sim_semaphore,
            simulator_model=args.simulator_model,
            max_retries=args.max_retries,
        )
        if not new_log:
            return existing_idx, key, None, "run_single_simulation returned None"
        return existing_idx, key, new_log, None

    rerun_tasks = [_rerun_simulation_key(i, k) for i, k in selected_items]
    rerun_results = await asyncio.gather(*rerun_tasks)

    refreshed_sim_keys: set[tuple[str, str, str]] = set()
    for existing_idx, key, new_log, err in rerun_results:
        if err:
            print(f"[WARN] simulation refresh skipped key={key}: {err}")
            continue
        if existing_idx is None:
            sim_logs.append(new_log)
        else:
            sim_logs[existing_idx] = new_log
        refreshed_sim_keys.add(_key(new_log))

    if not refreshed_sim_keys:
        print("No simulation logs were refreshed successfully.")
        return

    sim_model_data["logs"] = sim_logs
    _save_json(sim_by_model_path, sim_model_data)
    print(f"Updated simulation by_model: {sim_by_model_path}")

    if sim_combined_path.exists():
        sim_combined = _load_json(sim_combined_path)
        if isinstance(sim_combined, dict):
            sim_combined[model_name] = sim_model_data
            _save_json(sim_combined_path, sim_combined)
            print(f"Updated simulation combined: {sim_combined_path}")

    # ---- Team refresh for selected logs ----
    if team_by_model_path.exists():
        team_model_data = _load_json(team_by_model_path)
    else:
        team_model_data = {"model_name": model_name, "logs": list(sim_logs)}

    team_logs = team_model_data.get("logs", [])
    team_idx_map = {_key(log): i for i, log in enumerate(team_logs)}
    sim_idx_map = {_key(log): i for i, log in enumerate(sim_logs)}

    team_semaphore = asyncio.Semaphore(args.max_concurrent)
    refreshed_team_keys: set[tuple[str, str, str]] = set()

    for key in sorted(refreshed_sim_keys):
        sim_log = sim_logs[sim_idx_map[key]]

        if key in team_idx_map:
            i = team_idx_map[key]
            merged_log = dict(team_logs[i])
            # force-refresh simulation-origin fields
            merged_log["dialogue"] = sim_log.get("dialogue", [])
            merged_log["ground_truth"] = sim_log.get("ground_truth")
            merged_log["target_belief"] = sim_log.get("target_belief")
            merged_log["experiment_id"] = sim_log.get("experiment_id")
            merged_log["experiment_label"] = sim_log.get("experiment_label")
            merged_log["mode"] = sim_log.get("mode")
            merged_log["case_id"] = sim_log.get("case_id")
            team_logs[i] = merged_log
            log_for_team = merged_log
        else:
            log_for_team = dict(sim_log)
            team_logs.append(log_for_team)
            team_idx_map[key] = len(team_logs) - 1

        case_data = dataset_dict.get(key[0])
        if not case_data:
            print(f"[WARN] team refresh skipped {key}: case not found")
            continue

        try:
            team_result = await team_mod.recalc_one_log(
                log_entry=log_for_team,
                case_data=case_data,
                semaphore=team_semaphore,
                simulator_model=args.simulator_model,
            )
            i = team_idx_map[key]
            new_team_log = dict(team_logs[i])
            new_team_log["final_decision"] = team_result.get("final_decision", "Unknown")
            new_team_log["reasoning"] = team_result.get("reasoning", "")
            new_team_log["is_team_correct"] = bool(team_result.get("is_team_correct", False))
            new_team_log["decision_reasoning"] = team_result.get("decision_reasoning", "")
            new_team_log.pop("_team_eval_error", None)
            team_logs[i] = new_team_log
            refreshed_team_keys.add(key)
        except Exception as e:
            print(f"[WARN] team refresh failed {key}: {e}")

    team_model_data["model_name"] = model_name
    team_model_data["logs"] = team_logs
    team_acc, corr_power, conf_power = _recompute_team_metrics(team_logs)
    team_model_data["team_accuracy"] = team_acc
    team_model_data["correction_power"] = corr_power
    team_model_data["confirmation_power"] = conf_power

    _save_json(team_by_model_path, team_model_data)
    print(f"Updated team by_model: {team_by_model_path}")

    if team_combined_path.exists():
        team_combined = _load_json(team_combined_path)
        if isinstance(team_combined, dict):
            team_combined[model_name] = team_model_data
            _save_json(team_combined_path, team_combined)
            print(f"Updated team combined: {team_combined_path}")

    # ---- Evaluation refresh for selected logs ----
    if evaluation_path.exists():
        annotated = _load_json(evaluation_path)
    else:
        annotated = {}

    ann_model_data = dict(annotated.get(model_name, {"model_name": model_name, "logs": []}))
    ann_logs = list(ann_model_data.get("logs", []))
    ann_idx_map = {_key(log): i for i, log in enumerate(ann_logs)}
    team_idx_map = {_key(log): i for i, log in enumerate(team_logs)}

    for key in sorted(refreshed_team_keys):
        team_log = team_logs[team_idx_map[key]]
        case_data = dataset_dict.get(key[0])
        if not case_data:
            print(f"[WARN] evaluation refresh skipped {key}: case not found")
            continue

        ann_log = await eval_mod.evaluate_and_annotate_dialogue(
            case_data=case_data,
            log_data=team_log,
            judge_model=args.judge_model,
            keep_judge_raw=False,
            keep_turn_level_summary=False,
            keep_dialogue_level=True,
            drop_redundant_team_keys=True,
            use_openai_judge=args.use_openai_judge,
            skip_dialogue_level_judge=args.skip_dialogue_level_judge,
        )

        if key in ann_idx_map:
            ann_logs[ann_idx_map[key]] = ann_log
        else:
            ann_logs.append(ann_log)
            ann_idx_map[key] = len(ann_logs) - 1

    ann_model_data["logs"] = ann_logs
    # keep updated team-level summary if available
    ann_model_data["team_accuracy"] = team_model_data.get("team_accuracy")
    ann_model_data["correction_power"] = team_model_data.get("correction_power")
    ann_model_data["confirmation_power"] = team_model_data.get("confirmation_power")
    annotated[model_name] = ann_model_data

    _save_json(evaluation_path, annotated)
    print(f"Updated evaluation file: {evaluation_path}")

    # ---- Recalculate MedCOBE ----
    calculate_medcobe_scores(
        annotated_file=str(evaluation_path),
        output_csv=str(medcobe_out_path),
        solo_file=str(solo_path),
    )
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
