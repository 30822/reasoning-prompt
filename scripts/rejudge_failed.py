#!/usr/bin/env python3
"""
Re-judge only logs that used Judge fallback (parse failure → ACCEPT/INVALID).

Pipeline: evaluation only (no simulation, no team).
  - Find logs with brief_reason "Judge error/fallback" in any AI turn
  - Re-run judge for those logs
  - Update annotated_results.json in place
  - Recalculate medcobe

Usage:
  python scripts/rejudge_failed.py --model deepseek/deepseek-r1-0528 [--judge-model google/gemini-3-flash-preview]
"""
import argparse
import asyncio
import json
from pathlib import Path

from src import evaluation as eval_mod
from src.medcobe import calculate_medcobe_scores


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _key(log: dict) -> tuple:
    return (
        str(log.get("case_id")),
        str(log.get("mode")),
        str(log.get("experiment_id") or log.get("experiment_label")),
    )


def _has_judge_fallback(log: dict) -> bool:
    """True if any AI turn has brief_reason 'Judge error/fallback'."""
    for turn in log.get("dialogue", []):
        if turn.get("role") != "AI":
            continue
        reason = (turn.get("brief_reason") or "").strip()
        if "Judge error" in reason or "fallback" in reason.lower():
            return True
    return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge logs that used Judge fallback.")
    parser.add_argument("--model", required=True, help="Target model (e.g. deepseek/deepseek-r1-0528)")
    parser.add_argument("--judge-model", default="google/gemini-3-flash-preview", help="Judge model for re-run")
    parser.add_argument("--dataset-file", default="resources/data/experiments/reproducibility_check_v1.json")
    parser.add_argument("--evaluation-file", default="output/evaluation/annotated_results.json")
    parser.add_argument("--solo-file", default="output/solo/solo_performance_results.json")
    parser.add_argument("--medcobe-output", default="output/medcobe/final_medcobe_evaluation.csv")
    parser.add_argument("--use-openai-judge", action="store_true", help="Use OpenAI API for judge")
    parser.add_argument("--skip-dialogue-level-judge", action="store_true", default=True)
    args = parser.parse_args()

    root = _project_root()
    dataset_path = root / args.dataset_file
    evaluation_path = root / args.evaluation_file
    solo_path = root / args.solo_file
    medcobe_path = root / args.medcobe_output

    if not evaluation_path.exists():
        print(f"Evaluation file not found: {evaluation_path}")
        return

    with open(evaluation_path, "r", encoding="utf-8") as f:
        annotated = json.load(f)

    model_data = annotated.get(args.model)
    if not model_data:
        print(f"Model {args.model} not found in annotated results.")
        return

    logs = model_data.get("logs", [])
    failed_keys = set()
    for log in logs:
        if _has_judge_fallback(log):
            failed_keys.add(_key(log))

    if not failed_keys:
        print("No logs with Judge fallback found. Nothing to re-judge.")
        return

    print(f"Found {len(failed_keys)} logs with Judge fallback. Re-judging...")

    dataset_list = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_dict = {str(item.get("case_id")): item for item in dataset_list}

    ann_idx_map = {_key(log): i for i, log in enumerate(logs)}
    use_openai = args.use_openai_judge or (args.judge_model or "").strip().startswith("openai/")

    for key in sorted(failed_keys):
        case_id, mode, exp_id = key
        idx = ann_idx_map.get(key)
        if idx is None:
            print(f"  [skip] {key}: not in annotated")
            continue
        team_log = logs[idx]
        case_data = dataset_dict.get(case_id)
        if not case_data:
            print(f"  [skip] {key}: case not in dataset")
            continue

        ann_log = await eval_mod.evaluate_and_annotate_dialogue(
            case_data=case_data,
            log_data=team_log,
            judge_model=args.judge_model,
            keep_judge_raw=False,
            keep_turn_level_summary=False,
            keep_dialogue_level=True,
            drop_redundant_team_keys=True,
            use_openai_judge=use_openai,
            skip_dialogue_level_judge=args.skip_dialogue_level_judge,
        )
        logs[idx] = ann_log
        print(f"  Re-judged: case={case_id} mode={mode} exp={exp_id}")

    model_data["logs"] = logs
    annotated[args.model] = model_data

    with open(evaluation_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"Updated: {evaluation_path}")

    calculate_medcobe_scores(
        annotated_file=str(evaluation_path),
        output_csv=str(medcobe_path),
        solo_file=str(solo_path) if solo_path.exists() else None,
    )
    print("MedCOBE recalculated. Done.")


if __name__ == "__main__":
    asyncio.run(main())
