"""
Team performance: reads output/simulation/by_model, writes output/team/by_model.
Uses OpenAI API only (utils.call_llm_openai).
"""

import json
import asyncio
import re
import yaml
from pathlib import Path
from src.utils import call_llm_openai
from tqdm import tqdm


def _get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def _load_prompts() -> dict:
    project_root = _get_project_root()
    prompts_file = project_root / "resources" / "prompt" / "final_decision_prompts.yaml"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _output_root() -> Path:
    return _get_project_root() / "output"

def _team_by_model_dir() -> Path:
    return _output_root() / "team" / "by_model"

def _simulation_by_model_dir() -> Path:
    return _output_root() / "simulation" / "by_model"


PROMPTS = _load_prompts()


def format_options(options_dict: dict) -> str:
    lines = []
    for k, v in options_dict.items():
        lines.append(f"({k}) {v}")
    return "\n".join(lines)


def build_transcript(dialogue: list) -> str:
    parts = []
    for turn in dialogue:
        role = turn.get("role", "").strip()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role.lower() == "doctor":
            parts.append(f"Doctor: {content}")
        elif role.lower() == "ai":
            parts.append(f"AI: {content}")
        else:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def parse_final_answer(text: str) -> str:
    if not text:
        return "Unknown"

    t = text.strip()

    # 가장 강한 패턴 우선
    m = re.search(r"Final\s*Answer\s*:\s*\[\[?\s*([A-D])\s*\]?\]", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"Final\s*Answer\s*:\s*([ABCD])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # fallback: 마지막 줄에서 A-D 추출
    lines = t.splitlines()
    last_line = lines[-1] if lines else t
    m2 = re.search(r"\b([A-D])\b", last_line, flags=re.IGNORECASE)
    return m2.group(1).upper() if m2 else "Unknown"


def parse_reasoning(text: str) -> str:
    t = text.strip()
    m = re.search(r"Reasoning:\s*(.+?)(?=\n\s*Final Answer:|$)", t, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return t[:300]

def load_dataset(dataset_file: str) -> list[dict]:
    project_root = _get_project_root()
    p = Path(dataset_file)
    if not p.is_absolute():
        p = project_root / dataset_file
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


async def recalc_one_log(
    log_entry: dict,
    case_data: dict,
    semaphore: asyncio.Semaphore,
    simulator_model: str,
) -> dict:
    async with semaphore:
        dialogue = log_entry.get("dialogue", [])
        if not dialogue:
            return {
                "final_decision": log_entry.get("final_decision", "Unknown"),
                "reasoning": "No dialogue available",
                "is_team_correct": log_entry.get("is_team_correct", False),
                "decision_reasoning": log_entry.get("decision_reasoning", ""),
            }

        options_text = format_options(case_data.get("options", {}))
        correct_option = ((case_data.get("correct_option") or "").strip()).upper()

        transcript = build_transcript(dialogue)

        simulator_decision_sys = PROMPTS["simulator_base_decision_prompt"]

        user_block = (
            f"Dialogue Transcript:\n{transcript}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Based on the dialogue above, decide whether to CHANGE or MAINTAIN your initial decision."
        )

        messages = [
            {"role": "system", "content": simulator_decision_sys},
            {"role": "user", "content": user_block},
        ]

        raw = await call_llm_openai(
            model_name=simulator_model,
            messages=messages,
            temperature=0.0,
        )
        raw = raw.strip()

        final_letter = parse_final_answer(raw)
        reasoning = parse_reasoning(raw)
        is_team_correct = (final_letter == correct_option)

        return {
            "final_decision": final_letter,
            "reasoning": reasoning,
            "is_team_correct": is_team_correct,
            "decision_reasoning": raw,
        }


async def process_model(
    model_name: str,
    model_data: dict,
    dataset_dict: dict,
    semaphore: asyncio.Semaphore,
    simulator_model: str,
    resume: bool = True,
) -> dict:
    """
    For each log (case_id × mode × experiment_id), run team decision judge and
    annotate:
      - final_decision
      - reasoning
      - is_team_correct
      - decision_reasoning (raw LLM output)

    Keeps experiment_id/label as-is so downstream (medcobe.py) can groupby exp_id
    and produce 16 rows per model.
    """
    logs = model_data.get("logs", [])
    if not logs:
        return model_data

    def _case_key_from_log(log: dict) -> str:
        # dataset_dict is keyed by str(case_id)
        cid = log.get("case_id")
        return "" if cid is None else str(cid)

    def _already_done(log: dict) -> bool:
        # resume: if team judgment already exists, skip
        if not resume:
            return False
        if log.get("is_team_correct") is None:
            return False
        # 최소한 final_decision도 있으면 done으로 간주
        if log.get("final_decision") is None:
            return False
        return True

    # 1) task 생성 (원래 logs 순서를 유지하기 위해 인덱스를 저장)
    tasks: list[tuple[int, asyncio.Future] | None] = []
    valid_mask: list[bool] = []

    for i, log in enumerate(logs):
        if _already_done(log):
            tasks.append(None)
            valid_mask.append(False)  # 그대로 유지
            continue

        case_key = _case_key_from_log(log)
        case_data = dataset_dict.get(case_key)

        if not case_data:
            # case_id 매칭 실패: 로그는 그대로 두되, 디버깅용 마커만 남김
            log["_team_eval_error"] = f"case_id not found in dataset: {case_key}"
            tasks.append(None)
            valid_mask.append(False)
            continue

        tasks.append((i, recalc_one_log(log, case_data, semaphore, simulator_model)))
        valid_mask.append(True)

    # 2) 실행 (동시성: semaphore는 recalc_one_log 내부에서 사용)
    indexed_results: dict[int, dict | Exception] = {}

    coros = [(i, coro) for (i, coro) in tasks if (i is not None and coro is not None)]
    if coros:
        async def _run_one(idx: int, coro):
            try:
                r = await coro
                return idx, r
            except Exception as e:
                return idx, e

        task_objs = [asyncio.create_task(_run_one(idx, c)) for idx, c in coros]

        with tqdm(total=len(task_objs), desc=f"      {model_name}", leave=False, ncols=100) as pbar:
            for fut in asyncio.as_completed(task_objs):
                idx, r = await fut
                indexed_results[idx] = r
                pbar.update(1)

    # 3) 결과 merge (원래 logs 순서 유지)
    updated_logs: list[dict] = []
    for i, log in enumerate(logs):
        # 이미 done/invalid는 원본 유지
        if i not in indexed_results:
            updated_logs.append(log)
            continue

        r = indexed_results[i]
        if isinstance(r, Exception):
            case_id = log.get("case_id", "Unknown")
            mode = log.get("mode", "Unknown")
            exp = log.get("experiment_id", log.get("experiment_label", "Unknown"))
            log["_team_eval_error"] = f"{type(r).__name__}: {r}"
            print(f"    Error team-eval model={model_name} case_id={case_id} mode={mode} exp={exp} -> {type(r).__name__}: {r}")
            updated_logs.append(log)
            continue

        # r is dict from recalc_one_log
        new_log = dict(log)
        new_log["final_decision"] = r.get("final_decision", "Unknown")
        new_log["reasoning"] = r.get("reasoning", "")
        new_log["is_team_correct"] = bool(r.get("is_team_correct", False))
        new_log["decision_reasoning"] = r.get("decision_reasoning", "")
        updated_logs.append(new_log)

    # 4) model-level 요약값
    total_logs = len(updated_logs)
    correct_count = sum(1 for log in updated_logs if log.get("is_team_correct") is True)
    team_accuracy = (correct_count / total_logs) if total_logs else 0.0

    error_mode_logs = [log for log in updated_logs if log.get("mode") == "error"]
    correct_mode_logs = [log for log in updated_logs if log.get("mode") == "correct"]

    correction_power = (
        sum(1 for log in error_mode_logs if log.get("is_team_correct") is True) / len(error_mode_logs)
        if error_mode_logs else 0.0
    )
    confirmation_power = (
        sum(1 for log in correct_mode_logs if log.get("is_team_correct") is True) / len(correct_mode_logs)
        if correct_mode_logs else 0.0
    )

    out = dict(model_data)
    out["model_name"] = model_data.get("model_name", model_name)
    out["logs"] = updated_logs
    out["team_accuracy"] = team_accuracy
    out["correction_power"] = correction_power
    out["confirmation_power"] = confirmation_power
    return out


async def recalculate_final_decision_by_model_dir(
    dataset_file: str,
    simulator_model: str,
    max_concurrent_requests: int = 10,
    resume: bool = True,
):

    project_root = _get_project_root()

    # dataset
    dataset_path = Path(dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)
    dataset_dict = {str(item.get("case_id")): item for item in dataset_list}

    sim_dir = _simulation_by_model_dir()
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation by_model dir not found: {sim_dir}")

    out_dir = _team_by_model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Recalculate Final Decision (by_model)")
    print("=" * 80)
    print(f"Simulation dir:  {sim_dir}")
    print(f"Output dir:      {out_dir}")
    print(f"Dataset:         {dataset_path}")
    print(f"Decision model:  {simulator_model}")
    print(f"Concurrency:     {max_concurrent_requests}")
    print()

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    sim_files = sorted(sim_dir.glob("*.json"))
    if not sim_files:
        raise FileNotFoundError(f"No simulation json files found in: {sim_dir}")

    for idx, sim_path in enumerate(sim_files, 1):
        with open(sim_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        model_name = model_data.get("model_name") or sim_path.stem
        out_path = out_dir / f"{sim_path.stem}.json"

        if resume and out_path.exists():
            print(f"[{idx}/{len(sim_files)}] {model_name} -> SKIP (exists): {out_path}")
            continue

        print(f"[{idx}/{len(sim_files)}] {model_name} -> RUN ({len(model_data.get('logs', []))} logs)")

        updated_model = await process_model(
            model_name=model_name,
            model_data=model_data,
            dataset_dict=dataset_dict,
            semaphore=semaphore,
            simulator_model=simulator_model,
            resume=resume,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(updated_model, f, indent=2, ensure_ascii=False)

        print(f"   Saved: {out_path}")

    # summary (out_dir 파일 기반)
    total_logs = 0
    total_correct = 0
    for p in sorted(out_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        logs = d.get("logs", [])
        total_logs += len(logs)
        total_correct += sum(1 for log in logs if log.get("is_team_correct") is True)

    acc = (total_correct / total_logs * 100.0) if total_logs else 0.0
    print(f"\n[Summary] team results saved to: {out_dir}")
    print(f"[Summary] total_logs={total_logs}, total_correct={total_correct}, team_acc={acc:.2f}%")
