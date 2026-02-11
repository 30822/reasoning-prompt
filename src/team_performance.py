import json
import asyncio
import re
import yaml
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm

from src.utils import get_openai_key


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


PROMPTS = _load_prompts()


async def call_llm(
    model_name: str, 
    messages: list, 
    client: AsyncOpenAI,
    temperature: float = 0.0
) -> str:
    resp = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

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
    t = text.strip()

    m = re.search(r"Final Answer:\s*\[\[?\s*([A-D])\s*\]?\]", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    last_line = t.splitlines()[-1] if t.splitlines() else t
    m2 = re.search(r"\b([A-D])\b", last_line, re.IGNORECASE)
    return m2.group(1).upper() if m2 else "Unknown"


def parse_reasoning(text: str) -> str:
    t = text.strip()
    m = re.search(r"Reasoning:\s*(.+?)(?=\n\s*Final Answer:|$)", t, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return t[:300]


async def recalc_one_log(
    log_entry: dict, 
    case_data: dict, 
    semaphore: asyncio.Semaphore,
    simulator_model: str,
    client: AsyncOpenAI
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
        correct_option = (case_data.get("correct_option") or "").strip()

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

        raw = await call_llm(simulator_model, messages, client, temperature=0.0)
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
    client: AsyncOpenAI
) -> dict:
    logs = model_data.get("logs", [])
    if not logs:
        return model_data

    tasks = []
    valid_mask = [] 
    for log in logs:
        case_id = log.get("case_id")
        case_data = dataset_dict.get(case_id)
        if not case_data:
            tasks.append(None)
            valid_mask.append(False)
        else:
            tasks.append(recalc_one_log(log, case_data, semaphore, simulator_model, client))
            valid_mask.append(True)

    coros = [t for t in tasks if t is not None]
    results = []
    if coros:
        async def run_one(i, coro):
            try:
                return i, await coro
            except Exception as e:
                return i, e
        
        task_objs = [asyncio.create_task(run_one(i, c)) for i, c in enumerate(coros)]
        
        indexed_results = {}
        with tqdm(total=len(task_objs), desc=f"      {model_name}", leave=False, ncols=100) as pbar:
            for fut in asyncio.as_completed(task_objs):
                i, r = await fut
                indexed_results[i] = r
                pbar.update(1)

        results = [indexed_results[i] for i in sorted(indexed_results.keys())]

    updated_logs = []
    res_idx = 0
    for i, log in enumerate(logs):
        if not valid_mask[i]:
            updated_logs.append(log)
            continue

        r = results[res_idx]
        res_idx += 1

        if isinstance(r, Exception):
            case_id = log.get("case_id", "Unknown")
            mode = log.get("mode", "Unknown")
            print(f"    Error processing case_id={case_id}, mode={mode}: {type(r).__name__}: {r}")
            updated_logs.append(log)
            continue

        new_log = dict(log)
        new_log["final_decision"] = r["final_decision"]
        new_log["reasoning"] = r["reasoning"]
        new_log["is_team_correct"] = r["is_team_correct"]
        new_log["decision_reasoning"] = r["decision_reasoning"]
        updated_logs.append(new_log)

    total_logs = len(updated_logs)
    correct_count = sum(1 for log in updated_logs if log.get("is_team_correct", False))
    team_accuracy = correct_count / total_logs if total_logs > 0 else 0.0
    
    error_mode_logs = [log for log in updated_logs if log.get("mode") == "error"]
    correct_mode_logs = [log for log in updated_logs if log.get("mode") == "correct"]
    
    correction_power = sum(1 for log in error_mode_logs if log.get("is_team_correct", False)) / len(error_mode_logs) if error_mode_logs else 0.0
    confirmation_power = sum(1 for log in correct_mode_logs if log.get("is_team_correct", False)) / len(correct_mode_logs) if correct_mode_logs else 0.0

    out = dict(model_data)
    out["logs"] = updated_logs
    out["team_accuracy"] = team_accuracy
    out["correction_power"] = correction_power
    out["confirmation_power"] = confirmation_power
    return out


async def recalculate_final_decision(
    team_result_file: str,
    dataset_file: str = None,
    simulator_model: str = "gpt-4o",
    max_concurrent_requests: int = 10,
    csv_file: str = None
):
    project_root = _get_project_root()
    
    # 파일 경로 처리
    team_result_path = Path(team_result_file)
    if not team_result_path.is_absolute():
        team_result_path = project_root / team_result_file
    
    if not team_result_path.exists():
        raise FileNotFoundError(f"Team result file not found: {team_result_path}")
    
    # dataset path
    if dataset_file:
        dataset_path = Path(dataset_file)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_file
    else:
        filename = team_result_path.stem
        m = re.search(r"team_collaboration_results_(.+)", filename)
        if m:
            domain = m.group(1)
            dataset_path = project_root / "Jama" / f"jama_raw_{domain}.json"
        else:
            raise ValueError("Cannot infer dataset file. Please pass dataset_file explicitly.")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # CSV 파일 경로 처리
    csv_path = None
    if csv_file:
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = project_root / csv_file
        if not csv_path.exists():
            print(f"⚠️  Warning: CSV file not found: {csv_path}")
            print("   Team accuracy will not be updated in CSV.")
            csv_path = None
    
    print("=" * 80)
    print("Recalculate Final Decision (order-safe)")
    print("=" * 80)
    print(f"Team file: {team_result_path}")
    print(f"Dataset:  {dataset_path}")
    print(f"Decision model: {simulator_model}")
    print(f"Concurrency: {max_concurrent_requests}")
    if csv_path:
        print(f"CSV file: {csv_path}")
    print()
    
    with open(team_result_path, "r", encoding="utf-8") as f:
        team_results = json.load(f)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)
    
    dataset_dict = {str(item.get("case_id")): item for item in dataset_list}
    
    client = AsyncOpenAI(api_key=get_openai_key())
    
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    updated_results = {}
    model_items = list(team_results.items())
    
    # 모델별 team accuracy 저장
    model_team_accuracies = {}
    
    for idx, (model_name, model_data) in enumerate(model_items, 1):
        logs = model_data.get("logs", [])
        print(f"[{idx}/{len(model_items)}] {model_name} ({len(logs)} logs)")
        
        updated_model = await process_model(
            model_name, 
            model_data, 
            dataset_dict, 
            semaphore,
            simulator_model,
            client
        )
        updated_results[model_name] = updated_model
        model_team_accuracies[model_name] = updated_model.get("team_accuracy", 0.0)
    
    # JSON 파일 저장
    with open(team_result_path, "w", encoding="utf-8") as f:
        json.dump(updated_results, f, indent=2, ensure_ascii=False)
    
    # CSV 파일 업데이트
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            
            # "Team Accuracy" 열이 없으면 생성
            if "Team Accuracy" not in df.columns:
                df["Team Accuracy"] = float("nan")
            
            # 각 모델의 team accuracy 업데이트
            updated_count = 0
            for model_name, team_acc in model_team_accuracies.items():
                mask = df["Model"] == model_name
                if mask.any():
                    df.loc[mask, "Team Accuracy"] = team_acc
                    updated_count += 1
                else:
                    print(f"   ⚠️  Model '{model_name}' not found in CSV")
            
            # CSV 저장 (NaN은 'NaN' 문자열로 저장)
            df.to_csv(csv_path, index=False, na_rep='NaN')
            print(f"\n   ✅ Updated Team Accuracy for {updated_count} model(s) in CSV: {csv_path}")
        except Exception as e:
            print(f"   ❌ Error updating CSV file: {e}")
    
    total_logs = sum(len(d.get("logs", [])) for d in updated_results.values())
    total_correct = sum(
        sum(1 for log in d.get("logs", []) if log.get("is_team_correct"))
        for d in updated_results.values()
    )
    acc = (total_correct / total_logs * 100.0) if total_logs else 0.0