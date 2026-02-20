"""
Solo performance: by_model output to output/solo/by_model. Uses OpenAI API only (utils.call_llm_openai).
"""

import json
import re
import asyncio
from pathlib import Path
from src.utils import call_llm_openai
from tqdm import tqdm
import time


def _get_project_root():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def format_options(options_dict):
    text = ""
    for key, value in options_dict.items():
        text += f"({key}) {value}\n"
    return text

def extract_answer_key(text: str) -> str:
    if not text:
        return "Unknown"

    t = text.strip()

    patterns = [
        r"Final\s*Answer\s*:\s*([A-D])",
        r"Answer\s*:\s*([A-D])",
        r"^([A-D])$",
        r"^\(?([A-D])\)?$",
    ]

    for p in patterns:
        m = re.search(p, t, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    return "Unknown"


async def solve_case_solo_async(
    model_name,
    case,
    case_index,
    semaphore,
):
    async with semaphore:
        scenario = case['scenario']
        options_text = format_options(case['options'])

        system_prompt = """
        You are a clinical AI assistant.
        Select the single best answer based on the given case.
        Return ONLY the option letter (A, B, C, or D).
        Do NOT explain.
        """

        user_prompt = f"""
        [Clinical Case]
        {scenario}

        [Options]
        {options_text}

        Question: What is the most appropriate next step or diagnosis?
        """

        try:
            content = await call_llm_openai(
                model_name=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = content.strip()

            prediction = extract_answer_key(content)

            return {
                "raw_response": content,
                "prediction": prediction,
                "is_correct": prediction == case['correct_option'],
                "case_id": case.get("case_id"),
                "case_index": case_index
            }

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found" in error_msg.lower() or "invalid model" in error_msg.lower():
                print(f"  Error with {model_name} (case {case.get('case_id', 'unknown')}): Model not found or invalid")
            else:
                print(f"Error with {model_name} (case {case.get('case_id', 'unknown')}): {e}")
            return None


async def evaluate_model_async(
    model_name,
    dataset,
    semaphore,
):
    tasks = [
        solve_case_solo_async(model_name, case, idx, semaphore)
        for idx, case in enumerate(dataset)
    ]

    results = []
    with tqdm(total=len(tasks), desc=f"Evaluating {model_name}") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
            pbar.update(1)

    results.sort(key=lambda x: x.get('case_index', 0))
    return results


async def run_solo_performance(
    input_file: str,
    output_file: str,
    models_to_test: list,
    max_concurrent_requests: int = 10,
):

    project_root = _get_project_root()

    output_root = project_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    solo_dir = output_root / "solo" / "by_model"
    solo_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_file

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = output_root / "solo" / output_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"  Loading data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Starting Solo Performance Evaluation on {len(dataset)} cases...")
    print(f"Using async parallel processing (max {max_concurrent_requests} concurrent requests)")
    print("")

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    for model in models_to_test:
        out_path = solo_dir / f"{model.replace('/', '_')}.json"
        if out_path.exists():
            print(f"SKIP (exists): {out_path}")
            continue

        print(f"\nEvaluating Model: {model}")
        start_time = time.time()

        results = await evaluate_model_async(model, dataset, semaphore)

        model_logs = []
        correct_count = 0
        dataset_dict = {str(c["case_id"]): c for c in dataset}

        for result in results:
            if result is None:
                model_logs.append({
                    "case_id": None,
                    "experiment_id": "SOLO",
                    "prediction": "Unknown",
                    "is_correct": False,
                    "raw_response": "",
                    "_error": "LLM call failed"
                })
                continue

            idx = result['case_index']
            case = dataset_dict.get(str(result["case_id"]))
            log_entry = {
                "case_id": case.get("case_id"),
                "ground_truth": case['correct_option'],
                "prediction": result['prediction'],
                "is_correct": result['is_correct'],
                "raw_response": result['raw_response'],
                "experiment_id": "SOLO"
            }
            model_logs.append(log_entry)
            if result['is_correct']:
                correct_count += 1

        elapsed_time = time.time() - start_time
        accuracy = correct_count / len(dataset) if dataset else 0
        print(f"Model: {model} | Accuracy: {accuracy:.2%} | Time: {elapsed_time:.1f}s")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model,
                    "accuracy": accuracy,
                    "correct_count": correct_count,
                    "total_cases": len(dataset),
                    "logs": model_logs,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved: {out_path}")
