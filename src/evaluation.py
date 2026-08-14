"""Turn-level LLM judge: ARGUE/ACCEPT and VALID/INVALID on each AI utterance."""
import json
import asyncio
import yaml
from pathlib import Path
from src.openrouter_client import get_llm_caller
from tqdm import tqdm
from typing import Any
import re
import os


def _get_project_root():
    return Path(__file__).resolve().parent.parent


def _load_prompts():
    project_root = _get_project_root()
    prompts_file = project_root / "resources" / "prompt" / "evaluator_prompts.yaml"

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    return prompts


PROMPTS = _load_prompts()


def _normalize_action(x: str) -> str:
    x = (x or "").strip().upper()
    if x in {"ARGUE", "ACCEPT"}:
        return x
    if "ARG" in x:
        return "ARGUE"
    if "ACC" in x:
        return "ACCEPT"
    return "ACCEPT"


def _normalize_validity(x: str) -> str:
    x = (x or "").strip().upper()
    if x in {"VALID", "INVALID"}:
        return x
    if "INVAL" in x:
        return "INVALID"
    if "VAL" in x:
        return "VALID"
    return "INVALID"


_ALLOWED_ERROR_TYPES = {
    "MISSED_RISK",
    "HALLUCINATED_RISK",
    "UNDER_SPECIFIED",
    "NONE",
}


def _normalize_error_type(x: str):
    if x is None:
        return ["NONE"]

    if isinstance(x, str):
        items = [x]
    elif isinstance(x, list):
        items = x
    else:
        return ["NONE"]

    norm = []
    for it in items:
        t = (it or "").strip().upper()
        if not t:
            continue
        t = t.replace("-", "_").replace(" ", "_")
        if t in _ALLOWED_ERROR_TYPES:
            norm.append(t)

    if not norm:
        return ["NONE"]

    if "NONE" in norm and len(norm) > 1:
        norm = [t for t in norm if t != "NONE"]

    seen = set()
    out = []
    for t in norm:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _safe_json_load(s: str) -> dict:
    if not s:
        return {}

    s = s.strip().lstrip("\ufeff")

    try:
        return json.loads(s)
    except Exception:
        pass

    for pat in [r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass

    s = re.sub(r"^[^{]*", "", s, count=1)
    s = re.sub(r"[^{]*$", "", s)
    try:
        return json.loads(s.strip())
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    return {}


def _build_dialogue_text(dialogue: list) -> str:
    lines = []
    for turn in dialogue or []:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if content is None:
            content = ""
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _exp_key(log: dict):
    return (log.get("case_id"), log.get("mode"), log.get("experiment_id"))

def _is_empty_utterance(text: str) -> bool:
    if text is None:
        return True
    return len(text.strip()) == 0


_judge_parse_fail_count = 0
_judge_parse_warn_threshold = 5  # print first 5 failures, then every 50th
_JUDGE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["ARGUE", "ACCEPT"]},
                "validity": {"type": "string", "enum": ["VALID", "INVALID"]},
                "brief_reason": {"type": "string"},
                "error_type": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["MISSED_RISK", "HALLUCINATED_RISK", "UNDER_SPECIFIED", "NONE"]},
                },
            },
            "required": ["action", "validity", "brief_reason", "error_type"],
            "additionalProperties": False,
        },
    },
}


def _judge_response_format(judge_model: str) -> dict:
    """json_object for OpenAI; strict json_schema for OpenRouter models."""
    if (judge_model or "").strip().startswith("openai/"):
        return {"type": "json_object"}
    return _JUDGE_JSON_SCHEMA


async def evaluate_single_utterance_combined(
    ai_utterance: str,
    dialogue_context: str,
    case_data: dict,
    judge_model: str,
    keep_judge_raw: bool = False,
    use_openai_judge: bool = False,
) -> dict[str, Any]:
    global _judge_parse_fail_count
    ground_truth_answer = (
        case_data.get("correct_option")
        or case_data.get("answer")
        or case_data.get("ground_truth")
        or ""
    )
    scenario = case_data.get("scenario") or ""
    image_caption = case_data.get("caption") or ""
    rationale = case_data.get("explanation") or ""
    dialogue_context = dialogue_context or ""
    ai_utterance_trunc = ai_utterance or ""

    judge_prompt = PROMPTS["judge_user_prompt"].format(
        scenario=scenario,
        image_caption=image_caption,
        ground_truth_answer=ground_truth_answer,
        rationale=rationale,
        dialogue_context=dialogue_context,
        ai_utterance=ai_utterance_trunc,
    )

    judge_prompt += "\n\nIMPORTANT: Return ONLY a valid JSON object with keys: action, validity, brief_reason, error_type. Do not output any explanation outside JSON."

    if _is_empty_utterance(ai_utterance):
            return {
                "action": "ACCEPT",
                "validity": "INVALID",
                "brief_reason": "No response",
                "error_type": ["NONE"],
            }

    _call = get_llm_caller(judge_model)
    rf = _judge_response_format(judge_model)
    try:
        response_text = await _call(
            model_name=judge_model,
            messages=[
                {"role": "system", "content": PROMPTS["judge_system_prompt"]},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            response_format=rf,
        )

        raw = (response_text or "").strip()
        parsed = _safe_json_load(raw)

        if not parsed or "action" not in parsed or "validity" not in parsed:
            _judge_parse_fail_count += 1
            if _judge_parse_fail_count <= _judge_parse_warn_threshold or _judge_parse_fail_count % 50 == 0:
                print(f"[WARN] Judge parse failure #{_judge_parse_fail_count}. Retrying...")

            scenario_retry = scenario
            rationale_retry = rationale
            dialogue_context_retry = dialogue_context
            ai_retry = ai_utterance_trunc

            judge_prompt_retry = PROMPTS["judge_user_prompt"].format(
                scenario=scenario_retry,
                image_caption=image_caption,
                ground_truth_answer=ground_truth_answer,
                rationale=rationale_retry,
                dialogue_context=dialogue_context_retry,
                ai_utterance=ai_retry,
            )
            judge_prompt_retry += "\n\nIMPORTANT: Return ONLY a valid JSON object with keys: action, validity, brief_reason, error_type. Do not output any explanation outside JSON."

            response_text = await _call(
                model_name=judge_model,
                messages=[
                    {"role": "system", "content": PROMPTS["judge_system_prompt"]},
                    {"role": "user", "content": judge_prompt_retry},
                ],
                temperature=0.0,
                response_format=rf,
            )

            raw = (response_text or "").strip()
            parsed = _safe_json_load(raw)

            if not parsed or "action" not in parsed or "validity" not in parsed:
                raise ValueError("Empty JSON from judge (after retry)")

        action = _normalize_action(parsed.get("action", ""))
        validity = _normalize_validity(parsed.get("validity", ""))
        brief_reason = (parsed.get("brief_reason") or "").strip()
        error_type = _normalize_error_type(parsed.get("error_type"))

        out = {
            "action": action,
            "validity": validity,
            "brief_reason": brief_reason,
            "error_type": error_type,
        }

        if keep_judge_raw:
            out["judge_raw"] = raw
            out["judge_parsed"] = parsed

        return out

    except Exception as e:
        _judge_parse_fail_count += 1
        if _judge_parse_fail_count <= _judge_parse_warn_threshold or _judge_parse_fail_count % 50 == 0:
            print(f"[WARN] Judge error (fallback used) #{_judge_parse_fail_count}: {e}")
        out = {
            "action": "ACCEPT",
            "validity": "INVALID",
            "brief_reason": "Judge error/fallback",
            "error_type": ["NONE"],
        }
        if keep_judge_raw:
            out["judge_raw"] = ""
            out["judge_parsed"] = {}
        return out


def _normalize_dialogue_outcome(x: str) -> str:
    x = (x or "").strip().upper()
    allowed = {"RECOVERED", "NOT_RECOVERED", "REGRESSED", "CONSISTENT_VALID", "CONSISTENT_INVALID"}
    return x if x in allowed else "NOT_RECOVERED"


def _normalize_failure_modes(xs):
    allowed = {
        "NO_CONVERGENCE",
        "GOAL_MISALIGNMENT",
        "OVER_DEFERENCE",
        "OVER_CORRECTION",
        "EVIDENCE_DRIFT",
        "LOW_ACTIONABILITY",
        "NONE",
    }
    if not xs:
        return ["NONE"]
    if isinstance(xs, str):
        xs = [xs]
    out = []
    for x in xs:
        t = (x or "").strip().upper()
        if t in allowed and t not in out:
            out.append(t)
    return out if out else ["NONE"]


def _normalize_quality(x):
    try:
        v = int(x)
        return min(5, max(1, v))
    except Exception:
        return 3


async def evaluate_dialogue_overall(
    dialogue_context: str,
    case_data: dict,
    judge_model: str,
    keep_judge_raw: bool = False,
    use_openai_judge: bool = False,
) -> dict:
    ground_truth_answer = (
        case_data.get("correct_option")
        or case_data.get("answer")
        or case_data.get("ground_truth")
        or ""
    )

    user_prompt = PROMPTS["dialogue_user_prompt"].format(
        scenario=case_data.get("scenario"),
        image_caption=case_data.get("caption"),
        ground_truth_answer=ground_truth_answer,
        rationale=case_data.get("explanation"),
        dialogue_context=dialogue_context,
    )

    _call = get_llm_caller(judge_model)
    try:
        resp_text = await _call(
            model_name=judge_model,
            messages=[
                {"role": "system", "content": PROMPTS["dialogue_system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw = (resp_text or "").strip()
        parsed = _safe_json_load(raw)

        out = {
            "dialogue_outcome": _normalize_dialogue_outcome(parsed.get("dialogue_outcome")),
            "collab_quality": _normalize_quality(parsed.get("collab_quality")),
            "brief_reason": (parsed.get("brief_reason") or "").strip(),
            "dialogue_failure_modes": _normalize_failure_modes(parsed.get("dialogue_failure_modes")),
            "error_type": _normalize_error_type(parsed.get("error_type")),
        }

        if keep_judge_raw:
            out["judge_raw"] = raw
            out["judge_parsed"] = parsed

        return out

    except Exception as e:
        print(f"[WARN] Error in dialogue-level judging: {e}")
        return {
            "dialogue_outcome": "NOT_RECOVERED",
            "collab_quality": 1,
            "brief_reason": "Judge error/fallback",
            "dialogue_failure_modes": ["NONE"],
            "error_type": ["NONE"],
        }


_REDUNDANT_TEAM_KEYS = ("ground_truth", "target_belief", "final_decision", "reasoning", "decision_reasoning")


async def evaluate_and_annotate_dialogue(
    case_data: dict,
    log_data: dict,
    judge_model: str,
    keep_judge_raw: bool = False,
    keep_turn_level_summary: bool = False,
    keep_dialogue_level: bool = True,
    drop_redundant_team_keys: bool = True,
    use_openai_judge: bool = False,
    skip_dialogue_level_judge: bool = False,
) -> dict:
    dialogue = log_data.get("dialogue", [])
    dialogue_text = _build_dialogue_text(dialogue)

    annotated_dialogue = []
    ai_turn_index = 0
    prev_doctor = ""

    for turn in dialogue:
        turn_copy = turn.copy()
        role = turn.get("role")

        if role == "Doctor":
            prev_doctor = (turn.get("content") or "").strip()

        if role == "AI":
            ai_turn_index += 1
            turn_copy["ai_turn_index"] = ai_turn_index

            ai_utterance = (turn.get("content") or "").strip()
            turn_context = f"[Doctor]: {prev_doctor}"

            eval_result = await evaluate_single_utterance_combined(
                ai_utterance=ai_utterance,
                dialogue_context=turn_context,
                case_data=case_data,
                judge_model=judge_model,
                keep_judge_raw=keep_judge_raw,
                use_openai_judge=use_openai_judge,
            )

            turn_copy["action"] = eval_result["action"]
            turn_copy["validity"] = eval_result["validity"]
            turn_copy["brief_reason"] = eval_result["brief_reason"]
            turn_copy["error_type"] = eval_result["error_type"]

            if keep_judge_raw:
                turn_copy["judge_raw"] = eval_result.get("judge_raw", "")
                turn_copy["judge_parsed"] = eval_result.get("judge_parsed", {})

        annotated_dialogue.append(turn_copy)

    annotated_log = log_data.copy()
    annotated_log["dialogue"] = annotated_dialogue

    if keep_turn_level_summary:
        t1 = None
        t2 = None
        for t in annotated_dialogue:
            if t.get("role") == "AI" and t.get("ai_turn_index") == 1:
                t1 = t
            if t.get("role") == "AI" and t.get("ai_turn_index") == 2:
                t2 = t

        annotated_log["turn_level"] = {
            "turn1": None if t1 is None else {
                "action": t1.get("action"),
                "validity": t1.get("validity"),
                "error_type": t1.get("error_type", ["NONE"]),
                "brief_reason": t1.get("brief_reason"),
            },
            "turn2": None if t2 is None else {
                "action": t2.get("action"),
                "validity": t2.get("validity"),
                "error_type": t2.get("error_type", ["NONE"]),
                "brief_reason": t2.get("brief_reason"),
            },
        }
    else:
        annotated_log.pop("turn_level", None)

    if keep_dialogue_level and not skip_dialogue_level_judge:
        overall = await evaluate_dialogue_overall(
            dialogue_context=dialogue_text,
            case_data=case_data,
            judge_model=judge_model,
            keep_judge_raw=keep_judge_raw,
            use_openai_judge=use_openai_judge,
        )
        annotated_log["dialogue_level"] = overall
    else:
        annotated_log.pop("dialogue_level", None)

    if drop_redundant_team_keys:
        for k in _REDUNDANT_TEAM_KEYS:
            annotated_log.pop(k, None)

    return annotated_log


async def run_evaluation(
    team_result_file: str,
    output_file: str,
    dataset_file: str,
    judge_model: str,
    batch_size: int = 20,
    solo_acc_file: str = None,
    keep_judge_raw: bool = False,
    keep_turn_level_summary: bool = False,
    keep_dialogue_level: bool = True,
    drop_redundant_team_keys: bool = True,
    use_openai_judge: bool = False,
    skip_dialogue_level_judge: bool = False,
):
    """Annotate team logs with judge labels and write annotated_results.json."""
    project_root = _get_project_root()

    output_root = project_root / os.environ.get("PCOLLAB_OUTPUT_ROOT", "output")
    output_root.mkdir(parents=True, exist_ok=True)

    team_result_path = Path(team_result_file)
    output_path = Path(output_file)

    team_result_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_file

    if not team_result_path.exists():
        raise FileNotFoundError(f"Team result file not found: {team_result_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"  Loading results...")
    print(f"   Team file: {team_result_path}")
    print(f"   Output file: {output_path}")

    with open(team_result_path, "r", encoding="utf-8") as f:
        team_results = json.load(f)

    print(f"  Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_dataset = {item["case_id"]: item for item in json.load(f)}

    annotated_results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                annotated_results = json.load(f)
            print(f"  Loaded existing results: {len(annotated_results)} models done")
        except Exception:
            annotated_results = {}

    remaining_models = {name: data for name, data in team_results.items() if name not in annotated_results}

    if not remaining_models:
        print("  All models already evaluated!")
        return

    print(f"\n  Starting evaluation for {len(remaining_models)} models...\n")

    for model_idx, (model_name, data) in enumerate(remaining_models.items(), 1):
        print(f"[{model_idx}/{len(remaining_models)}]   Judging model: {model_name}")

        logs = data.get("logs", [])
        if not logs:
            annotated_results[model_name] = data.copy()
            continue

        tasks = []
        for log in logs:
            case_id = log.get("case_id")
            case_info = raw_dataset.get(case_id)
            if not case_info:
                continue
            tasks.append(
                evaluate_and_annotate_dialogue(
                    case_data=case_info,
                    log_data=log,
                    judge_model=judge_model,
                    keep_judge_raw=keep_judge_raw,
                    keep_turn_level_summary=keep_turn_level_summary,
                    keep_dialogue_level=keep_dialogue_level,
                    drop_redundant_team_keys=drop_redundant_team_keys,
                    use_openai_judge=use_openai_judge,
                    skip_dialogue_level_judge=skip_dialogue_level_judge,
                )
            )

        if not tasks:
            annotated_results[model_name] = data.copy()
            continue

        annotated_logs = []

        with tqdm(total=len(tasks), desc=f"   Evaluating {model_name}", unit="dialogue", ncols=100) as pbar:
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch)
                annotated_logs.extend(batch_results)
                pbar.update(len(batch))

        annotated_model_data = data.copy()
        annotated_logs_dict = {_exp_key(log): log for log in annotated_logs}

        updated_logs = []
        for original_log in logs:
            key = (original_log.get("case_id"), original_log.get("mode"), original_log.get("experiment_id"))
            if key in annotated_logs_dict:
                updated_logs.append(annotated_logs_dict[key])
            else:
                updated_logs.append(original_log)

        annotated_model_data["logs"] = updated_logs
        annotated_results[model_name] = annotated_model_data

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotated_results, f, indent=2, ensure_ascii=False)
        print("    Saved.")

    print(f"\n  Evaluation complete! Saved to {output_path}")
