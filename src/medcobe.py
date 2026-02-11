import json
import pandas as pd
import math
from pathlib import Path


def _get_project_root():
    """return the project root"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


_T1_ORDER = ["B", "B_COT", "B_CL", "B_COT_CL"]
_T2_ORDER = ["B", "B_CL", "B_SR", "B_CL_SR"]

def exp_id_to_cid(experiment_id: str) -> str:
    """
    experiment_id 예: 'o1-B_COT__B_CL_SR'
    -> C8 (row=B_COT, col=B_CL_SR)
    """
    try:
        rhs = experiment_id.split("-", 1)[1]
        t1, t2 = rhs.split("__", 1)
        r = _T1_ORDER.index(t1)
        c = _T2_ORDER.index(t2)
        return f"C{r*4 + c + 1}"
    except Exception:
        return "C?"


def get_all_ai_utterances(dialogue):
    ai_utterances = []
    for turn in (dialogue or []):
        if turn.get("role") == "AI":
            action = turn.get("action") or turn.get("ai_action")
            validity = turn.get("validity") or turn.get("reasoning_validity")
            if action is not None and validity is not None:
                ai_utterances.append((str(action).upper(), str(validity).upper()))
    return ai_utterances


def calculate_medcobe_scores(
    annotated_file: str,
    output_csv: str,
    solo_file: str = None
):
    project_root = _get_project_root()

    annotated_path = Path(annotated_file)
    output_path = Path(output_csv)

    if not annotated_path.exists():
        raise FileNotFoundError(f"Annotated file not found: {annotated_path}")
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    solo_path = None
    if solo_file:
        solo_path = Path(solo_file)
        if not solo_path.is_absolute():
            solo_path = project_root / solo_file
    
    if not annotated_path.exists():
        raise FileNotFoundError(f"Annotated JSON file not found: {annotated_path}")
    
    print(f"  Loading annotated JSON file: {annotated_path}")
    with open(annotated_path, "r", encoding="utf-8") as f:
        annotated_data = json.load(f)
    
    solo_data = {}
    if solo_path is not None and solo_path.exists():
        print(f"  Loading solo performance file: {solo_path}")
        with open(solo_path, "r", encoding="utf-8") as f:
            solo_data = json.load(f)
    else:
        if solo_path is not None:
            print(f"   Solo performance file not found: {solo_path}")
        else:
            print(f"   Solo performance file not provided")
        print(f"   Continuing without solo accuracy data (will use NaN)")
    
    rows = []

    for model_name, model_data in annotated_data.items():
        logs = model_data.get("logs", [])
        if not logs:
            continue

        # 1) logs를 experiment_id(없으면 label)로 groupby
        grouped = {}
        for log in logs:
            exp_id = log.get("experiment_id") or log.get("experiment_label") or "UNKNOWN_EXPERIMENT"
            grouped.setdefault(exp_id, []).append(log)

        # model-level accuracy는 model_data에 있을 수도/없을 수도 있음 (지금은 유지)
        team_acc_model_level = safe_float(model_data.get("team_accuracy"))

        # solo acc는 model_name 키로 찾음 (현재 유지)
        solo_acc = float("nan")
        if model_name in solo_data:
            solo_acc = safe_float(solo_data[model_name].get("accuracy"))

        # 2) experiment_id 그룹마다 1행 생성
        for exp_id, exp_logs in grouped.items():
            total_ai_utterances_error = 0
            total_ai_utterances_correct = 0
            success_argue_strict = 0
            success_accept = 0

            # (옵션) team accuracy를 experiment 단위로 계산하고 싶으면 여기서 계산 가능
            # 현재 team_results log에 is_team_correct가 있으니 exp 단위 accuracy 계산 가능
            exp_team_correct = 0
            exp_team_total = 0

            for log in exp_logs:
                mode = log.get("mode")
                dialogue = log.get("dialogue", [])

                # team acc per experiment (선택)
                if isinstance(log.get("is_team_correct"), bool):
                    exp_team_total += 1
                    if log["is_team_correct"]:
                        exp_team_correct += 1

                ai_utterances = get_all_ai_utterances(dialogue)
                if len(ai_utterances) == 0:
                    continue

                if mode == "error":
                    total_ai_utterances_error += len(ai_utterances)
                    for action, validity in ai_utterances:
                        if action == "ARGUE" and validity == "VALID":
                            success_argue_strict += 1

                elif mode == "correct":
                    total_ai_utterances_correct += len(ai_utterances)
                    for action, validity in ai_utterances:
                        if action == "ACCEPT" and validity == "VALID":
                            success_accept += 1

            # recall 계산
            recall_correction = (success_argue_strict / total_ai_utterances_error) if total_ai_utterances_error > 0 else float("nan")
            recall_confirmation = (success_accept / total_ai_utterances_correct) if total_ai_utterances_correct > 0 else float("nan")

            # medcobe score
            if pd.notna(recall_correction) and pd.notna(recall_confirmation) and recall_correction > 0 and recall_confirmation > 0:
                medcobe_score = math.sqrt(recall_correction * recall_confirmation)
            else:
                medcobe_score = float("nan")

            # experiment 단위 team acc (가능할 때만)
            team_acc_exp = float("nan")
            if exp_team_total > 0:
                team_acc_exp = exp_team_correct / exp_team_total

            c_id = exp_id_to_cid(exp_id)

            rows.append({
                "Model": model_name,
                "Experiment": c_id,
                "N_dialogues_judged": len(exp_logs),
                "Solo Accuracy": solo_acc,
                "Team Accuracy": team_acc_exp if pd.notna(team_acc_exp) else team_acc_model_level,
                "Recall (Correction)": recall_correction,
                "Recall (Confirmation)": recall_confirmation,
                "MedCOBE Score": medcobe_score,
            })
    
    out_df = pd.DataFrame(rows)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_df.to_csv(output_path, index=False, na_rep='NaN')
    
    print(f"{total_ai_utterances_error}")
    print(f"{success_argue_strict}")
    print(f"{recall_correction}")
    print("\n  MedCOBE Scores calculated and saved")
    print(f"   Annotated JSON file: {annotated_path}")
    print(f"   Solo performance file: {solo_path}")
    print(f"   Output CSV: {output_path}")
    print(f"\n  Calculated metrics for {len(rows)} models")