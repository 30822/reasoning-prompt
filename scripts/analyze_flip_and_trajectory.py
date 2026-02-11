import json
from pathlib import Path
import csv

def iter_ai_turns(dialogue):
    """Yield AI turns in order with ai_turn_index if present."""
    ai_turns = []
    for t in dialogue or []:
        if t.get("role") == "AI":
            ai_turns.append(t)
    # fallback: set ai_turn_index if missing
    for i, t in enumerate(ai_turns, 1):
        if "ai_turn_index" not in t:
            t["ai_turn_index"] = i
    return ai_turns

def classify_flip(t1, t2):
    """
    Flip definition examples:
      - action flip: ARGUE -> ACCEPT or ACCEPT -> ARGUE
      - validity flip: VALID -> INVALID etc (optional)
    Return dict flags.
    """
    a1, a2 = t1.get("action"), t2.get("action")
    v1, v2 = t1.get("validity"), t2.get("validity")

    return {
        "action_flip": (a1 is not None and a2 is not None and a1 != a2),
        "argue_to_accept": (a1 == "ARGUE" and a2 == "ACCEPT"),
        "accept_to_argue": (a1 == "ACCEPT" and a2 == "ARGUE"),
        "validity_flip": (v1 is not None and v2 is not None and v1 != v2),
        "valid_to_invalid": (v1 == "VALID" and v2 == "INVALID"),
        "invalid_to_valid": (v1 == "INVALID" and v2 == "VALID"),
    }

def classify_error_recovery(mode, t1, t2):
    """
    'Error recovery trajectory' is a research-defined metric.
    Here is a default operationalization you can edit:

    - For error mode:
        recovery if T1 is not (ARGUE&VALID) but T2 becomes (ARGUE&VALID)
        deterioration if T1 is (ARGUE&VALID) but T2 becomes not
        stable_good if both (ARGUE&VALID)
        stable_bad otherwise
    - For correct mode:
        recovery if T1 not (ACCEPT&VALID) but T2 becomes (ACCEPT&VALID)
        deterioration if reverse
        stable_good if both
        stable_bad otherwise
    """
    a1, v1 = t1.get("action"), t1.get("validity")
    a2, v2 = t2.get("action"), t2.get("validity")

    if mode == "error":
        good1 = (a1 == "ARGUE" and v1 == "VALID")
        good2 = (a2 == "ARGUE" and v2 == "VALID")
    else:  # correct
        good1 = (a1 == "ACCEPT" and v1 == "VALID")
        good2 = (a2 == "ACCEPT" and v2 == "VALID")

    if (not good1) and good2:
        return "recovery"
    if good1 and (not good2):
        return "deterioration"
    if good1 and good2:
        return "stable_good"
    return "stable_bad"

def main(
    annotated_json_path: str,
    out_flip_csv: str = "flip_table.csv",
    out_traj_csv: str = "trajectory_table.csv",
):
    p = Path(annotated_json_path)
    if not p.exists():
        raise FileNotFoundError(f"annotated json not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    flip_rows = []
    traj_rows = []

    # aggregated counters per model
    agg = {}

    for model_name, model_data in data.items():
        logs = model_data.get("logs", [])
        if model_name not in agg:
            agg[model_name] = {
                "n_dialogues": 0,
                "n_with_2_ai_turns": 0,
                "action_flip": 0,
                "argue_to_accept": 0,
                "accept_to_argue": 0,
                "validity_flip": 0,
                "valid_to_invalid": 0,
                "invalid_to_valid": 0,
                "recovery": 0,
                "deterioration": 0,
                "stable_good": 0,
                "stable_bad": 0,
            }

        for log in logs:
            agg[model_name]["n_dialogues"] += 1

            mode = log.get("mode")
            case_id = log.get("case_id")
            dialogue = log.get("dialogue", [])
            ai_turns = iter_ai_turns(dialogue)

            if len(ai_turns) < 2:
                continue

            agg[model_name]["n_with_2_ai_turns"] += 1
            t1, t2 = ai_turns[0], ai_turns[1]

            flips = classify_flip(t1, t2)
            traj = classify_error_recovery(mode, t1, t2)

            # per-dialogue record (optional, useful for drilldown)
            flip_rows.append({
                "Model": model_name,
                "case_id": case_id,
                "mode": mode,
                "t1_action": t1.get("action"),
                "t2_action": t2.get("action"),
                "t1_validity": t1.get("validity"),
                "t2_validity": t2.get("validity"),
                **flips,
            })

            traj_rows.append({
                "Model": model_name,
                "case_id": case_id,
                "mode": mode,
                "trajectory": traj,
                "t1_action": t1.get("action"),
                "t1_validity": t1.get("validity"),
                "t2_action": t2.get("action"),
                "t2_validity": t2.get("validity"),
            })

            # aggregate
            for k, v in flips.items():
                if v:
                    agg[model_name][k] += 1
            agg[model_name][traj] += 1

    # write per-dialogue tables
    if flip_rows:
        with open(out_flip_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flip_rows[0].keys()))
            w.writeheader()
            w.writerows(flip_rows)

    if traj_rows:
        with open(out_traj_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
            w.writeheader()
            w.writerows(traj_rows)

    # also write summary
    summary_path = Path("flip_trajectory_summary.csv")
    summary_rows = []
    for model_name, c in agg.items():
        denom = c["n_with_2_ai_turns"] if c["n_with_2_ai_turns"] else 1
        summary_rows.append({
            "Model": model_name,
            "N_dialogues": c["n_dialogues"],
            "N_with_2_ai_turns": c["n_with_2_ai_turns"],
            "action_flip_rate": c["action_flip"] / denom,
            "argue_to_accept_rate": c["argue_to_accept"] / denom,
            "accept_to_argue_rate": c["accept_to_argue"] / denom,
            "validity_flip_rate": c["validity_flip"] / denom,
            "recovery_rate": c["recovery"] / denom,
            "deterioration_rate": c["deterioration"] / denom,
            "stable_good_rate": c["stable_good"] / denom,
            "stable_bad_rate": c["stable_bad"] / denom,
        })

    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"Saved:\n- {out_flip_csv}\n- {out_traj_csv}\n- {summary_path}")

if __name__ == "__main__":
    # default assumes you run from project root
    main(
        annotated_json_path="annotated_results.json",
        out_flip_csv="flip_table.csv",
        out_traj_csv="trajectory_table.csv",
    )