# python scripts/bootstrap_reproducibility.py \
#   --base_dir experiments \
#   --runs r10528_v1,r10528_v2,r10528_v3 \
#   --n_boot 2000 \
#   --seed 42 \
#   --out_dir analysis/bootstrap_r10528_v2


import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.medcobe import get_all_ai_utterances
from src.utils import load_json_sanitized


_T1_ORDER = ["B", "B_COT", "B_CL", "B_COT_CL"]
_T2_ORDER = ["B", "B_CL", "B_SR", "B_CL_SR"]

def exp_id_to_cid(experiment_id: str) -> str:
    """
    experiment_id 예: 'o3-B_COT__B_CL_SR' -> C8
    """
    try:
        rhs = experiment_id.split("-", 1)[1]
        t1, t2 = rhs.split("__", 1)
        r = _T1_ORDER.index(t1)
        c = _T2_ORDER.index(t2)
        return f"C{r*4 + c + 1}"
    except Exception:
        return "C?"

# --- Per-case aggregation (block bootstrap unit) ---
def build_case_table(annotated_json_path: Path) -> pd.DataFrame:
    """
    Returns per-(model, experiment_id, case_id) block rows:
      err_n, err_succ, cor_n, cor_succ, team_total, team_correct
    where:
      err_succ = # of (ARGUE, VALID) in mode=error
      cor_succ = # of (ACCEPT, VALID) in mode=correct
      *_n = total AI utterances counted in that mode
    """
    data = load_json_sanitized(annotated_json_path)

    rows = []
    for model_name, model_data in data.items():
        logs = model_data.get("logs", []) or []
        for log in logs:
            exp_id = log.get("experiment_id") or log.get("experiment_label") or "UNKNOWN_EXPERIMENT"
            case_id = str(log.get("case_id"))
            mode = log.get("mode")  # "correct" or "error"
            dialogue = log.get("dialogue", []) or []
            ai_utts = get_all_ai_utterances(dialogue)
            if not ai_utts:
                # no AI utterances -> skip (or keep as zeros; skipping matches your current behavior)
                continue

            err_n = err_succ = cor_n = cor_succ = 0
            if mode == "error":
                err_n = len(ai_utts)
                err_succ = sum(1 for a, v in ai_utts if a == "ARGUE" and v == "VALID")
            elif mode == "correct":
                cor_n = len(ai_utts)
                cor_succ = sum(1 for a, v in ai_utts if a == "ACCEPT" and v == "VALID")

            team_total = 0
            team_correct = 0
            if isinstance(log.get("is_team_correct"), bool):
                team_total = 1
                team_correct = 1 if log["is_team_correct"] else 0

            rows.append({
                "Model": model_name,
                "experiment_id": exp_id,
                "Experiment": exp_id_to_cid(exp_id),
                "case_id": case_id,
                "err_n": err_n,
                "err_succ": err_succ,
                "cor_n": cor_n,
                "cor_succ": cor_succ,
                "team_total": team_total,
                "team_correct": team_correct,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # combine correct+error contributions within the same (model, exp, case)
    grouped = df.groupby(["Model", "experiment_id", "Experiment", "case_id"], as_index=False).agg({
        "err_n": "sum",
        "err_succ": "sum",
        "cor_n": "sum",
        "cor_succ": "sum",
        "team_total": "sum",
        "team_correct": "sum",
    })
    return grouped


def compute_metrics_from_case_blocks(case_blocks: pd.DataFrame) -> Dict[str, float]:
    """
    case_blocks is already filtered to one (Model, experiment_id) group.
    """
    err_n = int(case_blocks["err_n"].sum())
    err_succ = int(case_blocks["err_succ"].sum())
    cor_n = int(case_blocks["cor_n"].sum())
    cor_succ = int(case_blocks["cor_succ"].sum())

    recall_corr = (err_succ / err_n) if err_n > 0 else float("nan")
    recall_conf = (cor_succ / cor_n) if cor_n > 0 else float("nan")

    if (not math.isnan(recall_corr)) and (not math.isnan(recall_conf)) and recall_corr > 0 and recall_conf > 0:
        medcobe = math.sqrt(recall_corr * recall_conf)
    else:
        medcobe = float("nan")

    team_total = int(case_blocks["team_total"].sum())
    team_correct = int(case_blocks["team_correct"].sum())
    team_acc = (team_correct / team_total) if team_total > 0 else float("nan")

    return {
        "N_cases": case_blocks["case_id"].nunique(),
        "Team Accuracy": team_acc,
        "Recall (Correction)": recall_corr,
        "Recall (Confirmation)": recall_conf,
        "MedCOBE Score": medcobe,
    }

def intersect_common_cases(run_case_tables):
    """
    run_case_tables: list of case_table DataFrames (from build_case_table)
    Each case_table must contain 'case_id'.
    Returns set of common case_ids across runs.
    """
    case_sets = []
    for df in run_case_tables:
        case_sets.append(set(df["case_id"].unique()))
    common = set.intersection(*case_sets)
    return common


def bootstrap_one_group(case_blocks: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, Any]:
    """
    case-level bootstrap: resample case_id blocks with replacement.
    Returns point estimate + bootstrap CI + top-k stability ingredients.
    """
    rng = random.Random(seed)

    case_ids = list(case_blocks["case_id"].unique())
    n = len(case_ids)
    if n == 0:
        return {}

    # Point estimate
    point = compute_metrics_from_case_blocks(case_blocks)

    # Pre-index by case_id for fast resampling
    blocks_by_case = {cid: case_blocks[case_blocks["case_id"] == cid] for cid in case_ids}

    medcobe_samples = []
    team_samples = []
    rc_samples = []
    rf_samples = []

    for _ in range(n_boot):
        sampled = [case_ids[rng.randrange(n)] for _ in range(n)]
        # concat sampled blocks
        sampled_df = pd.concat([blocks_by_case[c] for c in sampled], axis=0, ignore_index=True)
        m = compute_metrics_from_case_blocks(sampled_df)

        medcobe_samples.append(m["MedCOBE Score"])
        team_samples.append(m["Team Accuracy"])
        rc_samples.append(m["Recall (Correction)"])
        rf_samples.append(m["Recall (Confirmation)"])

    def pct_ci(vals: List[float], lo=2.5, hi=97.5) -> Tuple[float, float]:
        s = pd.Series(vals).dropna().sort_values()
        if s.empty:
            return (float("nan"), float("nan"))
        return (float(s.quantile(lo/100)), float(s.quantile(hi/100)))

    out = {
        **point,
        "MedCOBE_CI_low": pct_ci(medcobe_samples)[0],
        "MedCOBE_CI_high": pct_ci(medcobe_samples)[1],
        "TeamAcc_CI_low": pct_ci(team_samples)[0],
        "TeamAcc_CI_high": pct_ci(team_samples)[1],
        "RecallCorr_CI_low": pct_ci(rc_samples)[0],
        "RecallCorr_CI_high": pct_ci(rc_samples)[1],
        "RecallConf_CI_low": pct_ci(rf_samples)[0],
        "RecallConf_CI_high": pct_ci(rf_samples)[1],
        # store samples optionally if you want (can be huge)
    }
    return out


def bootstrap_run(annotated_json_path: Path, n_boot: int, seed: int) -> pd.DataFrame:
    """
    For one run (v1 or v2 or v3): compute bootstrap CI per (Model, Experiment).
    """
    case_table = build_case_table(annotated_json_path)
    if case_table.empty:
        raise ValueError(f"No usable logs found in {annotated_json_path}")

    results = []
    for (model, exp_id, c_id), grp in case_table.groupby(["Model", "experiment_id", "Experiment"]):
        stats = bootstrap_one_group(grp, n_boot=n_boot, seed=seed)
        if not stats:
            continue
        results.append({
            "Run": annotated_json_path.parent.parent.name,  # e.g. o3_reproducibility_check_v1
            "Model": model,
            "experiment_id": exp_id,
            "Experiment": c_id,
            **stats
        })

    return pd.DataFrame(results)


def aggregate_across_runs(run_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate across runs and compute:
    - mean
    - std
    - bootstrap CI across runs (percentile of run-level bootstrap means)
    """

    all_df = pd.concat(run_dfs, ignore_index=True)

    metrics = [
        "Team Accuracy",
        "Recall (Correction)",
        "Recall (Confirmation)",
        "MedCOBE Score",
    ]

    def pct_ci(series, lo=2.5, hi=97.5):
        s = series.dropna().sort_values()
        if len(s) == 0:
            return pd.Series([float("nan"), float("nan")])
        return pd.Series([s.quantile(lo/100), s.quantile(hi/100)])

    agg_rows = []

    for (model, exp, exp_id), grp in all_df.groupby(["Model", "Experiment", "experiment_id"]):

        row = {
            "Model": model,
            "Experiment": exp,
            "experiment_id": exp_id,
            "N_runs": grp["Run"].nunique(),
            "N_cases": grp["N_cases"].mean(),
        }

        for m in metrics:

            mean = grp[m].mean()
            std = grp[m].std()

            ci_low, ci_high = pct_ci(grp[m])

            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_CI_low"] = ci_low
            row[f"{m}_CI_high"] = ci_high

        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)

    return agg.sort_values(
        ["Model", "MedCOBE Score_mean"],
        ascending=[True, False]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True,
                    help="e.g., experiments/ (contains o3_reproducibility_check_v1/v2/v3)")
    ap.add_argument("--runs", type=str, default="o3_reproducibility_check_v1,o3_reproducibility_check_v2,o3_reproducibility_check_v3",
                    help="comma-separated run folder names under base_dir")
    ap.add_argument("--annotated_relpath", type=str, default="evaluation/annotated_results.json",
                    help="relative path inside each run folder")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="analysis/bootstrap")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    run_dfs = []
        # 1) 먼저 모든 run의 case_table을 생성
    run_case_tables = []
    run_paths = []

    for rn in run_names:
        annotated_path = base_dir / rn / args.annotated_relpath
        if not annotated_path.exists():
            raise FileNotFoundError(f"Missing annotated_results.json: {annotated_path}")
        case_table = build_case_table(annotated_path)
        run_case_tables.append(case_table)
        run_paths.append((rn, annotated_path))

    # 2) 공통 case_id 계산
    common_case_ids = intersect_common_cases(run_case_tables)
    print(f"[INFO] Common case_ids across runs: {len(common_case_ids)}")

    # 3) 각 run에서 공통 케이스만 필터 후 bootstrap 수행
    run_dfs = []
    for (rn, annotated_path), case_table in zip(run_paths, run_case_tables):

        filtered_case_table = case_table[
            case_table["case_id"].isin(common_case_ids)
        ].copy()

        print(f"[BOOT] {rn} | using {filtered_case_table['case_id'].nunique()} cases")

        results = []
        for (model, exp_id, c_id), grp in filtered_case_table.groupby(["Model", "experiment_id", "Experiment"]):
            stats = bootstrap_one_group(grp, n_boot=args.n_boot, seed=args.seed)
            if not stats:
                continue
            results.append({
                "Run": rn,
                "Model": model,
                "experiment_id": exp_id,
                "Experiment": c_id,
                **stats
            })

        df_run = pd.DataFrame(results)
        df_run.to_csv(out_dir / f"{rn}_bootstrap_ci.csv", index=False)
        run_dfs.append(df_run)

    # Aggregate across runs (point estimate mean/std)
    agg = aggregate_across_runs(run_dfs)
    agg.to_csv(out_dir / "bootstrap_across_runs_summary.csv", index=False)

    print(f"[OK] Wrote per-run bootstrap CI to: {out_dir}")
    print(f"[OK] Wrote across-run summary to: {out_dir / 'bootstrap_across_runs_summary.csv'}")


if __name__ == "__main__":
    main()