#!/usr/bin/env python3
"""
Analyze MedCOBE experiment outputs and produce:
- Heatmap, CSV (metrics vs baseline)
- Figure 1: MedCOBE score + 95% CI (point plot)
- Figure 2: Paired difference vs C1 (forest plot)
- Figure 3: Reproducibility (boxplot/violin)
- Figure 4: Model consistency (o3 vs r1 scatter)
- Figure 7: Effect size vs baseline
- Figure 8: Turn-level (T1 vs T2 contribution)

Usage:
  python scripts/run_analysis.py [--figures 1,2,3,4,7,8]
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Config (edit model_name here)
# ============================
MODEL_NAME = "o3"  # or "r1_0528", "r1", "o1", etc.
MODEL_NAME_2 = "r1"  # for Figure 4 (model consistency: o3 vs r1)
N_CASES_LABEL = 528  # for figure titles (e.g. "528 cases")
# ============================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.medcobe import (
    _T1_ORDER,
    _T2_ORDER,
    exp_id_to_cid,
    get_all_ai_utterances,
)
from src.utils import load_json_sanitized


def _short_name(full: str) -> str:
    m = {"openai/o3": "o3", "deepseek/deepseek-r1-0528": "r1_0528",
         "deepseek/deepseek-r1": "r1", "openai/o1": "o1"}
    return m.get(full, full.split("/")[-1].replace("-", "_"))


def _get_ai_utterances_with_error(dialogue):
    for turn in (dialogue or []):
        if turn.get("role") != "AI":
            continue
        action = turn.get("action") or turn.get("ai_action")
        validity = turn.get("validity") or turn.get("reasoning_validity")
        if action is None or validity is None:
            continue
        et = turn.get("error_type")
        if isinstance(et, str):
            et = [et] if et else ["NONE"]
        elif not isinstance(et, list):
            et = ["NONE"]
        et = [str(x).strip().upper() for x in et if x]
        if not et:
            et = ["NONE"]
        yield (str(action).upper(), str(validity).upper(), et)


def _is_error_utterance(error_types):
    s = set(t for t in error_types if t)
    return s != {"NONE"} and len(s) > 0


def _resolve_paths(model_name: str) -> dict:
    """Resolve input/output paths from model short name."""
    short = model_name.replace("/", "_").replace("-", "_")
    if "o3" in short or short == "o3":
        short = "o3"
    return {
        "input_dir": PROJECT_ROOT / f"experiments/(main) {short}_combined_sample",
        "output_dir": PROJECT_ROOT / f"analysis/(main) {short}",
        "permutation_dir": PROJECT_ROOT / f"analysis/(main) paired bootstrap permutation_{short}",
        "bootstrap_dir": PROJECT_ROOT / f"analysis/(reproducibility) bootstrap_{short}",
    }


def _load_experiments_yaml() -> dict:
    p = PROJECT_ROOT / "resources" / "experiments.yaml"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _model_full_name(short: str) -> str:
    cfg = _load_experiments_yaml()
    models = cfg.get("models", {})
    return models.get(short, f"openai/{short}")


# --- Bootstrap (from bootstrap_reproducibility) ---
def _build_case_table(annotated_json_path: Path, model_filter: str = None) -> pd.DataFrame:
    data = load_json_sanitized(annotated_json_path)

    rows = []
    for model_name, model_data in data.items():
        if model_filter and model_filter not in model_name:
            continue
        logs = model_data.get("logs", []) or []
        for log in logs:
            exp_id = log.get("experiment_id") or log.get("experiment_label") or "UNKNOWN"
            case_id = str(log.get("case_id"))
            mode = log.get("mode")
            dialogue = log.get("dialogue", []) or []
            ai_utts = get_all_ai_utterances(dialogue)
            if not ai_utts:
                continue
            err_n = err_succ = cor_n = cor_succ = 0
            if mode == "error":
                err_n = len(ai_utts)
                err_succ = sum(1 for a, v in ai_utts if a == "ARGUE" and v == "VALID")
            elif mode == "correct":
                cor_n = len(ai_utts)
                cor_succ = sum(1 for a, v in ai_utts if a == "ACCEPT" and v == "VALID")

            team_total = 1 if isinstance(log.get("is_team_correct"), bool) else 0
            team_correct = 1 if log.get("is_team_correct") else 0

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

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["Model", "experiment_id", "Experiment", "case_id"], as_index=False).agg({
        "err_n": "sum", "err_succ": "sum", "cor_n": "sum", "cor_succ": "sum",
        "team_total": "sum", "team_correct": "sum",
    })


def _compute_medcobe(case_blocks: pd.DataFrame) -> float:
    err_n = int(case_blocks["err_n"].sum())
    err_succ = int(case_blocks["err_succ"].sum())
    cor_n = int(case_blocks["cor_n"].sum())
    cor_succ = int(case_blocks["cor_succ"].sum())
    rc = (err_succ / err_n) if err_n > 0 else float("nan")
    rf = (cor_succ / cor_n) if cor_n > 0 else float("nan")
    if not math.isnan(rc) and not math.isnan(rf) and rc > 0 and rf > 0:
        return math.sqrt(rc * rf)
    return float("nan")


def _bootstrap_one_group(case_blocks: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    case_ids = list(case_blocks["case_id"].unique())
    n = len(case_ids)
    if n == 0:
        return {}
    point = _compute_medcobe(case_blocks)
    blocks_by_case = {c: case_blocks[case_blocks["case_id"] == c] for c in case_ids}
    samples = []
    for _ in range(n_boot):
        sampled = [case_ids[rng.randrange(n)] for _ in range(n)]
        sampled_df = pd.concat([blocks_by_case[c] for c in sampled], axis=0, ignore_index=True)
        samples.append(_compute_medcobe(sampled_df))
    s = pd.Series(samples).dropna().sort_values()
    ci_lo = float(s.quantile(0.025)) if not s.empty else float("nan")
    ci_hi = float(s.quantile(0.975)) if not s.empty else float("nan")
    return {"MedCOBE": point, "CI_low": ci_lo, "CI_high": ci_hi}


def _boot_df_from_permutation(
    rows: list,
    perm_df: pd.DataFrame,
    annotated_path: Path,
    model_full: str,
) -> pd.DataFrame:
    """
    Build boot_df for Figure 1 from bootstrap_permutation result.
    permutation has delta vs C1 with CI for C2-C16. Convert to absolute MedCOBE CI:
      C_i CI = [C1_medcobe + delta_CI_low, C1_medcobe + delta_CI_high]
    C1 needs its own bootstrap (permutation doesn't output it).
    """
    c1_row = next((r for r in rows if r["Experiment"] == "C1"), None)
    c1_medcobe = c1_row["Medcobe_score"] if c1_row else float("nan")

    out = []
    for r in rows:
        cid = r["Experiment"]
        medcobe = r["Medcobe_score"]
        if cid == "C1":
            case_table = _build_case_table(annotated_path, model_filter=model_full.split("/")[-1].split("-")[0])
            c1_grp = case_table[case_table["Experiment"] == "C1"]
            st = _bootstrap_one_group(c1_grp, n_boot=2000, seed=42) if len(c1_grp) > 0 else {}
            out.append({
                "Experiment": cid,
                "MedCOBE": medcobe,
                "CI_low": st.get("CI_low", float("nan")),
                "CI_high": st.get("CI_high", float("nan")),
            })
        else:
            pm = perm_df[perm_df["cid"] == cid]
            perm_row = pm.iloc[0] if len(pm) > 0 else None
            if perm_row is not None:
                out.append({
                    "Experiment": cid,
                    "MedCOBE": medcobe,
                    "CI_low": c1_medcobe + perm_row["CI_low"],
                    "CI_high": c1_medcobe + perm_row["CI_high"],
                })
            else:
                out.append({"Experiment": cid, "MedCOBE": medcobe, "CI_low": float("nan"), "CI_high": float("nan")})
    return pd.DataFrame(out)


def _bootstrap_main(annotated_path: Path, model_full: str, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    case_table = _build_case_table(annotated_path, model_filter=model_full.split("/")[-1].split("-")[0])
    if case_table.empty:
        return pd.DataFrame()
    rows = []
    groups = list(case_table.groupby(["Model", "experiment_id", "Experiment"]))
    for i, ((_, exp_id, cid), grp) in enumerate(groups, 1):
        print(f"  Bootstrap {i}/{len(groups)} ({cid})...", end=" ", flush=True)
        st = _bootstrap_one_group(grp, n_boot=n_boot, seed=seed)
        if st:
            rows.append({"Experiment": cid, "experiment_id": exp_id, **st})
        print("done", flush=True)
    return pd.DataFrame(rows).sort_values("Experiment")


def infer_model_from_annotated(annotated_data: dict) -> str:
    for k in annotated_data:
        if "/o3" in k:
            return "o3"
        if "r1-0528" in k or "r1_0528" in k:
            return "r1_0528"
        if "r1" in k and "0528" not in k:
            return "r1"
    return list(annotated_data.keys())[0].split("/")[-1]


def compute_per_experiment_metrics(annotated_data: dict, model_key: str, model_full_name: str):
    model_data = annotated_data.get(model_full_name)
    if model_data is None:
        for k, v in annotated_data.items():
            if model_key in k:
                model_data, model_full_name = v, k
                break
    if model_data is None and len(annotated_data) == 1:
        model_full_name = list(annotated_data.keys())[0]
        model_data = annotated_data[model_full_name]
    if not model_data:
        raise ValueError(f"No model data for {model_full_name}. Keys: {list(annotated_data.keys())}")

    logs = model_data.get("logs", [])
    if not logs:
        raise ValueError("No logs")

    grouped = {}
    for log in logs:
        exp_id = log.get("experiment_id") or log.get("experiment_label") or "UNKNOWN"
        grouped.setdefault(exp_id, []).append(log)

    baseline, rows = None, []
    for exp_id, exp_logs in grouped.items():
        cid = exp_id_to_cid(exp_id)
        try:
            t1, t2 = exp_id.split("-", 1)[1].split("__", 1)
        except Exception:
            t1, t2 = "?", "?"

        err_total = err_success = cor_total = cor_success = 0
        err_valid = cor_valid = err_err_ct = cor_err_ct = total_valid = total_err_ct = 0

        for log in exp_logs:
            mode = log.get("mode")
            dialogue = log.get("dialogue", [])

            for action, validity, et in _get_ai_utterances_with_error(dialogue):
                total_valid += 1 if validity == "VALID" else 0
                total_err_ct += 1 if _is_error_utterance(et) else 0

                if mode == "error":
                    err_total += 1
                    err_valid += 1 if validity == "VALID" else 0
                    err_err_ct += 1 if _is_error_utterance(et) else 0
                    if action == "ARGUE" and validity == "VALID":
                        err_success += 1
                elif mode == "correct":
                    cor_total += 1
                    cor_valid += 1 if validity == "VALID" else 0
                    cor_err_ct += 1 if _is_error_utterance(et) else 0
                    if action == "ACCEPT" and validity == "VALID":
                        cor_success += 1

        total = err_total + cor_total
        r_argue = (err_success / err_total) if err_total > 0 else float("nan")
        r_accept = (cor_success / cor_total) if cor_total > 0 else float("nan")
        validity_rate = (total_valid / total) if total > 0 else float("nan")
        cor_vr = (cor_valid / cor_total) if cor_total > 0 else float("nan")
        err_vr = (err_valid / err_total) if err_total > 0 else float("nan")
        medcobe = math.sqrt(r_argue * r_accept) if not (math.isnan(r_argue) or math.isnan(r_accept)) and r_argue > 0 and r_accept > 0 else float("nan")

        row = {
            "Experiment": cid, "experiment_id": exp_id, "T1": t1, "T2": t2,
            "N_dialogues": len(exp_logs), "R_accept": r_accept, "R_argue": r_argue,
            "Medcobe_score": medcobe, "validity_rate": validity_rate,
            "error_count": total_err_ct, "total_utterances": total,
            "correct_R_accept": r_accept, "correct_N": cor_total,
            "correct_validity_rate": cor_vr, "correct_error_count": cor_err_ct,
            "error_R_argue": r_argue, "error_N": err_total,
            "error_validity_rate": err_vr, "error_error_count": err_err_ct,
        }
        rows.append(row)
        if cid == "C1":
            baseline = row

    return rows, baseline


def build_heatmap_data(rows):
    data = {}
    for r in rows:
        if r["T1"] in _T1_ORDER and r["T2"] in _T2_ORDER:
            data[(r["T1"], r["T2"])] = r["Medcobe_score"]
    return [[data.get((t1, t2), float("nan")) for t2 in _T2_ORDER] for t1 in _T1_ORDER]


def build_comparison_csv(rows, baseline) -> pd.DataFrame:
    if not baseline:
        baseline = next((r for r in rows if r["Experiment"] == "C1"), None)
    if not baseline:
        return pd.DataFrame(rows)

    def _d(a, b):
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            return None
        return float(a) - float(b)

    out = []
    for r in rows:
        out.append({
            "Experiment": r["Experiment"], "T1": r["T1"], "T2": r["T2"],
            "Mode": "aggregate",
            "R_accept": r["R_accept"], "R_argue": r["R_argue"],
            "Medcobe_score": r["Medcobe_score"], "validity_rate": r["validity_rate"],
            "error_count": r["error_count"],
            "R_accept_vs_C1": _d(r["R_accept"], baseline["R_accept"]),
            "R_argue_vs_C1": _d(r["R_argue"], baseline["R_argue"]),
            "Medcobe_vs_C1": _d(r["Medcobe_score"], baseline["Medcobe_score"]),
            "validity_rate_vs_C1": _d(r["validity_rate"], baseline["validity_rate"]),
            "error_count_vs_C1": r["error_count"] - baseline.get("error_count", 0),
        })
    base_cv = baseline.get("correct_validity_rate", float("nan"))
    base_ev = baseline.get("error_validity_rate", float("nan"))
    base_ce = baseline.get("correct_error_count", 0)
    base_ee = baseline.get("error_error_count", 0)

    for r in rows:
        out.append({
            "Experiment": r["Experiment"], "T1": r["T1"], "T2": r["T2"], "Mode": "correct",
            "R_accept": r["correct_R_accept"], "R_argue": None, "Medcobe_score": None,
            "validity_rate": r.get("correct_validity_rate", r["validity_rate"]),
            "error_count": r.get("correct_error_count", r["error_count"]),
            "R_accept_vs_C1": _d(r["correct_R_accept"], baseline["R_accept"]),
            "R_argue_vs_C1": None, "Medcobe_vs_C1": None,
            "validity_rate_vs_C1": _d(r.get("correct_validity_rate"), base_cv),
            "error_count_vs_C1": r.get("correct_error_count", r["error_count"]) - base_ce,
        })
        out.append({
            "Experiment": r["Experiment"], "T1": r["T1"], "T2": r["T2"], "Mode": "incorrect",
            "R_accept": None, "R_argue": r["error_R_argue"], "Medcobe_score": None,
            "validity_rate": r.get("error_validity_rate", r["validity_rate"]),
            "error_count": r.get("error_error_count", r["error_count"]),
            "R_accept_vs_C1": None, "R_argue_vs_C1": _d(r["error_R_argue"], baseline["R_argue"]),
            "Medcobe_vs_C1": None,
            "validity_rate_vs_C1": _d(r.get("error_validity_rate"), base_ev),
            "error_count_vs_C1": r.get("error_error_count", r["error_count"]) - base_ee,
        })
    return pd.DataFrame(out)


# --- Figures ---
def _ensure_mpl():
    return plt, np


def fig1_medcobe_ci(rows, boot_df, perm_df, output_dir: Path, model_name: str):
    """Figure 1: Prompt별 MedCOBE score + 95% CI. C1 강조, 유의한 것 별표."""
    plt, np = _ensure_mpl()
    if plt is None:
        print("  [skip] Figure 1: matplotlib required")
        return

    # Merge: rows (point) + boot_df (CI) + perm_df (sig)
    exp_order = [f"C{i}" for i in range(1, 17)]
    data = {r["Experiment"]: r for r in rows}
    if boot_df is not None and not boot_df.empty:
        for _, r in boot_df.iterrows():
            c = r["Experiment"]
            if c not in data:
                data[c] = {"Experiment": c, "Medcobe_score": r.get("MedCOBE", float("nan"))}
            data[c]["CI_low"] = r.get("CI_low", float("nan"))
            data[c]["CI_high"] = r.get("CI_high", float("nan"))

    sig = {}
    if perm_df is not None and not perm_df.empty:
        for _, r in perm_df.iterrows():
            sig[r["cid"]] = r.get("sig_perm", False) or r.get("sig_boot", False)

    x = list(range(len(exp_order)))
    y = [data.get(c, {}).get("Medcobe_score", float("nan")) for c in exp_order]
    lo = [data.get(c, {}).get("CI_low", float("nan")) for c in exp_order]
    hi = [data.get(c, {}).get("CI_high", float("nan")) for c in exp_order]
    colors = ["#2ecc71" if c == "C1" else "#3498db" for c in exp_order]
    markers = ["*" if sig.get(c, False) else "o" for c in exp_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, c in enumerate(exp_order):
        yi, loi, hii = y[i], lo[i], hi[i]
        if math.isnan(yi):
            continue
        err_lo = max(0, yi - loi) if not math.isnan(loi) else 0
        err_hi = max(0, hii - yi) if not math.isnan(hii) else 0
        if err_lo > 0 or err_hi > 0:
            ax.errorbar(i, yi, yerr=[[err_lo], [err_hi]], fmt="none", color="gray", capsize=3)
        ax.scatter(i, yi, c=colors[i], s=80, zorder=5, marker=markers[i], edgecolors="black")
        if sig.get(c, False):
            ax.text(i, hii + 0.02 if not math.isnan(hii) else yi + 0.02, "*", ha="center", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_order)
    ax.set_xlabel("Prompt (C1–C16)")
    ax.set_ylabel("MedCOBE Score")
    ax.set_title(f"Figure 1. Prompt별 MedCOBE score + 95% CI ({model_name}, {N_CASES_LABEL} cases)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_medcobe_ci.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 1 saved: figure1_medcobe_ci.png")


def fig2_forest_plot(perm_df, output_dir: Path, model_name: str):
    """Figure 2: Paired difference vs C1 (horizontal forest plot)."""
    plt, np = _ensure_mpl()
    if plt is None or perm_df is None or perm_df.empty:
        print("  [skip] Figure 2: matplotlib or permutation data required")
        return

    df = perm_df.sort_values("delta_vs_C1", ascending=True)
    y_pos = range(len(df))
    ax = plt.subplot(111)
    ax.errorbar(df["delta_vs_C1"], y_pos, xerr=[df["delta_vs_C1"] - df["CI_low"], df["CI_high"] - df["delta_vs_C1"]],
                fmt="o", capsize=3, color="#3498db")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["cid"])
    ax.set_xlabel("ΔMedCOBE vs C1")
    ax.set_ylabel("Prompt")
    ax.set_title(f"Figure 2. Paired difference vs C1 ({model_name})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 2 saved: figure2_forest_plot.png")


def fig3_reproducibility(bootstrap_dir: Path, output_dir: Path, model_name: str):
    """Figure 3: Reproducibility plot (133×3 runs) – boxplot per C1–C16."""
    plt, np = _ensure_mpl()
    if plt is None:
        print("  [skip] Figure 3: matplotlib required")
        return

    run_files = sorted(bootstrap_dir.glob("*_bootstrap_ci.csv"))
    if not run_files:
        print("  [skip] Figure 3: no *_bootstrap_ci.csv in", bootstrap_dir)
        return

    dfs = []
    for f in run_files:
        df = pd.read_csv(f)
        df["Run"] = f.stem.replace("_bootstrap_ci", "")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    exp_order = [f"C{i}" for i in range(1, 17)]
    data = [combined[combined["Experiment"] == c]["MedCOBE Score"].dropna().values for c in exp_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=exp_order, patch_artist=True)
    for i, c in enumerate(exp_order):
        if c == "C1":
            bp["boxes"][i].set_facecolor("#90EE90")
    ax.set_xlabel("Prompt (C1–C16)")
    ax.set_ylabel("MedCOBE Score")
    ax.set_title(f"Figure 3. Reproducibility plot ({model_name}, 133×3 runs)")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_reproducibility.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 3 saved: figure3_reproducibility.png")


def fig4_model_consistency(bootstrap_dir_m1: Path, bootstrap_dir_m2: Path,
                          output_dir: Path, m1_name: str, m2_name: str):
    """Figure 4: o3 vs r1 scatter, regression line, Pearson r."""
    plt, np = _ensure_mpl()
    if plt is None or np is None:
        print("  [skip] Figure 4: matplotlib required")
        return

    def load_mean(path: Path) -> dict:
        summary = path / "bootstrap_across_runs_summary.csv"
        if not summary.exists():
            run_files = sorted(path.glob("*_bootstrap_ci.csv"))
            if not run_files:
                return {}
            dfs = [pd.read_csv(f) for f in run_files]
            combined = pd.concat(dfs)
            agg = combined.groupby("Experiment")["MedCOBE Score"].mean()
            return agg.to_dict()
        df = pd.read_csv(summary)
        return dict(zip(df["Experiment"], df.get("MedCOBE Score_mean", df.get("MedCOBE Score", 0))))

    d1 = load_mean(bootstrap_dir_m1)
    d2 = load_mean(bootstrap_dir_m2)
    common = sorted(set(d1) & set(d2))
    if not common:
        print("  [skip] Figure 4: no common experiments between models")
        return

    x = np.array([d2[c] for c in common])
    y = np.array([d1[c] for c in common])
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    common = [c for c, m in zip(common, mask) if m]

    if len(x) < 2:
        print("  [skip] Figure 4: insufficient data")
        return

    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, c="#3498db", s=60)
    xline = np.linspace(x.min(), x.max(), 50)
    ax.plot(xline, p(xline), "r--", label=f"y = {z[0]:.3f}x + {z[1]:.3f}")
    for i, c in enumerate(common):
        ax.annotate(c, (x[i], y[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"{m2_name} MedCOBE mean (133×3)")
    ax.set_ylabel(f"{m1_name} MedCOBE mean (133×3)")
    ax.set_title(f"Figure 4. Model consistency ({m1_name} vs {m2_name})\nPearson r ≈ {r:.2f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "figure4_model_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 4 saved: figure4_model_consistency.png")


def fig7_effect_size(perm_df, output_dir: Path, model_name: str):
    """Figure 7: Effect size vs baseline – prompt on x, ΔMedCOBE on y, bootstrap CI."""
    plt, np = _ensure_mpl()
    if plt is None or perm_df is None or perm_df.empty:
        print("  [skip] Figure 7: matplotlib or permutation data required")
        return

    df = perm_df.sort_values("cid")
    x = range(len(df))
    y = df["delta_vs_C1"].values
    err_lo = y - df["CI_low"].values
    err_hi = df["CI_high"].values - y

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="o", capsize=4, color="#3498db")
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["cid"])
    ax.set_xlabel("Prompt")
    ax.set_ylabel("ΔMedCOBE vs C1")
    ax.set_title(f"Figure 7. Effect size vs baseline ({model_name})")
    plt.tight_layout()
    plt.savefig(output_dir / "figure7_effect_size.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 7 saved: figure7_effect_size.png")


def fig8_turn_level(rows, output_dir: Path, model_name: str):
    """Figure 8: Turn1 vs Turn2 MedCOBE contribution (marginal means)."""
    plt, np = _ensure_mpl()
    if plt is None:
        print("  [skip] Figure 8: matplotlib required")
        return

    df = pd.DataFrame(rows)
    t1_means = df.groupby("T1")["Medcobe_score"].mean()
    t2_means = df.groupby("T2")["Medcobe_score"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    x1 = range(len(_T1_ORDER))
    x2 = range(len(_T1_ORDER), len(_T1_ORDER) + len(_T2_ORDER))
    y1 = [t1_means.get(t, float("nan")) for t in _T1_ORDER]
    y2 = [t2_means.get(t, float("nan")) for t in _T2_ORDER]

    ax.bar([i - 0.2 for i in x1], y1, width=0.35, label="Turn1", color="#3498db")
    ax.bar([i + 0.2 for i in x2], y2, width=0.35, label="Turn2", color="#e74c3c")
    ax.set_xticks(list(x1) + list(x2))
    ax.set_xticklabels(list(_T1_ORDER) + list(_T2_ORDER), rotation=45, ha="right")
    ax.set_ylabel("MedCOBE Score (marginal mean)")
    ax.set_xlabel("Prompt variant")
    ax.set_title(f"Figure 8. Turn-level MedCOBE contribution ({model_name})")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "figure8_turn_level.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 8 saved: figure8_turn_level.png")


def save_heatmap(matrix: list, output_path: Path, model_key: str):
    plt, np = _ensure_mpl()
    if plt is None:
        print("  [WARN] matplotlib not installed; heatmap image skipped. Install: pip install matplotlib")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    arr = np.array(matrix, dtype=float)
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(_T2_ORDER)))
    ax.set_xticklabels(_T2_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(_T1_ORDER)))
    ax.set_yticklabels(_T1_ORDER)
    for i in range(len(_T1_ORDER)):
        for j in range(len(_T2_ORDER)):
            v = arr[i, j]
            ax.text(j, i, f"{v:.2f}" if not (np.isnan(v) or np.isinf(v)) else "-",
                    ha="center", va="center", fontsize=10)
    ax.set_xlabel("T2 (Turn 2)")
    ax.set_ylabel("T1 (Turn 1)")
    ax.set_title(f"MedCOBE Score by Prompt Components ({model_key})")
    plt.colorbar(im, ax=ax, label="MedCOBE Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {output_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=str, default="1,2,3,4,7,8",
                    help="Comma-separated figure numbers to generate (e.g. 1,2,3)")
    ap.add_argument("--model", type=str, default=None, help="Override MODEL_NAME")
    ap.add_argument("--model2", type=str, default=None, help="Override MODEL_NAME_2 for Figure 4")
    args = ap.parse_args()

    model_name = args.model or MODEL_NAME
    model2_name = args.model2 or MODEL_NAME_2
    model_full = _model_full_name(model_name)

    paths = _resolve_paths(model_name)
    input_dir = paths["input_dir"]
    output_dir = paths["output_dir"]
    permutation_dir = paths["permutation_dir"]
    bootstrap_dir = paths["bootstrap_dir"]
    bootstrap_dir_m2 = _resolve_paths(model2_name)["bootstrap_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = input_dir / "evaluation" / "annotated_results.json"
    if not annotated_path.exists():
        annotated_path = input_dir / "output" / "evaluation" / "annotated_results.json"
    if not annotated_path.exists():
        raise FileNotFoundError(f"annotated_results.json not found under {input_dir}")

    annotated_data = load_json_sanitized(annotated_path)

    inferred = infer_model_from_annotated(annotated_data)
    if model_full not in annotated_data:
        model_full = next((k for k in annotated_data if inferred in k), list(annotated_data.keys())[0])

    rows, baseline = compute_per_experiment_metrics(annotated_data, model_name, model_full)
    print(f"  Computed metrics for {len(rows)} experiments")

    perm_df = None
    perm_file = permutation_dir / f"{model_name}_paired_bootstrap_permutation_result.csv"
    if permutation_dir.exists() and perm_file.exists():
        perm_df = pd.read_csv(perm_file)
        print(f"  Loaded permutation bootstrap: {perm_file.name}")

    # boot_df for Figure 1: from permutation (paired bootstrap) when available
    boot_df = None
    if perm_df is not None and not perm_df.empty:
        boot_df = _boot_df_from_permutation(rows, perm_df, annotated_path, model_full)

    # Heatmap & CSV
    matrix = build_heatmap_data(rows)
    save_heatmap(matrix, output_dir / "heatmap_medcobe.png", model_name)
    pd.DataFrame(matrix, index=_T1_ORDER, columns=_T2_ORDER).to_csv(output_dir / "heatmap_medcobe.csv")
    build_comparison_csv(rows, baseline).to_csv(output_dir / "metrics_vs_baseline.csv", index=False, na_rep="")
    print("  metrics_vs_baseline.csv saved")

    figures = [int(x.strip()) for x in args.figures.split(",") if x.strip()]

    if 1 in figures:
        fig1_medcobe_ci(rows, boot_df, perm_df, output_dir, model_name)
    if 2 in figures and perm_df is not None:
        fig2_forest_plot(perm_df, output_dir, model_name)
    if 3 in figures:
        fig3_reproducibility(bootstrap_dir, output_dir, model_name)
    if 4 in figures:
        fig4_model_consistency(bootstrap_dir, bootstrap_dir_m2, output_dir, model_name, model2_name)
    if 7 in figures and perm_df is not None:
        fig7_effect_size(perm_df, output_dir, model_name)
    if 8 in figures:
        fig8_turn_level(rows, output_dir, model_name)

    print("\n  Analysis complete.")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
