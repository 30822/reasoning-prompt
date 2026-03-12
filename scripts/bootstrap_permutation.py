# bootstrap_permutation_dataset_level.py
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yaml

# 프로젝트 루트 import (기존 코드 스타일 유지)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.medcobe import get_all_ai_utterances
from src.utils import load_json_sanitized


# ============================
# Config
# ============================
EXPERIMENTS_YAML = "./resources/experiments.yaml"
MODEL_NAME = "deepseek/deepseek-r1-0528"

_T1_ORDER = ["B", "B_COT", "B_CL", "B_COT_CL"]
_T2_ORDER = ["B", "B_CL", "B_SR", "B_CL_SR"]

# 통계 설정
N_BOOT = 20000
N_PERM = 50000
SEED = 42

ALPHA = 0.05  # 95% CI


# ============================
# Helpers
# ============================
def load_model_map(experiments_yaml_path: str) -> dict[str, str]:
    cfg = yaml.safe_load(Path(experiments_yaml_path).read_text(encoding="utf-8"))
    model_map = cfg.get("models", cfg)
    return model_map

def resolve_short_model_name(full_model_name: str, model_map: dict[str, str]) -> str:
    inv = {v: k for k, v in model_map.items()}
    return inv.get(full_model_name, full_model_name.split("/")[-1].replace("-", "_"))

def exp_id_to_cid(experiment_id: str) -> Optional[str]:
    try:
        rhs = experiment_id.split("-", 1)[1]
        t1, t2 = rhs.split("__", 1)
        r = _T1_ORDER.index(t1)
        c = _T2_ORDER.index(t2)
        return f"C{r*4 + c + 1}"
    except Exception:
        return None

def holm_correction(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    max_val = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        max_val = max(max_val, val)
        adj[idx] = min(max_val, 1.0)
    return adj


# ============================
# 핵심: per-case counts를 미리 만들어두고
# bootstrap/permutation에서 빠르게 합산
# ============================
@dataclass(frozen=True)
class Counts:
    # correct 모드(= confirmation)
    tot_correct: int
    succ_accept: int
    # incorrect/error 모드(= correction)
    tot_incorrect: int
    succ_argue: int

def add_counts(a: Counts, b: Counts) -> Counts:
    return Counts(
        tot_correct=a.tot_correct + b.tot_correct,
        succ_accept=a.succ_accept + b.succ_accept,
        tot_incorrect=a.tot_incorrect + b.tot_incorrect,
        succ_argue=a.succ_argue + b.succ_argue,
    )

def medcobe_from_dataset_counts(c: Counts) -> float:
    """dataset-level MEDCOBE = sqrt(R_accept * R_argue) with strict validity."""
    if c.tot_correct <= 0 or c.tot_incorrect <= 0:
        return float("nan")

    r_accept = c.succ_accept / c.tot_correct
    r_argue = c.succ_argue / c.tot_incorrect

    # 논문/기존 구현 관례: 둘 중 하나가 0이면 score 0
    if r_accept <= 0 or r_argue <= 0:
        return 0.0
    return math.sqrt(r_accept * r_argue)

def build_case_prompt_counts(annotated_json_path: str) -> Dict[str, Dict[str, Counts]]:
    """
    return: case_id -> cid -> Counts
    """
    data = load_json_sanitized(annotated_json_path)
    logs = data[MODEL_NAME]["logs"]

    # 임시 누적용 dict
    tmp: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    # 위는 쓰기 복잡하니, 바로 counts로 누적
    out: Dict[str, Dict[str, Counts]] = {}

    for log in logs:
        cid = exp_id_to_cid(log.get("experiment_id", ""))
        if cid is None:
            continue
        case_id = log.get("case_id", "")
        if not case_id:
            continue

        mode = log.get("mode")
        dialogue = log.get("dialogue", [])
        ai_utts = get_all_ai_utterances(dialogue)
        if not ai_utts:
            continue

        # 현재 case/cid counts 가져오기
        case_map = out.setdefault(case_id, {})
        cur = case_map.get(cid, Counts(0, 0, 0, 0))

        if mode in ("incorrect", "error"):
            tot_inc = len(ai_utts)
            succ_arg = sum(1 for action, validity in ai_utts if action == "ARGUE" and validity == "VALID")
            cur = Counts(
                tot_correct=cur.tot_correct,
                succ_accept=cur.succ_accept,
                tot_incorrect=cur.tot_incorrect + tot_inc,
                succ_argue=cur.succ_argue + succ_arg,
            )
        elif mode == "correct":
            tot_cor = len(ai_utts)
            succ_acc = sum(1 for action, validity in ai_utts if action == "ACCEPT" and validity == "VALID")
            cur = Counts(
                tot_correct=cur.tot_correct + tot_cor,
                succ_accept=cur.succ_accept + succ_acc,
                tot_incorrect=cur.tot_incorrect,
                succ_argue=cur.succ_argue,
            )
        else:
            # mode unknown -> ignore
            pass

        case_map[cid] = cur

    return out


# ============================
# Bootstrap & Permutation (paired)
# ============================
def paired_bootstrap_delta(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    """
    delta = SMed(a) - SMed(b) on dataset-level.
    bootstrap: resample case_ids with replacement, recompute delta.
    """
    # 원본 delta
    base_counts_a = Counts(0, 0, 0, 0)
    base_counts_b = Counts(0, 0, 0, 0)
    for pid in case_ids:
        base_counts_a = add_counts(base_counts_a, counts_map[pid][cid_a])
        base_counts_b = add_counts(base_counts_b, counts_map[pid][cid_b])
    delta_obs = medcobe_from_dataset_counts(base_counts_a) - medcobe_from_dataset_counts(base_counts_b)

    # bootstrap
    n = len(case_ids)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = Counts(0, 0, 0, 0)
        cb = Counts(0, 0, 0, 0)
        for j in idx:
            pid = case_ids[j]
            ca = add_counts(ca, counts_map[pid][cid_a])
            cb = add_counts(cb, counts_map[pid][cid_b])
        boot[i] = medcobe_from_dataset_counts(ca) - medcobe_from_dataset_counts(cb)

    lo = np.percentile(boot, 100 * (alpha / 2))
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(delta_obs), float(lo), float(hi)

def paired_permutation_delta_pvalue(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> Tuple[float, float]:
    """
    Paired permutation for nonlinear statistic:
    - for each case, with p=0.5 swap (a,b) counts
    - recompute dataset-level delta
    """
    # observed
    ca0 = Counts(0, 0, 0, 0)
    cb0 = Counts(0, 0, 0, 0)
    for pid in case_ids:
        ca0 = add_counts(ca0, counts_map[pid][cid_a])
        cb0 = add_counts(cb0, counts_map[pid][cid_b])
    obs = medcobe_from_dataset_counts(ca0) - medcobe_from_dataset_counts(cb0)

    # permutation null
    extreme = 0
    for _ in range(n_perm):
        ca = Counts(0, 0, 0, 0)
        cb = Counts(0, 0, 0, 0)
        swaps = rng.random(len(case_ids)) < 0.5
        for pid, sw in zip(case_ids, swaps):
            a = counts_map[pid][cid_a]
            b = counts_map[pid][cid_b]
            if sw:
                a, b = b, a
            ca = add_counts(ca, a)
            cb = add_counts(cb, b)

        d = medcobe_from_dataset_counts(ca) - medcobe_from_dataset_counts(cb)
        if abs(d) >= abs(obs):
            extreme += 1

    p = (extreme + 1) / (n_perm + 1)  # add-one smoothing
    return float(obs), float(p)


# ============================
# Analysis wrapper
# ============================
def analyze_dataset_level(annotated_json_path: str) -> pd.DataFrame:
    print("Loading counts from JSON...")
    counts_map = build_case_prompt_counts(annotated_json_path)
    print(f"  Loaded {len(counts_map)} cases")

    # 비교마다 필요한 case만 씀(완전한 페어만 유지)
    # 즉, C1과 Ck가 모두 존재하는 case들만 사용
    def paired_case_ids(cid_a: str, cid_b: str) -> List[str]:
        ids = []
        for pid, cmap in counts_map.items():
            if cid_a in cmap and cid_b in cmap:
                # dataset-level score 계산에 필요한 모드가 있는지 최소 체크
                a = cmap[cid_a]
                b = cmap[cid_b]
                # 둘 다 correct/incorrect utterance가 모두 있는 케이스만 포함(원하면 완화 가능)
                if (a.tot_correct > 0 and a.tot_incorrect > 0 and
                    b.tot_correct > 0 and b.tot_incorrect > 0):
                    ids.append(pid)
        return ids

    rng = np.random.default_rng(SEED)

    rows = []
    baseline = "C1"

    # baseline의 dataset-level medcobe (C1이 있는 case들 전체)
    base_ids = paired_case_ids("C1", "C1")
    # (C1,C1)는 항상 참이지만 위 조건 때문에 C1에서 두 모드 다 있는 케이스만 남음
    c1_counts = Counts(0, 0, 0, 0)
    for pid in base_ids:
        c1_counts = add_counts(c1_counts, counts_map[pid]["C1"])
    c1_med = medcobe_from_dataset_counts(c1_counts)

    print(f"  C1 baseline: medcobe={c1_med:.4f}, n_cases={len(base_ids)}")
    # C1 row
    rows.append({
        "cid": "C1",
        "medcobe_dataset": c1_med,
        "n_cases_used": len(base_ids),
        "delta_vs_C1": np.nan,
        "CI_low": np.nan,
        "CI_high": np.nan,
        "perm_p": np.nan,
        "perm_p_holm": np.nan,
        "sig_boot": np.nan,
        "sig_perm": np.nan,
    })

    pvals = []
    tmp_results = []

    for i in range(2, 17):
        cid = f"C{i}"
        ids = paired_case_ids(cid, baseline)
        print(f"  {cid}: n_cases={len(ids)}, bootstrap...", end=" ", flush=True)

        # dataset-level medcobe for cid using exactly the same paired ids (해석 일관성)
        ci_counts = Counts(0, 0, 0, 0)
        for pid in ids:
            ci_counts = add_counts(ci_counts, counts_map[pid][cid])
        med_k = medcobe_from_dataset_counts(ci_counts)

        # bootstrap CI (delta)
        delta, lo, hi = paired_bootstrap_delta(
            case_ids=ids,
            counts_map=counts_map,
            cid_a=cid,
            cid_b=baseline,
            rng=rng,
            n_boot=N_BOOT,
            alpha=ALPHA,
        )
        print("permutation...", end=" ", flush=True)

        # permutation p-value
        obs, p = paired_permutation_delta_pvalue(
            case_ids=ids,
            counts_map=counts_map,
            cid_a=cid,
            cid_b=baseline,
            rng=rng,
            n_perm=N_PERM,
        )

        print(f"done (delta={delta:.4f})")
        tmp_results.append({
            "cid": cid,
            "medcobe_dataset": med_k,
            "n_cases_used": len(ids),
            "delta_vs_C1": delta,
            "CI_low": lo,
            "CI_high": hi,
            "perm_p": p,
        })
        pvals.append(p)

    pvals = np.array(pvals, dtype=float)
    adj = holm_correction(pvals)

    for r, p_adj in zip(tmp_results, adj):
        r["perm_p_holm"] = p_adj
        r["sig_boot"] = (r["CI_low"] > 0) or (r["CI_high"] < 0)
        r["sig_perm"] = (p_adj < 0.05)
        rows.append(r)

    res = pd.DataFrame(rows)
    # 보기 좋게 정렬: delta 큰 순(단, C1은 맨 위)
    c1 = res[res["cid"] == "C1"]
    others = res[res["cid"] != "C1"].sort_values("delta_vs_C1", ascending=False)
    res = pd.concat([c1, others], ignore_index=True)
    return res


# ============================
# Run
# ============================
if __name__ == "__main__":
    model_map = load_model_map(EXPERIMENTS_YAML)
    short = resolve_short_model_name(MODEL_NAME, model_map)

    annotated_json_path = f"experiments/(main) {short}_combined_sample/evaluation/annotated_results.json"
    out_dir = Path(f"analysis/(main) paired bootstrap permutation_{short}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {short}...")
    res = analyze_dataset_level(annotated_json_path)
    print("\nResult:")
    print(res)

    out_csv = out_dir / f"{short}_paired_bootstrap_permutation.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")