import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# ============================
# Model
# ============================

def load_model_map(experiments_yaml_path: str) -> dict[str, str]:
    cfg = yaml.safe_load(Path(experiments_yaml_path).read_text(encoding="utf-8"))
    model_map = cfg.get("models", cfg)
    return model_map

def resolve_short_model_name(full_model_name: str, model_map: dict[str, str]) -> str:
    inv = {v: k for k, v in model_map.items()}
    return inv.get(full_model_name, full_model_name.split("/")[-1].replace("-", "_"))

EXPERIMENTS_YAML = "./resources/experiments.yaml"
MODEL_NAME = "openai/o3"

model_map = load_model_map(EXPERIMENTS_YAML)
SHORT_MODEL_NAME = resolve_short_model_name(MODEL_NAME, model_map)

print(SHORT_MODEL_NAME)  # "o3"

# ============================
# Config
# ============================

N_BOOT = 20000
N_PERM = 50000
SEED = 42

_T1_ORDER = ["B", "B_COT", "B_CL", "B_COT_CL"]
_T2_ORDER = ["B", "B_CL", "B_SR", "B_CL_SR"]


# ============================
# experiment_id -> C1~C16
# ============================

def exp_id_to_cid(experiment_id: str) -> str:
    try:
        rhs = experiment_id.split("-", 1)[1]
        t1, t2 = rhs.split("__", 1)
        r = _T1_ORDER.index(t1)
        c = _T2_ORDER.index(t2)

        return f"C{r*4 + c + 1}"

    except Exception:
        return None


# ============================
# MEDCOBE score per dialogue
# ============================

def compute_medcobe_dialogue(log):
    mode = log["mode"]
    dialogue = log["dialogue"]
    scores = []

    for turn in dialogue:
        if turn.get("role") != "AI":
            continue

        action = turn.get("action")
        validity = turn.get("validity")

        if mode == "correct":
            score = int(action == "ACCEPT" and validity == "VALID")
        elif mode == "incorrect":
            score = int(action == "ARGUE" and validity == "VALID")
        else:
            continue

        scores.append(score)

    if len(scores) == 0:
        return np.nan

    return np.mean(scores)


# ============================
# Load JSON -> dataframe
# ============================

def load_table(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    logs = data[MODEL_NAME]["logs"]
    rows = []

    for log in logs:
        cid = exp_id_to_cid(log["experiment_id"])

        if cid is None:
            continue

        rows.append({
            "pair_id": log["case_id"] + "_" + log["mode"],
            "cid": cid,
            "medcobe": compute_medcobe_dialogue(log)
        })

    return pd.DataFrame(rows)


# ============================
# bootstrap CI
# ============================

def paired_bootstrap_ci(x, y):
    rng = np.random.default_rng(SEED)
    diffs = x - y
    n = len(diffs)
    boot = np.empty(N_BOOT)

    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(diffs[idx])

    delta = np.mean(diffs)
    ci_low = np.percentile(boot, 2.5)
    ci_high = np.percentile(boot, 97.5)

    return delta, ci_low, ci_high


# ============================
# permutation test
# ============================

def paired_permutation_test(x, y):
    rng = np.random.default_rng(SEED)
    diffs = x - y
    observed = np.mean(diffs)
    count = 0

    for _ in range(N_PERM):
        signs = rng.choice([-1,1], size=len(diffs))
        perm = np.mean(diffs * signs)
        if abs(perm) >= abs(observed):
            count += 1
    
    p = (count + 1) / (N_PERM + 1)
    return observed, p


# ============================
# Holm correction
# ============================

def holm_correction(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    max_val = 0

    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        max_val = max(max_val, val)
        adj[idx] = min(max_val, 1.0)

    return adj


# ============================
# Full analysis
# ============================

def analyze(json_path):

    df = load_table(json_path)
    pivot = df.pivot(index="pair_id", columns="cid", values="medcobe")
    baseline = pivot["C1"]
    results = []

    for i in range(2,17):

        cid = f"C{i}"
        x = pivot[cid]
        mask = (~x.isna()) & (~baseline.isna())

        x = x[mask].values
        y = baseline[mask].values

        delta, ci_low, ci_high = paired_bootstrap_ci(x, y)
        observed, p = paired_permutation_test(x, y)

        results.append({
            "cid": cid,
            "delta_vs_C1": delta,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "perm_p": p
        })

    res = pd.DataFrame(results)
    res["perm_p_holm"] = holm_correction(res["perm_p"].values)
    res["sig_boot"] = (res["CI_low"] > 0) | (res["CI_high"] < 0)
    res["sig_perm"] = res["perm_p_holm"] < 0.05
    res = res.sort_values("delta_vs_C1", ascending=False)

    return res


# ============================
# Run
# ============================

json_path = f"experiments/(main) {SHORT_MODEL_NAME}_combined_sample/evaluation/annotated_results.json"

results = analyze(json_path)

print(results)

results.to_csv(f"analysis/(main) paired bootstrap permutation_{SHORT_MODEL_NAME}/{SHORT_MODEL_NAME}_paired_bootstrap_permutation_result.csv", index=False)