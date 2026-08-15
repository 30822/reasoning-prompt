# Structured Reasoning Prompts Improve Clinician–AI Collaboration: A Controlled Simulation Study

![Figure 1](fig1.png)

Code accompanying the paper:

> **Structured Reasoning Prompts Improve Clinician–AI Collaboration: A Controlled Simulation Study**

This repository implements a controlled two-turn simulation of clinician–AI collaboration. A clinician simulator holds either a **correct** or an **incorrect** initial belief; an AI partner responds under a 4 × 4 matrix of structured reasoning prompts. Dialogues are judged turn-by-turn, and collaboration quality is summarized with **$P_\text{collab}$**.

Target models in this release: **openai/o3** and **deepseek/deepseek-r1-0528**.

---

## Overview

Each case is run in two belief conditions:

| Condition | Clinician simulator belief |
|-----------|----------------------------|
| `correct` | Ground-truth option |
| `incorrect`   | A distractor option |

The AI partner is prompted with one cell of a **4 × 4** design:

| Turn 1 \ Turn 2 | B | CL | SR | CL+SR |
|-----------------|---|------|------|---------|
| **B** | P1 | P2 | P3 | P4 |
| **CoT** | P5 | P6 | P7 | P8 |
| **CL** | P9 | P10 | P11 | P12 |
| **CoT+CL** | P13 | P14 | P15 | P16 |

- **B**: baseline reply  
- **CoT**: chain-of-thought  
- **CL**: checklist
- **SR**: self-revision  

The pipeline is:

```
cases JSON
    │
    ├─► [1] Solo          AI answers the MCQ alone
    ├─► [2] Simulation    2-turn clinician–AI dialogue (16 prompt cells × 2 beliefs)
    ├─► [3] Team          clinician makes a final decision from the transcript
    ├─► [4] Evaluation    LLM judge labels each AI turn (ARGUE/ACCEPT × VALID/INVALID)
    └─► [5] Pcollab       √(valid_argumentation × valid_acceptance) per cell
```

---

## Repository structure

```
.
├── conf.d/conf.example              # API key template (copy to conf.yaml)
├── resources/
│   ├── experiments.yaml             # models + 4×4 cell list (16 cells × 2 models)
│   ├── data/sample_data.json        # one example case (schema only)
│   └── prompt/
│       ├── simulator_prompts.yaml   # clinician simulator + AI partner prompts
│       ├── final_decision_prompts.yaml
│       └── evaluator_prompts.yaml   # turn-level judge
├── scripts/run_openrouter.sh        # end-to-end runner
└── src/
    ├── solo_performance_openrouter.py
    ├── simulation_openrouter.py
    ├── team_performance_openrouter.py
    ├── evaluation.py
    ├── collaborative_performance.py
    ├── analysis.py                     # bootstrap, permutation, component, interaction, correlation
    ├── openrouter_client.py
    └── utils.py
```

---

## Requirements

- Python 3.10+
- An **OpenAI** API key (`openai/*` models are called directly)
- An **OpenRouter** API key (all other models)

Install:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure keys (do not commit `conf.yaml`):

```bash
cp conf.d/conf.example conf.d/conf.yaml
# edit conf.d/conf.yaml
```

```yaml
openai:
  key: YOUR_OPENAI_API_KEY
openrouter:
  key: YOUR_OPENROUTER_API_KEY
```

---

## Data

The study used JAMA Clinical Challenge multiple-choice cases. **Full case text is not redistributed** (copyright).

`resources/data/sample_data.json` is a **schema example** (one case) so the pipeline can be smoke-tested. To reproduce the paper, assemble cases into the same JSON list format and pass the file with `--input`:

```json
{
  "case_id": "5000",
  "caption": "Image caption / figure description",
  "scenario": "Clinical vignette",
  "options": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "correct_option": "D",
  "answer": "Full text of the correct option",
  "distractors": ["...", "...", "..."],
  "explanation": "Published rationale (used by the judge)"
}
```

`case_id` must be unique. `distractors` are required for the `incorrect` belief condition.

---

## Running experiments

Make the runner executable once:

```bash
chmod +x scripts/run_openrouter.sh
```

### Smoke test (bundled example case)

```bash
./scripts/run_openrouter.sh pcollab \
  --model openai/o3 \
  --input resources/data/sample_data.json \
  --max-concurrent 4
```

### Full pipeline (paper setting)

Point `--input` at your case file. Defaults are `openai/o3` and `deepseek/deepseek-r1-0528`:

```bash
./scripts/run_openrouter.sh pcollab \
  --input path/to/your_cases.json \
  --max-concurrent 10
```

Or run one model at a time:

```bash
./scripts/run_openrouter.sh pcollab --model openai/o3 --input path/to/your_cases.json
./scripts/run_openrouter.sh pcollab --model deepseek/deepseek-r1-0528 --input path/to/your_cases.json
```

Model identifiers must match `resources/experiments.yaml` (`models:`). Each listed model has **16** cells (P1–P16); the simulator always runs both `correct` and `incorrect` beliefs.

### Individual stages

| Command | What it does |
|---------|----------------|
| `./scripts/run_openrouter.sh solo ...` | Solo accuracy only |
| `./scripts/run_openrouter.sh team ...` | Final clinician decision from existing simulation logs |
| `./scripts/run_openrouter.sh pcollab ...` | Steps 1–5 |

Completed per-model files are skipped on rerun (resume). Delete the corresponding file under `output/` to force a rerun.

**Note.** This pipeline issues many LLM calls (16 cells × 2 beliefs × 2 turns × N cases, plus solo, team decision, and judging). Start with `sample_data.json` before a full run.

---

## Outputs

All artifacts are written under `output/` (gitignored):

| Path | Contents |
|------|----------|
| `output/solo/by_model/` | Per-model solo logs and accuracy |
| `output/simulation/by_model/` | Two-turn dialogues |
| `output/team/by_model/` | Dialogues + clinician final decision |
| `output/evaluation/annotated_results.json` | Judge labels on each AI turn |
| `output/pcollab/final_pcollab_evaluation.csv` | Per-model, per-cell metrics |

CSV columns: `Model`, `Experiment` (P1–P16), `N_dialogues_judged`, `Solo Accuracy`, `Team Accuracy`, `Valid Argumentation`, `Valid Acceptance`, `Pcollab`.

- **Valid Argumentation:** share of AI turns in the `incorrect` condition labeled `ARGUE` and `VALID`  
- **Valid Acceptance:** share of AI turns in the `correct` condition labeled `ACCEPT` and `VALID`  
- **Pcollab:** geometric mean of valid argumentation and valid acceptance  

---

## Analysis

After judging, run the statistical analyses from `src/analysis.py` (no figures):

```bash
python src/analysis.py \
  --annotated output/evaluation/annotated_results.json \
  --out-dir output/analysis
```

Defaults match the paper setting: 20,000 bootstrap draws and 50,000 permutations vs **P1**, seed 42. Use `--model` to restrict to one model key; `--n-boot` / `--n-perm` to change resampling size.

| Output | Contents |
|--------|----------|
| `output/analysis/bootstrap_permutation.csv` | Paired bootstrap CI and permutation tests vs P1 (Holm-adjusted) |
| `output/analysis/component_level.csv` | Matched-pair component effects (CoT, CL1, CL2, SR) |
| `output/analysis/interaction.csv` | Two-way cell means and interaction contrasts |
| `output/analysis/correlation.csv` | Spearman / Kendall correlation of prompt effects across models |

Each table reports **Pcollab**, **Valid Argumentation**, and **Valid Acceptance**. Correlation requires at least two models in the annotated file. 

---

## Prompt and experiment configuration

- Prompt text: `resources/prompt/simulator_prompts.yaml` (`variant_addons` + turn templates).  
- Cell list: `resources/experiments.yaml`.  
- Clinician final-decision prompt: `resources/prompt/final_decision_prompts.yaml`.  
- Judge prompt: `resources/prompt/evaluator_prompts.yaml`.

The default clinician simulator is `openai/gpt-4o`. The default turn-level judge is `deepseek/deepseek-v3.2`. Both are set in `scripts/run_openrouter.sh`.

---

## Citation

If you use this code, please cite:

> Structured Reasoning Prompts Improve Clinician–AI Collaboration: A Controlled Simulation Study.

```
@article{structured_reasoning_clinician_ai,
  title   = {Structured Reasoning Prompts Improve Clinician--AI Collaboration: A Controlled Simulation Study},
  year    = {2026}
}
```
