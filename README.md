# `fl-entity-eval`: Entity-Aware Evaluation, Leakage Auditing, and Effective Sample Size Diagnostics for Intelligent-Driving Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](pyproject.toml)

Official software toolkit, 56-study English full-text audit table, size-matched leakage decoupling outputs, and federated scoring-protocol sensitivity benchmark accompanying:

> **Evaluation Validity and Data Leakage in Federated Learning for Intelligent Driving: An Audit, Benchmark, and Toolkit**  
> Kehao Liu, Hongfei Guo*, Yulong Zhao  
> *College of Intelligent Science and Technology, Inner Mongolia University of Technology, Hohhot, China*  
> (*Corresponding author: `ghf-2005@163.com`)

---

## Overview

Federated learning (FL) trains shared models across vehicles, drivers, and roadside units without centralising raw telemetry, yet partitioning data by entity during training does not ensure entity-isolated evaluation. When overlapping windows from one driver, trip, or road segment cross the train-test boundary, reported metrics reflect within-entity memorisation rather than generalisation to unseen deployments.

`fl-entity-eval` is a lightweight NumPy/Pandas evaluation package (with optional PyTorch and Flower index adapters) that implements the ten-item entity-aware reporting protocol described in the manuscript:

1. **`EntityGroupSplitter`** (`fl_entity_eval/splitter.py`): constructs whole-entity holdout splits (`driver_id`, `trip_id`, `subject_id`), chronological tail-block splits (`split_chronological`), size-matched paired splits (`S1`, `S1-matched`, `S2-matched`, `S2`) with built-in recording-isolation assertions, and SHA-256 JSON split manifests (`export_manifest`, `verify_manifest`).
2. **`LeakageAuditor`** (`fl_entity_eval/auditor.py`): measures driver, trip, and segment overlap alongside strict native-window physical temporal overlap ($|\Delta t| < W$) with automated `PASS`, `WARN`, and `FAIL` verdicts.
3. **`ICCDiagnostics`** (`fl_entity_eval/icc.py`): quantifies within-client pseudoreplication through one-way random-effects $\mathrm{ICC}(1)$, Kish $\mathrm{DEFF}$, Eldridge unequal-cluster $\mathrm{DEFF}_{\mathrm{unbal}}$, class-residualised signal $\mathrm{ICC}_{\mathrm{resid}}(1)$, post-training model-error $\mathrm{DEFF}$ (`compute_error_deff`), and window-overlap decay curves.
4. **`UnitLevelEvaluator`** (`fl_entity_eval/evaluator.py`): computes fixed-class-set (`fixed_macro_f1`) and present-class (`present_class_macro_f1`) entity macro-F1, window-, trip-, and entity-level mean and worst-entity summaries, federated protocol-sensitivity tables (`evaluate_protocol_sensitivity`), exact two-sided paired sign-flip tests, Holm-Bonferroni multiplicity adjustment, and an optional `cohen_kappa` helper for multi-coder audits.
5. **`adapters`** (`fl_entity_eval/adapters.py`): provides strict train-only per-channel standardisation (`train_only_standardize`), PyTorch `Subset` wrappers (`make_pytorch_subsets`), a dependency-free per-client index partitioner (`FlowerEntityPartitioner`) supporting both within-client session holdout and `unseen_client_frac` holdout without requiring `flwr`, and the post-hoc support-masked head-averaging diagnostic (`support_masked_head_avg`).

---

## Repository Structure

```text
fl-entity-eval/
├── fl_entity_eval/                    # Core Python package
│   ├── __init__.py
│   ├── splitter.py                    # EntityGroupSplitter and SHA-256 split manifests
│   ├── auditor.py                     # LeakageAuditor (strict |Δt| < window_sec default)
│   ├── icc.py                         # ICC(1), Kish/Eldridge DEFF, residualised and error DEFF
│   ├── evaluator.py                   # Fixed/present-class F1, protocol summaries, sign-flip tests
│   └── adapters.py                    # Train-only scaling, PyTorch/Flower adapters, head-avg probe
├── data/
│   └── full_text_audit_en.csv         # 56-study (+1 excluded survey) full-text coding and evidence notes
├── results/
│   ├── phase2_decoupling/             # 6-seed decoupling, raw-unit UAH, present-class F1, ICC/error DEFF
│   │   └── manifests/                 # Verified seed-1 S2 SHA-256 split manifests (UAH, HAR, WISDM)
│   ├── phase3_federated/              # Unstratified and class-stratified FL benchmarks and ablations
│   │   ├── unstratified/              # 6-seed unstratified S2-session (UAH, HAR, WISDM)
│   │   ├── stratified/                # 6-seed class-stratified S2-session (UAH, HAR)
│   │   ├── ditto_lambda_sweep/        # Matched 6-seed HAR Ditto proximal sweeps
│   │   ├── gru_backbone/              # Tuned GRU1DFLNet and matched 6-seed FedPer sync ablations
│   │   ├── har_ablations/             # Compute-matched pFedMe K=1 and 60-round CNN convergence checks
│   │   └── uah_ablations/             # UAH 5-channel IMU, Ditto/pFedMe sweeps, and road-regime splits
│   └── sensitivity/                   # GRU decoupling, WISDM 20 Hz resampling, 5-fold CV, UAH LODO
├── scripts/
│   ├── rebuild_uah_raw.py             # Rebuild UAH-DriveSet windows in raw physical units
│   ├── prepare_uci_har_npz.py         # Convert raw UCI HAR Inertial Signals to uci_har_windows.npz
│   ├── run_entity_decoupling.py       # Multi-dataset size-matched decoupling and ICC/DEFF sweep
│   ├── run_fl_protocol_sensitivity.py # Five-method FL benchmark and support-masked diagnostic
│   ├── run_sensitivity_checks.py      # GRU decoupling, WISDM 20 Hz, 5-fold subject CV, UAH LODO
│   ├── run_tier1_experiments.py       # Self-contained synthetic decoupling demonstration
│   └── compute_kappa_and_trends.py    # 56-study audit summary, temporal trends, and kappa utility
├── tests/
│   └── test_toolkit.py                # Unit test suite (9 tests covering all five modules)
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Installation and Quick Start

```bash
# Editable install (core toolkit requires only NumPy and Pandas; add [torch] or [experiments] for runners)
pip install -e .

# Run the unit test suite (9 tests)
python -m unittest discover -s tests -v

# Reproduce the 56-study full-text audit and vehicular/traffic-domain FL sensitivity summaries
python scripts/compute_kappa_and_trends.py

# Run the self-contained synthetic decoupling demonstration
python scripts/run_tier1_experiments.py --demo
```

### Minimal Python Usage

```python
import numpy as np
import pandas as pd
from fl_entity_eval import EntityGroupSplitter, LeakageAuditor, ICCDiagnostics

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "driver_id": np.repeat([f"D{i}" for i in range(6)], 40),
    "trip_id": np.repeat([f"T{i}" for i in range(24)], 10),
    "time_sec": np.tile(np.arange(10) * 15.0, 24),
    "accel_mag": rng.normal(loc=np.repeat(rng.normal(1.0, 0.3, 24), 10), scale=0.1),
})

tr_idx, te_idx = EntityGroupSplitter.split_whole_entity(
    df, entity_col="trip_id", test_frac=0.30, stratify_col="driver_id", seed=1
)
audit = LeakageAuditor.audit_split(
    df, tr_idx, te_idx, driver_col="driver_id", trip_col="trip_id", time_col="time_sec", window_sec=30.0
)
icc = ICCDiagnostics.compute_icc_and_neff(df["accel_mag"].to_numpy(), df["trip_id"].to_numpy())
print(audit["status"], f"DEFF={icc['deff']:.1f}x, N_eff={icc['n_eff']:.1f}")
```

### Reproducing Real-Dataset Benchmarks

To re-run the real-dataset pipelines (`run_entity_decoupling.py`, `run_sensitivity_checks.py`, and `run_fl_protocol_sensitivity.py`), point two environment variables to local copies of UAH-DriveSet, UCI HAR, and WISDM 1.1. Single-threaded OpenMP/MKL execution is recommended on CPU workstations to prevent worker contention:

```bash
export FL_ENTITY_EVAL_DATA_ROOT=/path/to/datasets_root   # expects data/processed/uci_har_windows.npz and data/raw/WISDM/wisdm-dataset.zip
export FL_ENTITY_EVAL_RAW_UAH=/path/to/uah_raw_npz       # expects windows_L30_raw.npz built via scripts/rebuild_uah_raw.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/run_entity_decoupling.py
python scripts/run_sensitivity_checks.py
python scripts/run_fl_protocol_sensitivity.py --split-mode stratified --datasets UAH HAR --seeds 1 2 3 4 5 6
```

---

## Summary of Main Empirical Findings

### 1. Full-Text Audit of 56 Empirical Driving-FL Studies (`data/full_text_audit_en.csv`)

Under a deterministic five-way split taxonomy separating 8 non-supervised, reinforcement-learning, or simulation studies from 48 supervised studies eligible for static splitting:

- **16/48 (33.3%)** supervised studies omit split criteria (`14/40`, 35.0%, in vehicular/traffic-domain FL).
- **20/48 (41.7%)** rely on within-client, random, or static global-test image splits (`17/40`, 42.5%, in vehicular/traffic-domain FL).
- **5/48 (10.4%)** hold out unseen entities or domains (`5/40`, 12.5%, in vehicular/traffic-domain FL).
- **46/56 (82.1%)** omit dispersion (SD, CI, or error bars) or formal statistical tests, and **44/56 (78.6%)** omit multi-run averages as well.

### 2. Size-Matched Leakage Decoupling and Pseudoreplication (`results/phase2_decoupling/`)

| Benchmark | Holdout Regime (`S2`) | Primary Metric | `S1-matched` | `S2-matched` | Pure Leakage ($\Delta_{\mathrm{leak}}$) | Sample-Size Confound ($\Delta_{\mathrm{size}}$) | Signal / Resid. / Error $\mathrm{DEFF}$ (50% Overlap) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UAH-DriveSet** (6 drivers, 40 trips, raw units) | Within-driver unseen trips | Trip-Vote Acc | 0.912 | 0.480 | **+0.432** | -0.010 | $19.5\times$ / $18.4\times$ / $14.3\times$ ($N_{\mathrm{eff}} = 5.13\%$) |
| **UCI HAR** (30 subjects, 400 sessions) | Cross-subject unseen entities | Worst-Subject F1 (present) | 0.813 | 0.760 | **+0.052** | -0.013 | $25.1\times$ / $16.4\times$ / $17.5\times$ ($N_{\mathrm{eff}} = 3.99\%$) |
| **WISDM 1.1** (51 subjects, 307 segments) | Cross-subject unseen entities | Subject-Mean F1 (present) | 0.928 | 0.565 | **+0.363** | -0.003 | $206.3\times$ / $177.5\times$ / $157.3\times$ ($N_{\mathrm{eff}} = 0.48\%$) |

All findings persist under a raw-unit bidirectional GRU backbone (`GRU1DNet`), uniform 20 Hz WISDM resampling, 5-fold subject-grouped cross-validation ($p \le 0.00045$), and UAH leave-one-driver-out unseen-driver holdout (`6/6` drivers, $p = 0.0313$).

### 3. Federated Scoring-Protocol Sensitivity (`results/phase3_federated/`)

- **Unstratified `S2-session`** (`15.3%` zero-support trips on UAH, `22.0%` on UCI HAR, and `97.7%` on the single-segment WISDM 1.1 stress test): local class-support sparsity causes steep `deployed` (local-model) drops below `FedAvg` on WISDM (`FedPer 0.018`, `Ditto 0.069`, `pFedMe 0.158` vs. `FedAvg 0.531`; all $p_{\mathrm{Holm}} < 10^{-4}$) and UCI HAR `FedPer` (`0.649` vs. `0.853`, $p_{\mathrm{Holm}} < 10^{-4}$), which `uniform` (global-model) scoring masks on the same checkpoints. The post-hoc support-masked diagnostic `FedPer-Guard` (`support_masked_head_avg`) recovers **+13.0 pp** on UCI HAR (`0.649 -> 0.779`), **+15.0 pp** on WISDM 1.1 (`0.018 -> 0.168`, $15\times$ unmasked averaging), and **+1.4 pp** on UAH (`0.278 -> 0.292`), diagnosing unsupported-class gradient contamination while remaining below `uniform` FedPer and `FedAvg` owing to fallback-row norm shrinkage.
- **Class-Stratified `S2-session`** (`0.0%` local zero-support): on laboratory UCI HAR (`Conv1DFLNet`), regularised full-model personalisation leads `FedAvg` across both 30-round (`0.826`: `Ditto λ=0.1/1.0/5.0: 0.859/0.908/0.913`, `pFedMe K=1/K=3: 0.837/0.900`) and converged 60-round (`0.909`: `pFedMe K=1: 0.934`, `Ditto λ=1.0: 0.936`; $p_{\mathrm{Holm}}=0.291$, n.s.) budgets, whereas head-only `FedPer` incurs a consistent ~2 pp penalty from final-round encoder overwrite (`60-round CNN: 0.880 -> 0.898`; matched 6-seed `GRU1DFLNet: 0.699 -> 0.719` vs. `FedAvg 0.860`) and retains a `-13.9 pp` deficit under high-learning-rate recurrent training. On real driving telemetry (**UAH-DriveSet**, `83.3%` locally unseen class-road pairs), personalised models still trail `FedAvg` (`0.510`, window macro-F1 `0.475`; all $p_{\mathrm{Holm}} \ge 0.125$ at $n=6$): pruning GPS speed and yaw raises pooled trip accuracy (`+16.7` to `+17.6 pp`) largely via `NORMAL` recall surges (`0.750-1.000`) while degrading global `FedAvg`/`FedProx` drowsy/aggressive recall (`-15.2`/`-12.1 pp`; only 5-channel `pFedMe` lifts drowsy/aggressive recall to `0.439`, window macro-F1 `0.459`), whereas retaining all seven channels with strong global anchoring (`Ditto λ=10.0`, `0.461`) restores drowsy/aggressive recall (`0.242 -> 0.439`) and 7-channel `FedAvg`/`FedProx` retain the highest overall window macro-F1 (`0.475-0.477`).

---

## Citation

```bibtex
@article{liu2026flentityeval,
  title   = {Evaluation Validity and Data Leakage in Federated Learning for Intelligent Driving: An Audit, Benchmark, and Toolkit},
  author  = {Liu, Kehao and Guo, Hongfei and Zhao, Yulong},
  year    = {2026},
  note    = {Manuscript under review; software released under the MIT Licence},
  url     = {https://github.com/xs2822835114/fl-entity-eval}
}
```
