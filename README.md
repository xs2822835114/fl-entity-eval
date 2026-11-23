# `fl-entity-eval`: Entity-Aware Evaluation, Leakage Auditing, and ICC Diagnostics for Intelligent-Driving Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](pyproject.toml)

Official code, 56-study full-text audit table, and multi-dataset decoupling benchmark outputs for:

> **Evaluation Validity, Data Leakage, and Effective Sample Size in Federated Learning for Intelligent Driving: A Full-Text Audit, Decoupled Benchmark, and Open Toolkit**  
> Kehao Liu, Hongfei Guo*, Yulong Zhao  
> *College of Intelligent Science and Technology, Inner Mongolia University of Technology, Hohhot, China*

---

## Key Capabilities

`fl-entity-eval` is a lightweight, pure-NumPy/Pandas evaluation harness (compatible with PyTorch and Flower pipelines) providing four core modules:

1. **`EntityGroupSplitter`** — Generates whole-entity holdout splits (`driver_id`, `trip_id`, `subject_id`), size-matched paired splits (`S1`, `S1-matched`, `S2-matched`, `S2`) to decouple pure leakage from sample-size confounding, and immutable SHA-256 JSON split manifests.
2. **`LeakageAuditor`** — Automatically computes driver/trip/segment overlap and temporal-proximity leakage rates with `PASS` / `WARN` / `FAIL` verdicts.
3. **`ICCDiagnostics`** — Quantifies within-client pseudoreplication via one-way random-effects $\mathrm{ICC}(1)$, Design Effect ($\mathrm{DEFF}$), effective independent sample size ($N_{\mathrm{eff}}$), and window-overlap decay curves.
4. **`UnitLevelEvaluator`** — Reports window-, trip-, and entity-level mean/worst metrics alongside exact two-sided paired sign-flip tests, Holm–Bonferroni correction, and unweighted Cohen's $\kappa$.

---

## Repository Structure

```text
fl-entity-eval/
├── fl_entity_eval/                    # Core Python package (pure NumPy/Pandas)
│   ├── __init__.py
│   ├── splitter.py                    # EntityGroupSplitter + SHA-256 manifests
│   ├── auditor.py                     # LeakageAuditor (PASS/WARN/FAIL)
│   ├── icc.py                         # ICC(1), DEFF, N_eff & overlap sweeps
│   └── evaluator.py                   # Unit-level metrics & exact sign-flip tests
├── data/
│   └── Tier1_Master_57_FullText_Corpus.csv  # Complete 56-study (+1 survey) full-text audit
├── results/
│   └── phase2_decoupling/             # Real 6-seed outputs (UAH-DriveSet, UCI HAR, WISDM 1.1)
├── scripts/
│   ├── run_tier1_experiments.py       # Demo & reproducible decoupling harness
│   └── compute_kappa_and_trends.py    # Audit trend & dual-coder kappa helper
├── tests/
│   └── test_toolkit.py                # Unit test suite
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Run unit tests
python -m unittest discover -s tests -v

# Run the built-in synthetic decoupling demo
python scripts/run_tier1_experiments.py --demo
```

---

## Summary of Main Empirical Findings

| Benchmark | Holdout Regime (`S2`) | Metric | `S1-matched` | `S2-matched` | Pure Leakage ($\Delta_{\mathrm{leak}}$) | Sample-Size Confound ($\Delta_{\mathrm{size}}$) | $\mathrm{DEFF}$ (50% Overlap) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UAH-DriveSet** (6 drivers, 40 trips) | Within-driver unseen trips | Trip-Vote Acc | 0.920 | 0.500 | **+0.420** | +0.020 | $17.2\times$ ($N_{\mathrm{eff}}$ = 5.82%) |
| **UCI HAR** (30 subjects, 400 sessions) | Cross-subject unseen entities | Worst-Subject F1 | 0.816 | 0.734 | **+0.082** | -0.007 | $25.1\times$ ($N_{\mathrm{eff}}$ = 3.99%) |
| **WISDM 1.1** (51 subjects, 307 segments) | Cross-subject unseen entities | Subject-Mean F1 | 0.916 | 0.557 | **+0.359** | -0.000 | $206.3\times$ ($N_{\mathrm{eff}}$ = 0.48%) |

All raw per-seed CSV tables are included under `results/phase2_decoupling/`.

---

## Citation

If you use `fl-entity-eval` or the 56-study full-text audit table in your research, please cite:

```bibtex
@article{liu2026flentityeval,
  title   = {Evaluation Validity, Data Leakage, and Effective Sample Size in Federated Learning for Intelligent Driving: A Full-Text Audit, Decoupled Benchmark, and Open Toolkit},
  author  = {Liu, Kehao and Guo, Hongfei and Zhao, Yulong},
  year    = {2026},
  note    = {Manuscript under preparation; toolkit released under MIT License}
}
```
