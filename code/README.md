# Code entry points

Install from the repository root using [the main README](../README.md).

| File | Purpose |
|---|---|
| `mfa.py` | Graphs, MFA-1, MFA-2 and matrix MMFA |
| `biased.py` | MBA/KMBA and BDA/KBDA |
| `gait.py` | Gait-cycle averaging and sequence matching |
| `retrieval.py` | Feedback loops and held-out evaluation |
| `retrieval_features.py` | HSV/Haar/edge descriptors |
| `prepare_data.py` | Download and verify data |
| `run_experiment.py` | Small Digits experiment |
| `run_all.py` | Checks and full substitute-task experiments |
| `visual_comparison.py` | Controlled 2D comparison |
| `check_paper_contract.py` | Definition fixtures and frozen snapshot hashes |
| `test_*.py` | 28 unit and numerical checks |

Optional upstream comparisons under `reference_checks/` need [separate setup](../docs/REFERENCES.md). The default reproduction uses only the Python implementation.
