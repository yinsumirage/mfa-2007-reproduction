# Provenance and export scope

Public export date: 2026-11-03.

The implementation was written from the paper's definitions with AI coding assistance. Third-party implementations were inspected and executed for matched numerical comparisons; their source was not copied into the core implementation and is not redistributed in this repository.

The core files `mfa.py`, `biased.py`, `gait.py`, `retrieval.py` and `retrieval_features.py` are byte-identical to the checked local snapshot. The test suite and experiment runners are also retained. Source and result hashes are in [reproduction-manifest.json](../reproduction-manifest.json).

Public-export adaptations:

1. `check_paper_contract.py` uses the shipped manifest and paper DOI instead of requiring a local licensed PDF. It prints its checks without writing a report into the frozen snapshot.
2. `visual_comparison.py` writes figures under `figures/` instead of a presentation asset folder.
3. Public README and protocol notes replace local presentation/build instructions.

The original paper PDFs, course slides, presentation, local environments, image caches and upstream working copies are not included. Published numerical results are the existing saved experiments, rather than new benchmark measurements made during publication.

The preserved graph rule remains class-versus-rest selection of k2 pairs for every MFA solver. Upstream variants were aligned only within the historical reference checks and do not change the main definition.
