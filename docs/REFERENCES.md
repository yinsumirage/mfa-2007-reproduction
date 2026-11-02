# Optional reference comparisons

The main implementation is independent Python code. Historical matched checks used these upstream sources:

| Source | Pinned commit | Executed scope |
|---|---|---|
| [Rdimtools](https://github.com/kisungyou/Rdimtools) | `09f56cab121155bf9544e4665a764465861ecb49` | R neighborhood helper only |
| [Face_dynamics_ADOS](https://github.com/nz0001na/Face_dynamics_ADOS) | `1bb12e7392e13929cc8ad8b306af11af3fdc551c` | MFA/LGE under Octave |
| [2018-TIE-MMFA](https://github.com/XLearning-SCU/2018-TIE-MMFA) | `3032fbd9f2ae7ad080cc1a0d394a4195f6558242` | Multiple MFA Python routine |

They have different neighbor/pair counting conventions. The historical scripts compare matched definitions and distances; they are not three interchangeable official implementations of the 2007 paper. Multiple MFA is different from Matrix-based MFA.

To obtain local working copies at these commits:

```sh
python tools/fetch_references.py
```

Source is fetched into ignored `code/third_party/` folders. Preserve the upstream licenses and attribution; the MATLAB reading copy had no repository-wide license identified and is not vendored here. The manifest is [upstream-manifest.json](../code/upstream-manifest.json).

Running `python code/run_all.py --references` additionally expects R and Octave executables under `code/.reference-env/bin/`. The retained explicit Conda lock describes the original macOS ARM environment; it is not a cross-platform setup recipe. The default Python tests and experiments do not require this environment.
