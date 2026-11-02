# Implementation conventions

The graph and objective definitions follow [the paper](../docs/PAPER_DEFINITIONS.md). Core numerical files were exported unchanged from the checked implementation.

- Samples are rows, float64. Construct graphs before projection preprocessing. Center using training data only.
- Equal distances use stable index ordering. Invalid neighborhood sizes raise errors rather than silently truncating.
- Numerical zero directions are excluded with relative rank tolerance 1e-12.
- MFA-1 uses 98% training PCA energy, symmetric generalized eigendecomposition and Euclidean-normalized columns. Default ridge is zero.
- MFA-2 uses the positive support of A+B. High-dimensional thin-SVD reduction preserves the exact training row space. Stopping checks the ratio change and normalized spectral residual, avoiding sign-dependent eigenvector comparisons.
- Matrix MFA uses fixed alternating updates, not a test-selected iteration count. Classification uses 4×4 output and 10 updates.
- The 2D visualization uses a common train-only PCA98 space and orthonormal learned subspaces for its display metric. This is separate from the 10-class, 16D classification experiment.
- Retrieval uses fixed descriptor conventions, ridge scaling and RBF width; see [protocols](../docs/EXPERIMENTS.md). Kernel directions are normalized by RKHS norm.

The suite checks literal pair sums, independent spectral-root calculations, singular support, out-of-sample transforms, matrix projections and feedback isolation. It is numerical evidence, not a proof of every possible input case.
