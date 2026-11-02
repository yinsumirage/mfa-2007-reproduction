# Saved experiments and reproduction scope

The saved measurements were produced on CPU in the local study. Source, fixed splits, predictions and result hashes are retained. Publishing the repository did not rerun the complete benchmark suite.

## Image classification

Settings: k1=3, k2=50, seeds 16/17/18, Euclidean 1-NN. Vector outputs are 16D, matrix MMFA is 4×4 with 10 updates; Digits LDA has 9D because it has 10 classes.

- Digits: 600 training and 540 test images per split, pixels divided by 16.
- Olivetti: 6 training and 4 test images per person, 240/160 total; fixed 2×2 block means reduce 64×64 images to 32×32.
- Fit preprocessing on training images only. MFA-1 retains 98% PCA energy; main MFA-2 uses positive A+B support.

| Method | Digits mean accuracy (%) | Olivetti mean accuracy (%) |
|---|---:|---:|
| Raw pixels | 98.02 | 91.25 |
| PCA | 97.28 | 88.54 |
| LDA | 94.32 | 96.25 |
| MFA-1 | 96.91 | 94.17 |
| MFA-2 | 81.67 | 92.50 |
| MMFA | 95.43 | 90.21 |
| MFA-2 / shared PCA diagnostic | 96.54 | 94.79 |

The shared-PCA diagnostic keeps the solver and split fixed and changes the preprocessing subspace. It does not replace the main result or establish that preprocessing explains every difference.

[Classification JSON](../results/round2/classification.json), [split indices](../results/round2/classification_splits.npz) and [predictions](../results/round2/classification_predictions.npz) preserve each evaluation. Standard deviations in JSON are sample SD across three splits, not confidence intervals.

## Feedback retrieval

A pinned Corel1K mirror provides 10 classes of 100 images. The source commit and individual file hashes are recorded in [the data manifest](../results/round2/corel1k-source.json). This is a third-party mirror, not the original paper's Corel22K data. Original images and generated descriptor caches are not distributed here.

Descriptors: 256 HSV + 24 Haar + 5 edge features, block L2 normalization. All embeddings are 2D. Use explicit ridge=1e-3×tr(A+B)/p and fixed RBF width from 200 fixed unlabeled images' median squared distance. There is no test-selected kernel-width sweep.

Held-out evaluation uses 400 feedback-pool images and 600 evaluation images. There are 100 queries. Each query is a known positive during fitting but excluded from its gallery, leaving 599 fixed candidates. Other evaluation labels never enter fitting. Each round selects at most 5 positive and 5 negative labels from the top 40 unseen feedback candidates; shortages are retained, without looking elsewhere.

| Method | Initial P@20 (%) | Round 5 P@20 (%) |
|---|---:|---:|
| BDA | 60.80 | 80.50 |
| KBDA | 60.80 | 82.20 |
| MBA | 60.80 | 54.60 |
| KMBA | 60.80 | 64.30 |

[Held-out result JSON](../results/round2/heldout_retrieval.json) and [pool/query indices](../results/round2/heldout_retrieval_split.npz) are retained. KBDA is regularized kernel BDA; the original paper's KDBDA is a different method. Separate nine-round, seen-feedback and identical-feedback diagnostics are retained in the same result directory; they do not select the main configuration.

The ordering differs from the paper. Training-objective optimization does not guarantee better test classification or retrieval performance.

## Gait components

Eq. (8) averages complete cycles with supplied cycle lengths; Eq. (9) takes the median over probe cycles of their nearest gallery-cycle distances. This sequence score is directional.

The [synthetic integration fixture](../results/round2/gait_integration.json) uses a supplied 12-frame period, 96 training cycle images, 24 gallery and 12 probe sequences. It checks the connection between components, not real-world USF recognition performance. The matrix fixture explicitly uses ridge=1e-6 for rank-deficient backgrounds.

## Two-dimensional teaching comparison

Digits classes 0/1/2, 179 training and 162 held-out test images, fixed seed-16 split, shared train-only PCA98 (29D), 2D orthonormal learned subspaces. Graphs use the original training features. This display metric differs from each method's native coordinate normalization.

| Method | Correct / test images | Accuracy (%) |
|---|---:|---:|
| PCA | 151 / 162 | 93.21 |
| LDA | 161 / 162 | 99.38 |
| LPP | 152 / 162 | 93.83 |
| MFA-1 | 157 / 162 | 96.91 |

[Comparison data](../results/visual_comparison/comparison.json), [embeddings](../results/visual_comparison/projections.npz) and the [offline demo](../demo/index.html) are included. Query 56 was fixed before fitting; example 51 was selected afterward to illustrate disagreement, without retuning.

## What is not reproduced

The original USF gait protocol, official silhouettes/period inputs, original Corel22K descriptors and the paper's full benchmark tables/curves are not reproduced. Dataset access, preprocessing and numerical conventions limit comparisons. See [the original paper](https://doi.org/10.1109/TIP.2007.906769) for the original measurements.
