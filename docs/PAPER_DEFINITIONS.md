# Paper definitions and fixed settings

Source: Xu et al., IEEE TIP 16(11), 2811–2821, 2007, [DOI](https://doi.org/10.1109/TIP.2007.906769). Page numbers below count the PDF from 1.

## Two graphs, Eqs. (5) and (6), PDF page 3

For each training sample, select k1 nearest neighbors within its class. If either direction selects an edge, include that edge in the undirected intrinsic graph S.

For each class c, pool pairs between samples in c and samples in all other classes, then select the k2 nearest pairs. The undirected union forms the penalty graph P. Distances used for graph construction are Euclidean distances in the input training features.

Classification fixes k1=3 and k2=50, within the paper's reported parameter ranges. There is no test-based parameter sweep. k2 is counted in pairs per class; duplicate selections become a single undirected edge.

For column-sample X, A = X Ls Xᵀ and B = X Lp Xᵀ, where L is degree minus adjacency. Eq. (7) minimizes tr(WᵀAW) / tr(WᵀBW), with positive denominator and the specified projection normalization. The implementation stores samples as rows.

## The paper's solver names, PDF page 7

- MFA-1 solves the Ratio Trace surrogate max tr[(WᵀAW)⁻¹(WᵀBW)] through the largest generalized eigenvectors of Bv=λAv. The inverse metric requires valid support. The classification implementation first retains 98% training PCA energy.
- MFA-2 follows Procedure 2. Restrict to positive support of A+B, set C=A+B, compute λ=tr(WᵀBW)/tr(WᵀCW), and update orthonormal columns using the leading eigenvectors of B−λC.

The graphs are shared. Multidimensional Ratio Trace and Trace Ratio are different formulations. A shared-PCA MFA-2 experiment is retained as a diagnostic, rather than replacing the main configuration.

## Matrix and retrieval variants

MMFA is Matrix-based Marginal Fisher Analysis: Y=UᵀXV, with alternating updates from Procedure 1. Its GED subproblems do not establish a joint global optimum.

MBA compacts positive neighbors and separates selected positive-negative pairs. KMBA kernelizes that representation. The retrieval experiment uses an explicit regularization and kernel-width convention; KBDA here is not the original paper's KDBDA method.

See [implementation notes](../code/implementation-notes.md) for numerical conventions and [experimental protocols](EXPERIMENTS.md) for the scope of each saved result.
