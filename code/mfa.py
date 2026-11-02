"""CPU reimplementation of Xu et al., TIP 2007, Eqs. (5)-(7), Procedures 1/2.

New implementation from the paper. Upstream code was read for comparison, not
copied. Samples are rows, float64. Ties are broken by sample index. No silent
regularization or test-data access. See implementation-notes.md for deviations.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


def _integer(x, name):
    if not isinstance(x, Integral) or isinstance(x, bool) or x < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(x)


def _data(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite samples-by-features matrix")
    return x


def _sym(x):
    return (x + x.T) * 0.5


def laplacian(adjacency):
    s = np.asarray(adjacency, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all():
        raise ValueError("Adjacency must be finite and square")
    if np.any(s < 0) or not np.allclose(s, s.T):
        raise ValueError("Expected an undirected nonnegative graph")
    return np.diag(s.sum(axis=1)) - s


@dataclass
class Graphs:
    within: np.ndarray
    between: np.ndarray

    def scatters(self, x):
        x = _data(x)
        if len(x) != len(self.within):
            raise ValueError("Training graph/sample count mismatch")
        return (_sym(x.T @ laplacian(self.within) @ x),
                _sym(x.T @ laplacian(self.between) @ x))


def build_graphs(x, labels, k1=3, k2=50):
    """Paper Eq. (5) and class-versus-rest pair selection in Eq. (6)."""
    x = _data(x)
    labels = np.asarray(labels)
    k1, k2 = _integer(k1, "k1"), _integer(k2, "k2")
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError("One label is required per training sample")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError("Labels must be finite")
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    dist = cdist(x, x, metric="sqeuclidean")
    within = np.zeros((len(x), len(x)), dtype=np.float64)
    between = np.zeros_like(within)
    for c in classes:
        inside = np.flatnonzero(labels == c)
        outside = np.flatnonzero(labels != c)
        if len(inside) <= k1 or len(inside) * len(outside) < k2:
            raise ValueError("Not enough samples/pairs for requested k1 or k2")
        for i in inside:
            others = inside[inside != i]
            nearest = others[np.argsort(dist[i, others], kind="stable")[:k1]]
            within[i, nearest] = within[nearest, i] = 1
        # Each class competes with ALL other classes together, not per point
        # or per class-pair. Symmetrize by the union of selected undirected pairs.
        flat = np.argsort(dist[np.ix_(inside, outside)].ravel(), kind="stable")[:k2]
        ii, jj = np.unravel_index(flat, (len(inside), len(outside)))
        rows, cols = inside[ii], outside[jj]
        between[rows, cols] = between[cols, rows] = 1
    return Graphs(within, between)


def trace_ratio(a, b, w):
    denominator = float(np.trace(w.T @ b @ w))
    if denominator <= 0:
        return float("inf")
    return float(np.trace(w.T @ a @ w) / denominator)


def positive_subspace(a, rtol=1e-12):
    values, vectors = eigh(_sym(a))
    if values[-1] <= 0:
        raise ValueError("No positive scatter subspace")
    if values[0] < -rtol * values[-1] * 100:
        raise ValueError("Scatter is not positive semidefinite")
    return vectors[:, values > rtol * values[-1]]


def ged_projection(within, between, ndim, ridge=0.0):
    """Ratio Trace surrogate: largest generalized eigenvectors of (B,A).

    Each column is Euclidean-normalized. ridge is explicit, relative to mean
    within variance; default 0 leaves the original quadratic form unchanged.
    """
    ndim = _integer(ndim, "ndim")
    if ndim > len(within) or ridge < 0:
        raise ValueError("Invalid output dimension or ridge")
    metric = _sym(within).copy()
    if ridge:
        metric += ridge * np.trace(metric) / len(metric) * np.eye(len(metric))
    if eigh(metric, eigvals_only=True)[0] <= 0:
        raise ValueError("GED metric is singular; use explicit preprocessing/ridge")
    _, v = eigh(_sym(between), metric)
    w = v[:, -ndim:][:, ::-1]
    return w / np.linalg.norm(w, axis=0)


@dataclass
class TraceSolution:
    projection: np.ndarray
    history: list
    converged: bool
    spectral_residual: float
    retained_rank: int


def solve_trace_ratio(a, b, ndim, seed=16, max_iter=100, tol=1e-10,
                      rtol=1e-12, initial=None):
    """Procedure 2: between/total ratio and ordinary trace-difference EVD.

    Floating-point stopping uses lambda change plus the spectral root residual,
    rather than raw eigenvector difference (which is sign/basis dependent).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Scatter matrices must have identical square shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scatter matrices must be finite")
    ndim, max_iter = _integer(ndim, "ndim"), _integer(max_iter, "max_iter")
    if tol <= 0 or rtol <= 0:
        raise ValueError("Tolerances must be positive")
    q = positive_subspace(a + b, rtol)
    rank = q.shape[1]
    if ndim > rank:
        raise ValueError("ndim exceeds positive scatter rank")
    aw, bw = _sym(q.T @ a @ q), _sym(q.T @ b @ q)
    total = aw + bw
    if initial is None:
        initial = np.random.default_rng(seed).normal(size=(rank, ndim))
    else:
        initial = q.T @ np.asarray(initial, float)
        if initial.shape != (rank, ndim) or np.linalg.matrix_rank(initial) < ndim:
            raise ValueError("Initial projection has invalid shape or rank")
    u = np.linalg.qr(initial, mode="reduced")[0]

    def ratio(z):
        return float(np.trace(z.T @ bw @ z) / np.trace(z.T @ total @ z))

    lam = ratio(u)
    history = [{"iteration": 0, "lambda": lam,
                "within_between": trace_ratio(aw, bw, u)}]
    converged = False
    residual = float("inf")
    scale = max(np.linalg.norm(bw, 2) + np.linalg.norm(total, 2), np.finfo(float).tiny)
    for iteration in range(1, max_iter + 1):
        _, u = eigh(_sym(bw - lam * total), subset_by_index=(rank - ndim, rank - 1))
        new_lam = ratio(u)
        if new_lam < lam - 1e-10:
            raise ArithmeticError("Trace Ratio lambda decreased beyond rounding")
        root_eigenvalues = eigh(_sym(bw - new_lam * total), eigvals_only=True)
        residual = float(abs(root_eigenvalues[-ndim:].sum()) / (ndim * scale))
        history.append({"iteration": iteration, "lambda": new_lam,
                        "within_between": trace_ratio(aw, bw, u),
                        "normalized_spectral_residual": residual})
        if abs(new_lam - lam) <= tol and residual <= tol * 10:
            converged = True
            break
        lam = new_lam
    return TraceSolution(q @ u, history, converged, residual, rank)


@dataclass
class MFAModel:
    mean: np.ndarray
    projection: np.ndarray
    graphs: Graphs
    solver: str
    trace_solution: TraceSolution | None = None
    pca_rank: int | None = None

    def transform(self, x):
        x = _data(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("Input feature dimension mismatch")
        return (x - self.mean) @ self.projection


def fit_mfa(x, labels, ndim=16, k1=3, k2=50, solver="mfa2", seed=16,
            pca_energy=0.98, ridge=0.0, max_iter=100, tol=1e-10,
            mfa2_pca_energy=None):
    x = _data(x)
    graphs = build_graphs(x, labels, k1, k2)
    mean = x.mean(axis=0)
    centered = x - mean
    ndim = _integer(ndim, "ndim")
    if solver == "mfa2":
        # Scatter support is contained in span(X.T). An exact thin-SVD
        # reduction avoids allocating p-by-p matrices when p > n. Unlike
        # an energy truncation, this removes only numerical zero directions.
        q = None
        if x.shape[1] > len(x) or mfa2_pca_energy is not None:
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
            if not s[0] > 0:
                raise ValueError("Training samples have no variance")
            rank = int(np.sum(s > s[0] * 1e-12))
            if mfa2_pca_energy is not None:
                if not 0 < mfa2_pca_energy <= 1:
                    raise ValueError("Diagnostic PCA energy must be in (0,1]")
                rank = min(rank, int(np.searchsorted(
                    np.cumsum(s*s) / np.sum(s*s), mfa2_pca_energy) + 1))
            q = vt[:rank].T
        a, b = graphs.scatters(centered if q is None else centered @ q)
        solution = solve_trace_ratio(a, b, ndim, seed=seed, max_iter=max_iter, tol=tol)
        if q is not None:
            solution.projection = q @ solution.projection
        return MFAModel(mean, solution.projection, graphs, solver, solution)
    if solver != "mfa1" or not 0 < pca_energy <= 1:
        raise ValueError("Use mfa1/mfa2 and PCA energy in (0,1]")
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if not s[0] > 0:
        raise ValueError("Training samples have no variance")
    rank = int(np.searchsorted(np.cumsum(s*s) / np.sum(s*s), pca_energy) + 1)
    rank = min(rank, int(np.sum(s > s[0] * 1e-12)))
    if ndim > rank:
        raise ValueError("ndim exceeds PCA-retained dimension")
    q = vt[:rank].T
    a, b = graphs.scatters(centered @ q)
    w = q @ ged_projection(a, b, ndim, ridge)
    return MFAModel(mean, w, graphs, solver, pca_rank=rank)


def matrix_scatter(images, graph, fixed, update):
    """Eqs. (12)-(15); factor two agrees with ordered pair-distance sums."""
    if update == "right":
        z = np.einsum("ia,nij->naj", fixed, images)
    elif update == "left":
        z = images @ fixed
    else:
        raise ValueError("update must be left/right")
    lz = (laplacian(graph) @ z.reshape(len(z), -1)).reshape(z.shape)
    if update == "right":
        result = 2 * np.einsum("nki,nkj->ij", z, lz)
    else:
        result = 2 * np.einsum("nik,njk->ij", z, lz)
    return _sym(result)


@dataclass
class MMFAModel:
    mean: np.ndarray
    left: np.ndarray
    right: np.ndarray
    graphs: Graphs
    history: list

    def transform(self, images):
        images = np.asarray(images, dtype=float)
        if images.ndim != 3 or images.shape[1:] != self.mean.shape or not np.isfinite(images).all():
            raise ValueError("Expected finite images of the fitted shape")
        z = np.einsum("ia,nij,jb->nab", self.left, images - self.mean, self.right)
        return z.reshape(len(z), -1)


def fit_mmfa(images, labels, shape=(4, 4), k1=3, k2=50, iterations=10, ridge=0.0):
    images = np.asarray(images, dtype=float)
    if images.ndim != 3 or not np.isfinite(images).all():
        raise ValueError("Expected finite samples-by-height-by-width images")
    h, w = images.shape[1:]
    r, s = (_integer(v, "output shape") for v in shape)
    if r > h or s > w:
        raise ValueError("Output matrix shape exceeds input shape")
    graphs = build_graphs(images.reshape(len(images), -1), labels, k1, k2)
    mean = images.mean(axis=0)
    centered = images - mean
    left, right = np.eye(h), np.eye(w)
    model = MMFAModel(mean, left, right, graphs, [])
    for t in range(_integer(iterations, "iterations")):
        aw = matrix_scatter(centered, graphs.within, left, "right")
        bw = matrix_scatter(centered, graphs.between, left, "right")
        right = ged_projection(aw, bw, s, ridge)
        aw = matrix_scatter(centered, graphs.within, right, "left")
        bw = matrix_scatter(centered, graphs.between, right, "left")
        left = ged_projection(aw, bw, r, ridge)
        model.left, model.right = left, right
        z = model.transform(images)
        numerator = float(np.sum(z * (laplacian(graphs.within) @ z)))
        denominator = float(np.sum(z * (laplacian(graphs.between) @ z)))
        model.history.append({"iteration": t + 1, "within_between": numerator / denominator})
    return model
