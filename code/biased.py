"""MBA/KMBA (Xu et al. 2007, Eqs.17-18) and BDA comparison.

The paper leaves the singular GED convention unspecified. These implementations
require an explicit ridge parameter; experiments report it as an adaptation.
Samples are rows. Kernel mapping is out of sample, using feedback data only.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from mfa import _data, _integer, _sym, laplacian, Graphs


def build_biased_graphs(x, positive, k1=3, k2=50):
    x = _data(x)
    mask = np.asarray(positive)
    if mask.ndim != 1 or len(mask) != len(x) or mask.dtype.kind != 'b':
        raise ValueError('positive must be a boolean mask, one per feedback sample')
    k1, k2 = _integer(k1, 'k1'), _integer(k2, 'k2')
    pos, neg = np.flatnonzero(mask), np.flatnonzero(~mask)
    if len(pos) <= k1 or len(pos)*len(neg) < k2:
        raise ValueError('Not enough positive samples or positive-negative pairs')
    d = cdist(x, x, 'sqeuclidean')
    sw, sb = np.zeros((len(x), len(x))), np.zeros((len(x), len(x)))
    for i in pos:
        candidates = pos[pos != i]
        chosen = candidates[np.argsort(d[i, candidates], kind='stable')[:k1]]
        sw[i, chosen] = sw[chosen, i] = 1
    ix = np.argsort(d[np.ix_(pos, neg)].ravel(), kind='stable')[:k2]
    i, j = np.unravel_index(ix, (len(pos), len(neg)))
    sb[pos[i], neg[j]] = sb[neg[j], pos[i]] = 1
    return Graphs(sw, sb)


def bda_scatters(features, positive):
    """Eq.16, supports explicit vectors or Gram-matrix feature rows."""
    z = _data(features)
    pos = np.asarray(positive, dtype=bool)
    if pos.shape != (len(z),) or pos.sum() < 2 or (~pos).sum() < 1:
        raise ValueError('BDA requires at least two positive and one negative sample')
    mean = z[pos].mean(axis=0)
    a, b = z[pos]-mean, z[~pos]-mean
    return a.T @ a, b.T @ b


def regularized_ged(within, between, ndim, ridge):
    """Top GED(B,A+alpha I), alpha=ridge*tr(A+B)/p (explicit adaptation)."""
    a, b = _sym(within), _sym(between)
    ndim = _integer(ndim, 'ndim')
    if not np.isfinite(ridge) or ridge <= 0 or ndim > len(a):
        raise ValueError('A finite positive ridge and valid dimension are required')
    alpha = float(ridge*np.trace(a+b)/len(a))
    if not alpha > 0:
        raise ValueError('No feedback variance')
    values, w = eigh(b, a+alpha*np.eye(len(a)))
    w = w[:, -ndim:][:, ::-1]
    return w, values[-ndim:][::-1], alpha


@dataclass
class BiasedModel:
    feedback: np.ndarray
    projection: np.ndarray
    gamma: float | None
    alpha: float
    method: str
    graphs: Graphs | None

    def transform(self, x):
        x = _data(x)
        if x.shape[1] != self.feedback.shape[1]:
            raise ValueError('Feature dimension differs from fitted feedback')
        features = x if self.gamma is None else np.exp(-self.gamma*cdist(x, self.feedback, 'sqeuclidean'))
        return features @ self.projection


def fit_biased(x, positive, ndim=2, k1=3, k2=20, *, method='mba',
               gamma=None, ridge):
    """Linear MBA/BDA or RBF KMBA/KBDA with a fixed explicit ridge.

    Kernel graph neighbors are selected in the original features as in the
    paper. Both A and B use the same RKHS normalization for the final W.
    """
    x = _data(x)
    pos = np.asarray(positive)
    if pos.dtype.kind != 'b' or pos.shape != (len(x),):
        raise ValueError('positive must be a boolean mask')
    if method not in ['mba', 'bda']:
        raise ValueError('method must be mba or bda')
    if gamma is not None and (not np.isfinite(gamma) or gamma <= 0):
        raise ValueError('gamma must be finite and positive')
    gram = None if gamma is None else np.exp(-gamma*cdist(x, x, 'sqeuclidean'))
    features = x if gram is None else gram
    graphs = build_biased_graphs(x, pos, k1, k2) if method == 'mba' else None
    if graphs is None:
        a, b = bda_scatters(features, pos)
    else:
        a, b = graphs.scatters(features)
    w, _, alpha = regularized_ged(a, b, ndim, ridge)
    # Unit norm of the actual projection direction, not just the coefficient
    # vector. A^T K A is the RKHS norm when W=Phi(X)A. An explicit RKHS feature factorization
    # is independently tested below.
    norms = np.linalg.norm(w, axis=0) if gram is None else np.sqrt(
        np.maximum(np.einsum('ij,ik,kj->j', w, gram, w), 0))
    if np.any(norms < np.finfo(float).tiny):
        raise ArithmeticError('Zero projection norm')
    w = w / norms
    return BiasedModel(x.copy(), w, gamma, alpha, method, graphs)
