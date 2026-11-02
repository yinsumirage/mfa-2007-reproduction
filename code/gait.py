"""Paper Eqs.8-9: cycle averages and directed median-of-nearest matching.

Gait periods must be supplied. No estimated period is presented as ground truth.
These components do not themselves reproduce the USF benchmark.
"""
import numpy as np
from scipy.spatial.distance import cdist
from mfa import _data, _integer


def cycle_average(frames, period):
    a = np.asarray(frames, dtype=float)
    period = _integer(period, 'period')
    if a.ndim != 3 or not np.isfinite(a).all() or np.any((a != 0) & (a != 1)):
        raise ValueError('Expected binary frames of shape time,height,width')
    cycles = len(a) // period
    if cycles < 1:
        raise ValueError('Sequence is shorter than a complete cycle')
    return a[:cycles*period].reshape(cycles, period, *a.shape[1:]).mean(axis=1)


def sequence_distance(probe, gallery):
    a, b = _data(probe), _data(gallery)
    if a.shape[1] != b.shape[1]:
        raise ValueError('Sequence feature dimensions differ')
    # The paper's score is directed: the median runs over probe images.
    return float(np.median(cdist(a, b, 'sqeuclidean').min(axis=1)))


def identify(probe_sequences, gallery_sequences, gallery_labels, rank=1):
    rank = _integer(rank, 'rank')
    if len(gallery_sequences) != len(gallery_labels) or rank > len(gallery_sequences):
        raise ValueError('Invalid gallery/rank')
    labels = np.asarray(gallery_labels)
    distances = np.array([[sequence_distance(p,g) for g in gallery_sequences]
                          for p in probe_sequences])
    return labels[np.argsort(distances, axis=1, kind='stable')[:, :rank]], distances
