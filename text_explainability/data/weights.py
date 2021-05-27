import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as pwd


def pairwise_distances(a, b, metric='cosine', multiply=100):
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return pwd(a, b, metric=metric).ravel() * multiply


def exponential_kernel(d, kw):
    return np.sqrt(np.exp(-(d ** 2) / kw ** 2))
