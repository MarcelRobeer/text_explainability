"""Functions for computing weights for training models (e.g. based on distance to original sample)."""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as pwd
from sklearn.metrics.pairwise import rbf_kernel as rbf


def pairwise_distances(a, b, metric: str = 'cosine', multiply=100) -> np.ndarray:
    """Pairwise distancens between two vectors.

    Args:
        a: Vector A.
        b: Vector B.
        metric (str, optional): Metric name (e.g. 'cosine', 'euclidean'). Defaults to 'cosine'.
        multiply (int, optional): Multiply the final distance value by this constant. Defaults to 100.

    Returns:
        np.ndarray: Pairwise distances.
    """
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return pwd(a, b, metric=metric).ravel() * multiply


def exponential_kernel(d, kw):
    """Exponential kernel."""
    return np.sqrt(np.exp(-(d ** 2) / kw ** 2))


def rbf_kernel(X, gamma=None):
    """Radial basis function (RBF) kernel."""
    return rbf(X, gamma=gamma)
