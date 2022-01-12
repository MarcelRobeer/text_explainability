import numpy as np
import pytest

from text_explainability.data.weights import (exponential_kernel,
                                              pairwise_distances)

DIMS = [5, 10, 50, 100, 200]

@pytest.mark.parametrize('dim_a', DIMS)
@pytest.mark.parametrize('dim_b', DIMS)
def test_exponential_kernel(dim_a, dim_b):
    assert exponential_kernel(np.random.rand(dim_a, dim_b), 2).shape == (dim_a, dim_b), 'Shape should not have changed'

@pytest.mark.parametrize('dim_a', DIMS)
@pytest.mark.parametrize('dim_b', DIMS)
def test_pairwise_distances_dims(dim_a, dim_b):
    a = np.random.rand(dim_a, 20)
    b = np.random.rand(dim_b, 20)
    assert pairwise_distances(a, b).shape == (dim_a * dim_b,), 'Shape should depend on a and b'

@pytest.mark.parametrize('dim_a', DIMS)
def test_pairwise_distances_dims(dim_a):
    a = np.random.rand(dim_a, 20)
    assert pairwise_distances(a, np.random.rand(20)).shape == (dim_a,), 'Shape should depend on a when b.shape == (1,)'
