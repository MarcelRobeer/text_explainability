import numpy as np
import pytest

from text_explainability.data.embedding import (CountVectorizer,
                                                SentenceTransformer,
                                                TfidfVectorizer, as_2d, as_3d)

DIMS = [3, 5, 10, 100, 300]
DIM_RED_METHODS = ['pca', 'kernel_pca', 'incremental_pca', 'nmf', 'tsne']
EXAMPLE_STRINGS = ['A sentence to embed.', 'And a second one', 'A third that is AMAZING!']
VECTORIZERS = [SentenceTransformer, CountVectorizer, TfidfVectorizer]


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('method', DIM_RED_METHODS)
def test_as_2d(dim, method):
    assert as_2d(np.random.rand(100, dim), method=method).shape[-1] == 2

@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('method', DIM_RED_METHODS)
def test_as_3d(dim, method):
    assert as_3d(np.random.rand(100, dim), method=method).shape[-1] == 3

@pytest.mark.parametrize('as_nd', [as_2d, as_3d])
@pytest.mark.parametrize('method', DIM_RED_METHODS)
def test_too_few_dims(as_nd, method):
    if method not in ['nmf', 'kernel_pca']:
        with pytest.raises(ValueError):
            as_nd(np.random.rand(100, 1), method=method)

@pytest.mark.parametrize('as_nd', [as_2d, as_3d])
def test_unknown_method(as_nd):
    with pytest.raises(ValueError):
        as_nd(np.random.rand(100, 100), method='sdlskadasnm')

@pytest.mark.parametrize('vectorizer', VECTORIZERS)
def test_vectorizer(vectorizer):
    v = vectorizer()
    embedded = v.embed(EXAMPLE_STRINGS)
    assert isinstance(embedded, np.ndarray), f'Unexpected type for embedded (type={type(embedded)}'
    assert embedded.shape[0] == len(EXAMPLE_STRINGS), f'Not all strings were embedded, expected {len(EXAMPLE_STRINGS)} got {embedded.shape[0]}'
