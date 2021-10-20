import string

import pytest

from text_explainability.data import from_list
from text_explainability.data.embedding import CountVectorizer, TfidfVectorizer
from text_explainability.data.sampling import (KMedoids, LabelwiseKMedoids,
                                               LabelwiseMMDCritic, MMDCritic)

DATA = ['This is an example sentence...',
        'Another example',
        'More examples, more to do',
        'all lower',
        '!?!?1?!',
        'punctuation',
        'A good sentence is not a bad one',
        'More to generate..',
        '? why not start with punctuation']
LABELS = ['punctuation' if string.punctuation in instance else 'no_punctuation' for instance in DATA]

EMBEDDERS = [CountVectorizer, TfidfVectorizer]
SEEDS = [0, 1, 2, 3, 5, 6, 8, 10, 15, 20, 25, 30, 44, 99, 100]
N1 = [1, 2, 3, 4, 5]
N2 = [i + len(DATA) for i in [1, 2, 3, 4, 5, 99, 100]]


@pytest.mark.parametrize('embedder', EMBEDDERS)
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('n', N1)
def test_kmedoids_generate_n(embedder, seed, n):
    assert len(KMedoids(from_list(DATA, LABELS).dataset, embedder=embedder, seed=seed).prototypes(n=n)) == n


@pytest.mark.parametrize('n', N2)
def test_kmedoids_n_too_high(n):
    with pytest.raises(ValueError):
        KMedoids(from_list(DATA, LABELS).dataset).prototypes(n=n)


@pytest.mark.parametrize('embedder', EMBEDDERS)
@pytest.mark.parametrize('n', N1)
def test_mddcritic_generate_n_prototypes(embedder, n):
    assert len(MMDCritic(from_list(DATA, LABELS).dataset, embedder=embedder).prototypes(n=n)) == n


@pytest.mark.parametrize('embedder', EMBEDDERS)
@pytest.mark.parametrize('n', N1)
def test_mddcritic_generate_n_criticisms(embedder, n):
    mmdcritic = MMDCritic(from_list(DATA, LABELS).dataset, embedder=embedder)
    mmdcritic.prototypes(n=1)
    assert len(mmdcritic.criticisms(n=n)) == n


@pytest.mark.parametrize('embedder', EMBEDDERS)
@pytest.mark.parametrize('n', N1)
def test_mddcritic_generate_criticisms_no_prototypes(embedder, n):
    with pytest.raises(Exception):
        MMDCritic(from_list(DATA, LABELS).dataset, embedder=embedder).criticisms(n=n)


@pytest.mark.parametrize('n', N2)
def test_mmdcritic_n_prototypes_too_high(n):
    with pytest.raises(ValueError):
        MMDCritic(from_list(DATA, LABELS).dataset).prototypes(n=n)


@pytest.mark.parametrize('n', [len(DATA)] + N2)
def test_mmdcritic_n_criticisms_too_high(n):
    with pytest.raises(ValueError):
        mmdcritic = MMDCritic(from_list(DATA, LABELS).dataset)
        mmdcritic.prototypes(n=1)
        mmdcritic.criticisms(n=n)