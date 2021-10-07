import pytest

from text_explainability.generation.target_encoding import FactFoilEncoder

LABELS = ['a'] * 3 + ['b'] * 4 + ['c'] * 2 + ['a']
LABELSET = ['a', 'b', 'c']


def test_initialize_factfoilencoder():
    ffe = FactFoilEncoder.from_str('a', LABELSET)
    assert isinstance(ffe, FactFoilEncoder)

@pytest.mark.parametrize('labelset', [['a'], ['a', 'b'], LABELSET, ['c', 'b', 'a']])
def test_labelset_factfoilencoder(labelset):
    ffe = FactFoilEncoder.from_str('a', labelset)
    assert ffe.labelset == labelset

def test_labelset_error_factfoilencoder():
    with pytest.raises(ValueError):
        ffe = FactFoilEncoder.from_str('d', LABELSET)

@pytest.mark.parametrize('label', LABELSET)
def test_apply_factfoilencoder(label):
    labelset = LABELSET
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert ffe.encode(LABELS).count(0) == LABELS.count(label)

@pytest.mark.parametrize('label', LABELSET)
def test_apply_factfoilencoder_inverse(label):
    labelset = LABELSET
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert (len(LABELS) - ffe.encode(LABELS).count(1)) == LABELS.count(label)

@pytest.mark.parametrize('label', LABELSET)
def test_apply_factfoilencoder_binary(label):
    labelset = LABELSET
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert all(i in [0, 1] for i in ffe.encode(LABELS))

@pytest.mark.parametrize('label', LABELSET)
def test_apply_factfoilencoder_string(label):
    labelset = LABELSET
    y_ = [labelset.index(y__) for y__ in LABELS]
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert ffe.encode(LABELS) == ffe.encode(y_)
