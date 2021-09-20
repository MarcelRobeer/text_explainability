import pytest

from text_explainability.generation.target_encoding import FactFoilEncoder

y = ['a'] * 3 + ['b'] * 4 + ['c'] * 2 + ['a']

def test_initialize_factfoilencoder():
    ffe = FactFoilEncoder.from_str('a', ['a', 'b', 'c'])
    assert isinstance(ffe, FactFoilEncoder)

@pytest.mark.parametrize('labelset', [['a'], ['a', 'b'], ['a', 'b', 'c'], ['c', 'b', 'a']])
def test_labelset_factfoilencoder(labelset):
    ffe = FactFoilEncoder.from_str('a', labelset)
    assert ffe.labelset == labelset

def test_labelset_error_factfoilencoder():
    with pytest.raises(ValueError):
        ffe = FactFoilEncoder.from_str('d', ['a', 'b', 'c'])

@pytest.mark.parametrize('label', ['a', 'b', 'c'])
def test_apply_factfoilencoder(label):
    labelset = ['a', 'b', 'c']
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert ffe.encode(y).count(0) == y.count(label)

@pytest.mark.parametrize('label', ['a', 'b', 'c'])
def test_apply_factfoilencoder_inverse(label):
    labelset = ['a', 'b', 'c']
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert (len(y) - ffe.encode(y).count(1)) == y.count(label)

@pytest.mark.parametrize('label', ['a', 'b', 'c'])
def test_apply_factfoilencoder_binary(label):
    labelset = ['a', 'b', 'c']
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert all(i in [0, 1] for i in ffe.encode(y))

@pytest.mark.parametrize('label', ['a', 'b', 'c'])
def test_apply_factfoilencoder_string(label):
    labelset = ['a', 'b', 'c']
    y_ = [labelset.index(y__) for y__ in y]
    ffe = FactFoilEncoder.from_str(label, labelset)
    assert ffe.encode(y) == ffe.encode(y_)
