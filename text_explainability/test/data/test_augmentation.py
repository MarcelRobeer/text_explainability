import pytest
from instancelib.instances.text import MemoryTextInstance

from text_explainability.data import from_string
from text_explainability.data.augmentation import LeaveOut, TokenReplacement
from text_explainability.utils import default_detokenizer

SAMPLES = [from_string(string) for string in ['Dit is een voorbeeld.',
                                              'Nog een voorbeeld!',
                                              'Examples all the way',
                                              '?!...!1',
                                              'T3st t3st m0ar test']]
EMPTY = from_string('')


@pytest.mark.parametrize('sample', SAMPLES)
def test_equal_length_replacement(sample):
    repl = TokenReplacement(None, default_detokenizer)(sample)
    assert all(len(i.tokenized) == len(sample.tokenized) for i in repl.get_all()), 'Replacement yielded shorter instances'


@pytest.mark.parametrize('sample', SAMPLES)
def test_shorter_length_deletion(sample):
    repl = LeaveOut(None, default_detokenizer)(sample)
    assert all(len(i.tokenized) < len(sample.tokenized) for i in repl.get_all()), 'Removal did not yield shorter instances'


@pytest.mark.parametrize('sample', SAMPLES)
def test_replacement_applied_detokenized(sample):
    replacement = 'TEST_THIS_WORD'
    repl = TokenReplacement(None, default_detokenizer, replacement=replacement)(sample)
    assert all(replacement in i.tokenized for i in repl.get_all()), 'Replacement not found in resulting tokens'


@pytest.mark.parametrize('sample', SAMPLES)
def test_replacement_applied_detokenized(sample):
    replacement = 'TEST_THIS_WORD'
    repl = TokenReplacement(None, default_detokenizer, replacement=replacement)(sample)
    assert all(replacement in i.data for i in repl.get_all()), 'Replacement not found in resulting string'


@pytest.mark.parametrize('sample', SAMPLES)
def test_replacement_n_samples(sample):
    n_samples = 100
    repl = TokenReplacement(None, default_detokenizer)(sample, n_samples=n_samples)
    assert sum(1 for _ in list(repl)) <= n_samples, 'Replacement yielded too many samples'


@pytest.mark.parametrize('sample', SAMPLES)
def test_deletion_n_samples(sample):
    n_samples = 100
    repl = LeaveOut(None, default_detokenizer)(sample, n_samples=n_samples)
    assert sum(1 for _ in list(repl)) <= n_samples, 'Deletion yielded too many samples'


def test_EMPTY_instance_replacement():
    assert sum(1 for _ in list(TokenReplacement(None, default_detokenizer)(EMPTY))) <= 1, 'Empty input yielded too many samples (TokenReplacement)'


def test_EMPTY_instance_replacement():
    assert sum(1 for _ in list(LeaveOut(None, default_detokenizer)(EMPTY))) <= 1, 'Empty input yielded too many samples (LeaveOut)'
