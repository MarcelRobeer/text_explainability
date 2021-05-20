from instancelib import TextInstance

from explainability.data.augmentation import TokenReplacement, LeaveOut
from explainability.utils import default_tokenizer, default_detokenizer

def craft_instance(data: str):
    s = TextInstance(0, data, None)
    s.tokenized = default_tokenizer(s.data)
    return s


sample = craft_instance('Dit is een voorbeeld.')
empty = craft_instance('')


def test_equal_length_replacement():
    repl = TokenReplacement(default_detokenizer)(sample)
    assert all(len(i.tokenized) == len(sample.tokenized) for i in repl), 'Replacement yielded shorter instances'

def test_shorter_length_deletion():
    repl = LeaveOut(default_detokenizer)(sample)
    assert all(len(i.tokenized) < len(sample.tokenized) for i in repl), 'Removal did not yield shorter instances'

def test_replacement_applied_detokenized():
    replacement = 'TEST_THIS_WORD'
    repl = TokenReplacement(default_detokenizer, replacement=replacement)
    assert all(replacement in i.tokenized for i in repl), 'Replacement not found in resulting tokens'

def test_replacement_applied_detokenized():
    replacement = 'TEST_THIS_WORD'
    repl = TokenReplacement(default_detokenizer, replacement=replacement)
    assert all(replacement in i.data for i in repl), 'Replacement not found in resulting string'

def test_replacement_n_samples():
    n_samples = 100
    repl = TokenReplacement(default_detokenizer)(sample, n_samples=n_samples)
    assert sum(1 for _ in repl) <= n_samples, 'Replacement yielded too many samples'

def test_deletion_n_samples():
    n_samples = 100
    repl = LeaveOut(default_detokenizer)(sample, n_samples=n_samples)
    assert sum(1 for _ in repl) <= n_samples, 'Deletion yielded too many samples'

def test_empty_instance_replacement():
    assert sum(1 for _ in TokenReplacement(default_tokenizer)(empty)) == 0, 'Empty input yielded too many samples (TokenReplacement)'

def test_empty_instance_replacement():
    assert sum(1 for _ in LeaveOut(default_tokenizer)(empty)) == 0, 'Empty input yielded too many samples (LeaveOut)'
