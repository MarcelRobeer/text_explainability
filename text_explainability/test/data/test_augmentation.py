from instancelib.instances.text import MemoryTextInstance

from text_explainability.data.augmentation import TokenReplacement, LeaveOut
from text_explainability.utils import default_tokenizer, default_detokenizer

def craft_instance(data: str):
    return MemoryTextInstance(0, data, None, tokenized = default_tokenizer(data))


sample = craft_instance('Dit is een voorbeeld.')
empty = craft_instance('')

def test_equal_length_replacement():
    repl = TokenReplacement(None, default_detokenizer)(sample)
    assert all(len(i.tokenized) == len(sample.tokenized) for i in repl.get_all()), 'Replacement yielded shorter instances'

def test_shorter_length_deletion():
    repl = LeaveOut(None, default_detokenizer)(sample)
    assert all(len(i.tokenized) < len(sample.tokenized) for i in repl.get_all()), 'Removal did not yield shorter instances'

def test_replacement_applied_detokenized():
    replacement = 'TEST_THIS_WORD'
    repl = TokenReplacement(None, default_detokenizer, replacement=replacement)(sample)
    assert all(replacement in i.tokenized for i in repl.get_all()), 'Replacement not found in resulting tokens'

def test_replacement_applied_detokenized():
    replacement = 'TEST_THIS_WORD'
    repl = TokenReplacement(None, default_detokenizer, replacement=replacement)(sample)
    assert all(replacement in i.data for i in repl.get_all()), 'Replacement not found in resulting string'

def test_replacement_n_samples():
    n_samples = 100
    repl = TokenReplacement(None, default_detokenizer)(sample, n_samples=n_samples)
    assert sum(1 for _ in list(repl)) <= n_samples, 'Replacement yielded too many samples'

def test_deletion_n_samples():
    n_samples = 100
    repl = LeaveOut(None, default_detokenizer)(sample, n_samples=n_samples)
    assert sum(1 for _ in list(repl)) <= n_samples, 'Deletion yielded too many samples'

def test_empty_instance_replacement():
    assert sum(1 for _ in list(TokenReplacement(None, default_detokenizer)(empty))) <= 1, 'Empty input yielded too many samples (TokenReplacement)'

def test_empty_instance_replacement():
    assert sum(1 for _ in list(LeaveOut(None, default_detokenizer)(empty))) <= 1, 'Empty input yielded too many samples (LeaveOut)'

