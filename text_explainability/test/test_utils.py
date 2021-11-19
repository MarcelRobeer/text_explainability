import numpy as np
import pytest

from text_explainability.utils import (binarize,
                                       default_detokenizer, default_tokenizer,
                                       character_tokenizer, character_detokenizer)

test_list = [
    'Dit is een voorbeeld tekst',
    'Ook deze tekst stelt een voorbeeld voor',
    'Mag deze tekst ook gebruikt worden als voorbeeld?',
    'Welke output geeft de augmenter op deze tekst?',
    'Deze tekst heeft maar één hoofdletter!',
    'Misschien is dit ook een voorbeeld?',
    'Dit vind ik sowieso een goed voorbeeld',
    'Mag ik deze tekst hardop roepen?',
    'Wij zijn er van overtuigd dat dit een goede test is',
    'Some text in English to try out!',
    'And some more amazing text...'
]

@pytest.mark.parametrize('tokenizer', [(default_tokenizer, default_detokenizer),
                                       (character_tokenizer, character_detokenizer)])
def test_tokenize_detokenize(tokenizer):
    tok, detok = tokenizer
    assert all(t == detok(tok(t)) for t in test_list), 'Tokenization + detokenization should be non-destructive'

@pytest.mark.parametrize('tokenizer', [default_tokenizer, character_tokenizer])
def test_empty_tokenize(tokenizer):
    assert tokenizer('') == [], 'Tokenizer made up tokens'

@pytest.mark.parametrize('detokenizer', [default_detokenizer, character_detokenizer])
def test_empty_detokenize(detokenizer):
    assert detokenizer([]) == '', 'Detokenizer made up tokens'

def test_single_word_tokenize():
    word = 'TESTWORD'
    assert default_tokenizer(word) == [word], 'Single word tokenize failed'

def test_single_word_detokenize():
    word = 'TESTWORD'
    assert default_detokenizer([word]) == word, 'Single word detokenize failed'

@pytest.mark.parametrize('size', [1, 10, 100, 1000])
def test_binarize(size):
    binarized = binarize(np.random.randint(-100, 100, size=(size,)))
    assert np.max(binarized) <= 1, 'Maximum too high'
    assert np.min(binarized) >= 0, 'Minimum too low'
