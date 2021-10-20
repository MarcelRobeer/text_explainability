from text_explainability.utils import default_detokenizer, default_tokenizer

test_list = [
    'Dit is een voorbeeld tekst',
    'Ook deze tekst stelt een voorbeeld voor',
    'Mag deze tekst ook gebruikt worden als voorbeeld?',
    'Welke output geeft de augmenter op deze tekst?',
    'Deze tekst heeft maar één hoofdletter!',
    'Misschien is dit ook een voorbeeld?',
    'Dit vind ik sowieso een goed voorbeeld',
    'Mag ik deze tekst hardop roepen?',
    'Wij zijn er van overtuigd dat dit een goede test is'
]

def test_tokenize_detokenize():
    assert all(t == default_detokenizer(default_tokenizer(t)) for t in test_list), 'Tokenization + detokenization should be non-destructive'

def test_empty_tokenize():
    assert default_tokenizer('') == [], 'Tokenizer made up tokens'

def test_empty_detokenize():
    assert default_detokenizer([]) == '', 'Detokenizer made up tokens'

def test_single_word_tokenize():
    word = 'TESTWORD'
    assert default_tokenizer(word) == [word], 'Single word tokenize failed'

def test_single_word_detokenize():
    word = 'TESTWORD'
    assert default_detokenizer([word]) == word, 'Single word detokenize failed'
