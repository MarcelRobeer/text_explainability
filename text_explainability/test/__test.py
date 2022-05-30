from genbase_test_helpers import TEST_MODEL
from instancelib import TextEnvironment

TEST_INSTANCES = [
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

TEST_LABELS = [['punctuation'] if any(c in item for c in '!@#$%&*().,?') else ['no_punctuation'] for item in TEST_INSTANCES]

TEST_ENVIRONMENT = TextEnvironment.from_data(target_labels={'punctuation', 'no_punctuation'},
                                             indices=list(range(len(TEST_INSTANCES))),
                                             data=TEST_INSTANCES,
                                             ground_truth=TEST_LABELS,
                                             vectors=None)

__all__ = ['TEST_INSTANCES', 'TEST_LABELS', 'TEST_ENVIRONMENT', 'TEST_MODEL']
