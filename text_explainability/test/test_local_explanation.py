import pytest

from text_explainability.generation.return_types import (FeatureAttribution,
                                                         Rules)
from text_explainability.local_explanation import (LIME, KernelSHAP,
                                                   LocalRules, LocalTree)
from text_explainability.test.__test import TEST_ENVIRONMENT, TEST_MODEL


@pytest.mark.parametrize('label', ['punctuation', 'no_punctuation'])
def test_labels(label):
    assert label in LIME(TEST_ENVIRONMENT).__call__(sample='Explain this instance!', model=TEST_MODEL, labels=label).scores

@pytest.mark.parametrize('method', [LIME, KernelSHAP])
def test_feature_attribution(method):
    assert isinstance(method(TEST_ENVIRONMENT).__call__(sample='Test!!!', model=TEST_MODEL), FeatureAttribution), 'Wrong return type'

@pytest.mark.parametrize('method', [LocalTree])
def test_rules(method):
    assert isinstance(method(TEST_ENVIRONMENT).__call__(sample='Test!!!', model=TEST_MODEL), Rules), 'Wrong return type'

# @pytest.mark.parametrize('method', [LocalRules])
# def test_rules_foil(method):
#     assert isinstance(method(TEST_ENVIRONMENT).__call__(sample='Test!!!', model=TEST_MODEL, foil_fn=0), Rules), 'Wrong return type'

