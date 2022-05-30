from warnings import simplefilter

import pytest
from instancelib import AbstractEnvironment
from sklearn.exceptions import ConvergenceWarning

from text_explainability.generation.return_types import FeatureAttribution, Rules
from text_explainability.local_explanation import LIME, FoilTree, KernelSHAP, LocalRules, LocalTree, default_env
from text_explainability.test.__test import TEST_ENVIRONMENT, TEST_MODEL

simplefilter(action='ignore', category=ConvergenceWarning)


@pytest.mark.parametrize('label', ['punctuation', 'no_punctuation'])
def test_labels(label):
    assert label in LIME(TEST_ENVIRONMENT).__call__(sample='Explain this instance!', model=TEST_MODEL, labels=label).scores

@pytest.mark.parametrize('method', [LIME, KernelSHAP])
def test_feature_attribution(method):
    assert isinstance(method(TEST_ENVIRONMENT).__call__(sample='Test!!!', model=TEST_MODEL), FeatureAttribution), 'Wrong return type'

@pytest.mark.parametrize('method', [LocalTree])
def test_rules(method):
    assert isinstance(method(TEST_ENVIRONMENT).__call__(sample='Test!!!', model=TEST_MODEL), Rules), 'Wrong return type'

@pytest.mark.parametrize('method', [LocalRules])
@pytest.mark.parametrize('foil_fn', [0, 1, 'punctuation', 'no_punctuation'])
def test_rules_foil(method, foil_fn):
    assert isinstance(method(TEST_ENVIRONMENT).__call__(sample='Test!!!', model=TEST_MODEL, foil_fn=foil_fn), Rules), 'Wrong return type'

@pytest.mark.parametrize('method', [FoilTree])
@pytest.mark.parametrize('foil_fn', [0, 1, 'punctuation', 'no_punctuation'])
def test_trees_foil(method, foil_fn):
    assert isinstance(method(TEST_ENVIRONMENT).__call__(sample='Test!!!', model=TEST_MODEL, foil_fn=foil_fn), Rules), 'Wrong return type'

def test_default_env():
    assert isinstance(default_env(None), AbstractEnvironment), 'Wrong return type'
