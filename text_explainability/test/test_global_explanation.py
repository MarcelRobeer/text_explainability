import pytest

from text_explainability.generation.return_types import FeatureList
from text_explainability.global_explanation import (TokenFrequency,
                                                    TokenInformation)
from text_explainability.test.__test import TEST_ENVIRONMENT, TEST_MODEL

METHODS = [TokenFrequency, TokenInformation]


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('labelprovider', [None, TEST_ENVIRONMENT.labels])
def test_requires_model(method, labelprovider):
    with pytest.raises(ValueError):
        method(TEST_ENVIRONMENT.dataset).__call__(model=None, labelprovider=labelprovider, explain_model=True), 'No model but still forms explanation'

@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('model', [None, TEST_MODEL])
def test_requires_labelprovider(method, model):
    with pytest.raises(ValueError):
        method(TEST_ENVIRONMENT.dataset).__call__(model=model, labelprovider=None, explain_model=False), 'No labelprovider but still forms explanation'

@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('explain_model', [True, False])
def test_return_type(method, explain_model):
    assert isinstance(method(TEST_ENVIRONMENT.dataset).__call__(model=TEST_MODEL,
                                                                labelprovider=TEST_ENVIRONMENT.labels,
                                                                explain_model=explain_model),
                      FeatureList), 'Wrong return type'