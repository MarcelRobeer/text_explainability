import pytest

from text_explainability.global_explanation import (TokenFrequency,
                                                    TokenInformation)
from text_explainability.local_explanation import LIME, KernelSHAP
from text_explainability.test.__test import TEST_ENVIRONMENT, TEST_MODEL
from text_explainability.ui.notebook import Render

G_CONFIGS = [method(TEST_ENVIRONMENT.dataset).__call__(labelprovider=TEST_ENVIRONMENT.labels, explain_model=False) for method in [TokenFrequency, TokenInformation]]
L_CONFIGS = [method(TEST_ENVIRONMENT).__call__(sample='Test!!', n_samples=10, model=TEST_MODEL) for method in [KernelSHAP, LIME]]

def test_unknown_config():
    assert isinstance(Render({'META': {'type': 'unknown'}, 'CONTENT': {}}).as_html(), str)

@pytest.mark.parametrize('config', G_CONFIGS)
def test_global_config(config):
    assert isinstance(Render(config.to_config()).as_html(), str)

@pytest.mark.parametrize('config', L_CONFIGS)
def test_local_config(config):
    assert isinstance(Render(config.to_config()).as_html(), str)
