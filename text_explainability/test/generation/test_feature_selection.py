import numpy as np
import py
import pytest
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from text_explainability.generation.feature_selection import FeatureSelector
from text_explainability.generation.surrogate import LinearSurrogate

TRUE_METHODS_NO_MODEL = ['lasso_path', 'aic', 'bic', 'l1_reg']
TRUE_METHODS_MODEL = ['forward_selection', 'highest_weights']
TRUE_METHODS = TRUE_METHODS_NO_MODEL + TRUE_METHODS_MODEL
PARTIAL_METHODS = [method for method in TRUE_METHODS if method not in ['aic', 'bic', 'l1_reg']]
LOCAL_MODELS = [LinearSurrogate(model) for model in [LinearRegression(), Ridge(), Lasso()]]
FAKE_DATA = np.random.binomial(1, .5, size=(100, 20))



@pytest.mark.parametrize('method', TRUE_METHODS)
def test_featureselector_unknown_method(method):
    with pytest.raises(ValueError):
        FeatureSelector().select(X=np.array([]), y=np.array([]), method=method + '##')

@pytest.mark.parametrize('method', TRUE_METHODS_MODEL)
def test_featureselector_requires_model(method):
    with pytest.raises(ValueError):
        FeatureSelector(model=None).select(X=np.array([]), y=np.array([]), method=method)

@pytest.mark.parametrize('method', PARTIAL_METHODS)
@pytest.mark.parametrize('local_model', LOCAL_MODELS)
@pytest.mark.parametrize('n', [1, 3, 5, 10, 15, 20])
def test_featureselector_n_features(method, local_model, n):
    y = np.random.binomial(1, .5, size=100)
    assert len(FeatureSelector(model=local_model).select(X=FAKE_DATA, y=y, method=method, n_features=n)) == n, 'Wrong return length'

@pytest.mark.parametrize('method', ['aic', 'bic'])
def test_featureselector_n_features2(method):
    y = np.random.binomial(1, .5, size=100)
    assert len(FeatureSelector().select(X=FAKE_DATA, y=y, method=method)) <= 20, 'Return length too long'
