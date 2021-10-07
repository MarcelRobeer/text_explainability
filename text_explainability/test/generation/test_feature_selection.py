import pytest
import numpy as np

from text_explainability.generation.feature_selection import FeatureSelector
from text_explainability.generation.surrogate import LinearSurrogate
from sklearn.linear_model import LinearRegression, Ridge, Lasso


TRUE_METHODS_NO_MODEL = ['lasso_path', 'aic', 'bic', 'l1_reg']
TRUE_METHODS_MODEL = ['forward_selection', 'highest_weights']
TRUE_METHODS = TRUE_METHODS_NO_MODEL + TRUE_METHODS_MODEL
LOCAL_MODELS = [LinearSurrogate(model) for model in [LinearRegression(), Ridge(), Lasso()]]


@pytest.mark.parametrize('method', TRUE_METHODS)
def test_featureselector_unknown_method(method):
    with pytest.raises(ValueError):
        FeatureSelector().select(X=np.array([]), y=np.array([]), method=method + '##')


@pytest.mark.parametrize('method', TRUE_METHODS_MODEL)
def test_featureselector_requires_model(method):
    with pytest.raises(ValueError):
        FeatureSelector(model=None).select(X=np.array([]), y=np.array([]), method=method)
