"""Wrappers for surrogate models, used for local/global explanations."""

from sklearn.base import clone

from text_explainability.default import Readable


class BaseSurrogate(Readable):
    def __init__(self, model):
        super().__init__()
        self._model = clone(model)

    def fit(self, X, y, weights=None):
        self._model.fit(X, y, sample_weight=weights)
        return self

    @property
    def feature_importances(self):
        raise NotImplementedError


class LinearSurrogate(BaseSurrogate):
    def __init__(self, model):
        """Wrapper around sklearn linear model for usage in local/global surrogate models."""
        super().__init__(model)
        self.__alpha_original = self._model.alpha

    @property
    def coef(self):
        return self._model.coef_

    @property
    def feature_importances(self):
        return self.coef

    @property
    def intercept(self):
        return self._model.intercept_

    def score(self, X, y, weights=None):
        return self._model.score(X, y, sample_weight=weights)

    def alpha_zero(self):
        self._model.alpha = 0

    def alpha_reset(self):
        self._model.alpha = self.__alpha_original

    @property
    def fit_intercept(self):
        return self._model.fit_intercept

    @fit_intercept.setter
    def fit_intercept(self, fit_intercept):
        self._model.fit_intercept = fit_intercept


class TreeSurrogate(BaseSurrogate):
    """Wrapper around sklearn tree model for usage in local/global surrogate models."""

    @property
    def feature_importances(self):
        return self._model.feature_importances_
