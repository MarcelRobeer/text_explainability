"""Wrappers for surrogate models, used for local/global explanations.

Todo:
    * Add documentation
    * Differentiate between classifiers and regressors
    * Extract rules from decision tree (https://mljar.com/blog/extract-rules-decision-tree/)
"""

import numpy as np
from sklearn.base import clone
from typing import Optional, Sequence

from text_explainability.default import Readable


class BaseSurrogate(Readable):
    def __init__(self, model):
        super().__init__()
        self._model = clone(model)

    def fit(self, X, y, weights=None):
        self._model.fit(X, y, sample_weight=weights)
        return self

    def predict(self, X):
        return self._model.predict(X)

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

    @property
    def classes(self):
        return self._model.classes_

    def decision_path(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim < 2:
            X = X.reshape(1, -1)
        return self._model.decision_path(X).toarray()

    def max_rule_size(self, size: Optional[int]):
        self._model.set_params(max_depth=size)

    def features(self, tokens_to_map: Optional[Sequence[str]] = None):
        def map_token(token):
            if tokens_to_map is None:
                return token
            return tokens_to_map[token]
        return [None if f < 0 else map_token[f] for f in self._model.tree_.feature]

    def leaf_classes(self):
        # TODO: check if truly classification
        return [self._model.classes_[np.argmax(self._model.tree_.value[i])] if f < 0 else None
                for i, f in enumerate(self._model.tree_.feature)]


class RuleSurrogate(BaseSurrogate):
    """Wrapper around `SkopeRules`_ model for usage in local/global surrogate models.

    _SkopeRules:
        https://github.com/scikit-learn-contrib/skope-rules
    """

    @property
    def rules(self):
        return self._model.rules_

    def score_top_rules(self, X):
        return self._model.score_top_rules(X)
