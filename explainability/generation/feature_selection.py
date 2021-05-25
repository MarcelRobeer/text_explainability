"""TO-DO
- Update documentation
"""

import numpy as np

from typing import Optional
from sklearn.linear_model import LassoLarsIC, Lasso, lars_path

from explainability.generation.surrogate import LinearSurrogate
from explainability.default import Readable


class FeatureSelector(Readable):
    def __init__(self, model: Optional[LinearSurrogate] = None):
        super().__init__()
        self.model = model
        if self.model is not None:
            self.model.alpha_zero()
            self.model.fit_intercept = True

    def _forward_selection(self, X: np.ndarray,
                           y: np.ndarray,
                           weights: np.ndarray = None,
                           n_features: int = 10):
        """LIME"""
        n_features = min(X.shape[1], n_features)
        used_features = []
        for _ in range(n_features):
            max_ = -100000000
            best = 0
            for feature in range(X.shape[1]):
                if feature in used_features:
                    continue
                self.model.fit(X[:, used_features + [feature]], y,
                               weights=weights)
                score = self.model.score(X[:, used_features + [feature]], y, weights=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)
            
    def _highest_weights(self, X: np.ndarray, y: np.ndarray,
                         weights: np.ndarray = None, n_features: int = 10):
        """LIME"""
        self.model.fit(X, y, weights=weights)
        weighted_data = self.model.feature_importances * X[0]
        feature_weights = sorted(
            zip(range(X.shape[1]), weighted_data),
            key=lambda x: np.abs(x[1]),
            reverse=True)
        return np.array([x[0] for x in feature_weights[:n_features]])

    def _lasso_path(self, X: np.ndarray, y: np.ndarray,
                    weights: np.ndarray = None, n_features: int = 10):
        """LIME"""
        if weights is None:
            weights = np.ones(X.shape[0])
        weighted_data = ((X - np.average(X, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
        weighted_labels = ((y - np.average(y, weights=weights))
                            * np.sqrt(weights))
        nonzero = range(weighted_data.shape[1])
        _, _, coefs = lars_path(weighted_data, weighted_labels, method='lasso', verbose=False)
        for i in range(len(coefs.T) - 1, 0, -1):
            nonzero = coefs.T[i].nonzero()[0]
            if len(nonzero) <= n_features:
                break
        used_features = nonzero
        return np.array(used_features)

    def _information_criterion(self, X: np.ndarray, y: np.ndarray, criterion='aic'):
        """SHAP"""
        assert criterion in ['aic', 'bic'], f'Unknown criterion "{criterion}"'
        return np.nonzero(LassoLarsIC(criterion=criterion).fit(X, y).coef_)[0]

    def _l1_reg(self, X: np.ndarray, y: np.ndarray,
                n_features: int = 10, alpha: Optional[float] = None):
        """SHAP"""
        if alpha is not None:
            return np.nonzero(Lasso(alpha=alpha).fit(X, y).coef_)[0]
        # use n_features
        if y.ndim > 1:
            # To-do: multiclass support?
            y = y[:, 0]
        return lars_path(X, y, max_iter=n_features)[1]

    def __call__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 weights: np.ndarray = None,
                 n_features: int = 10,
                 method: str = None,
                 alpha: Optional[float] = None):
        if self.model is None:
            assert method not in ['forward_selection', 'highest_weights', 'lasso_path'], \
                f'{self.__class__.__name__} requires a `model` to use methods forward_selection, ' \
                'highest_weights and lasso_path'
        assert method in [None, 'forward_selection', 'highest_weights', 'lasso_path',
                          'aic', 'bic', 'l1_reg'], \
            f'Unknown method "{method}"'
        n_features = min(X.shape[1], n_features)

        if n_features == X.shape[1] and method not in ['aic', 'bic', 'l1_reg'] or method is None:
            return np.arange(X.shape[1])

        if method == 'forward_selection':
            return self._forward_selection(X, y, weights=weights, n_features=n_features)
        elif method == 'highest_weights':
            return self._highest_weights(X, y, weights=weights, n_features=n_features)
        elif method == 'lasso_path':
            return self._lasso_path(X, y, weights=weights, n_features=n_features)
        elif method in ['aic', 'bic']:
            return self._information_criterion(X, y, criterion=method)
        elif method == 'l1_reg':
            return self._l1_reg(X, y, n_features=n_features, alpha=alpha)
