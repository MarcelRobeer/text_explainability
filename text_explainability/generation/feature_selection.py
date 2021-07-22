"""Feature selection methods for limiting explanation length."""

import numpy as np

from typing import Optional
from sklearn.linear_model import LassoLarsIC, Lasso, lars_path

from text_explainability.generation.surrogate import LinearSurrogate
from text_explainability.default import Readable


class FeatureSelector(Readable):
    def __init__(self, model: Optional[LinearSurrogate] = None):
        """[summary]

        Args:
            model (Optional[LinearSurrogate], optional): Linear surrogate used to calculate 
                feature importance scores. Defaults to None.
        """
        super().__init__()
        self.model = model
        if self.model is not None:
            self.model.alpha_zero()
            self.model.fit_intercept = True

    def _forward_selection(self, X: np.ndarray,
                           y: np.ndarray,
                           weights: np.ndarray = None,
                           n_features: int = 10) -> np.ndarray:
        """Feature selection with forward selection, as used by `LIME`_.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for y.
            weights (np.ndarray, optional): Relative weights of X. Defaults to None.
            n_features (int, optional): [description]. Defaults to 10.

        Raises:
            AssertionError: The local linear model used to calculate forward_selection
                was not defined.

        Returns:
            np.ndarray: Indices of selected features.

        .. _LIME:
            https://github.com/marcotcr/lime/blob/master/lime/lime_base.py
        """
        assert self.model is not None, 'forward_selection requires a local linear model'
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
                         weights: np.ndarray = None, n_features: int = 10) -> np.ndarray:
        """Feature selection according to highest feature importance, as used by `LIME`_.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for X.
            weights (np.ndarray, optional): Relative weights of X. Defaults to None.
            n_features (int, optional): Number of features to select. Defaults to 10.

        Raises:
            AssertionError: The local linear model used to calculate forward_selection
                was not defined.

        Returns:
            np.ndarray: Indices of selected features.

        .. _LIME:
            https://github.com/marcotcr/lime/blob/master/lime/lime_base.py
        """
        assert self.model is not None, 'highest_weights requires a local linear model'
        self.model.fit(X, y, weights=weights)
        weighted_data = self.model.feature_importances * X[0]
        feature_weights = sorted(
            zip(range(X.shape[1]), weighted_data),
            key=lambda x: np.abs(x[1]),
            reverse=True)
        return np.array([x[0] for x in feature_weights[:n_features]])

    def _lasso_path(self, X: np.ndarray, y: np.ndarray,
                    weights: np.ndarray = None, n_features: int = 10) -> np.ndarray:
        """Feature selection with `LASSO`_, as used by `LIME`_.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for X.
            weights (np.ndarray, optional): Relative weights of X. Defaults to None.
            n_features (int, optional): Number of features to select. Defaults to 10.

        Returns:
            np.ndarray: Indices of selected features.

        .. _LASSO:
            https://en.wikipedia.org/wiki/Lasso_(statistics)
        .. _LIME:
            https://github.com/marcotcr/lime/blob/master/lime/lime_base.py
        """
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

    def _information_criterion(self, X: np.ndarray, y: np.ndarray, criterion='aic') -> np.ndarray:
        """AIC/BIC for feature selection, as used by `SHAP`_.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for X.
            criterion (str, optional): Whether to use `Akaike Information Criterion`_ (`aic`) or 
                `Bayesian Information Criterion`_ (`bic`). Defaults to 'aic'.

        Returns:
            np.ndarray: Indices of selected features.

        .. _SHAP:
            https://github.com/slundberg/shap
        .. _Akaike Information Criterion:
            https://en.wikipedia.org/wiki/Akaike_information_criterion
        .. _Bayesian Information Criterion:
            https://en.wikipedia.org/wiki/Bayesian_information_criterion
        """
        assert criterion in ['aic', 'bic'], f'Unknown criterion "{criterion}"'
        return np.nonzero(LassoLarsIC(criterion=criterion).fit(X, y).coef_)[0]

    def _l1_reg(self, X: np.ndarray, y: np.ndarray,
                n_features: int = 10, alpha: Optional[float] = None) -> np.ndarray:
        """L1-regularization for feature selection, as used by `SHAP`_.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for X.
            n_features (int, optional): Number of features to select. Defaults to 10.
            alpha (Optional[float], optional): Hyperparameter for L1 regularization. Defaults to None.

        Returns:
            np.ndarray: Indices of selected features.

        .. _SHAP:
            https://github.com/slundberg/shap
        """
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
                 alpha: Optional[float] = None) -> np.ndarray:
        """Apply feature selection for dataset X and targets y.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for X.
            weights (np.ndarray, optional): Relative weights of X. Defaults to None.
            n_features (int, optional): Number of features to select. Defaults to 10.
            method (str, optional): Method to apply for feature selection, choose from `None`,
                `forward_selection`, `highest_weights`, `lasso_path`, `aic`, `bic`, `l1_reg`. 
                Defaults to None.
            alpha (Optional[float], optional): Hyperparameter for L1 regularization. Defaults to None.

        Raises:
            AssertionError: Unknown method, or the requirements of a method have not been satisfied.

        Returns:
            np.ndarray: Indices of selected features.
        """
        if self.model is None:
            assert method not in ['forward_selection', 'highest_weights'], \
                f'{self.__class__.__name__} requires a `model` to use methods forward_selection and ' \
                'highest_weights'
        assert method in [None, 'forward_selection', 'highest_weights', 'lasso_path',
                          'aic', 'bic', 'l1_reg'], \
            f'Unknown method "{method}"'
        n_features = min(X.shape[1], n_features)

        # Do not perform feature selection, but return all
        if n_features == X.shape[1] and method not in ['aic', 'bic', 'l1_reg'] or method is None:
            return np.arange(X.shape[1])

        # Perform feature selection
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
