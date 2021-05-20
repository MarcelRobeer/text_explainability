"""TO-DO:
- Implement Anchors
- Implement SHAP
- Implement Foil Trees
"""

import numpy as np

from instancelib import TextBucketProvider, DataPointProvider, TextEnvironment
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

from typing import Callable, Tuple, Optional, Union

from explainability.data.augmentation import LocalTokenPertubator, TokenReplacement
from explainability.data.weights import pairwise_distances, exponential_kernel
from explainability.generation.surrogate import LinearSurrogate, TreeSurrogate
from explainability.generation.feature_selection import FeatureSelector
from explainability.generation.return_types import FeatureAttribution
from explainability.utils import default_detokenizer


class LocalExplanation:
    def __init__(self,
                 dataset: Optional[TextEnvironment] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 seed: int = 0):
        self.dataset = dataset
        if augmenter is None:
            augmenter = TokenReplacement(detokenizer=default_detokenizer)
        self.augmenter = augmenter
        self._seed = seed

    def augment_sample(self, sample, model,
                       sequential = False,
                       contiguous = False,
                       n_samples: int = 50,
                       avoid_proba: bool = False) -> Tuple[TextBucketProvider, np.array, np.array]:
        provider = TextBucketProvider(DataPointProvider.from_data([]), []) if self.dataset is None \
                   else self.dataset.create_empty_provider()

        sample.vector = np.ones(len(sample.tokenized), dtype=int)
        provider.add(sample)

        # Do sampling
        for perturbed_sample in self.augmenter(sample, sequential=sequential, contiguous=contiguous, n_samples=n_samples):
            provider.add(perturbed_sample)
            provider.add_child(sample, perturbed_sample)

        # Perform prediction
        if avoid_proba:
            y = model.predict(provider.bulk_get_all(), return_labels=False)
        else:
            y = model(provider.bulk_get_all(), return_labels=False)

        # Mapping to which instances were perturbed
        perturbed = np.stack([instance.vector for instance in provider.get_all()])

        return provider, perturbed, y

    def binarize(self, X: np.ndarray):
        return (X > 0).astype(int)


class WeightedExplanation:
    def __init__(self, kernel: Optional[Callable] = None, kernel_width: Union[int, float] = 25):
        """Add weights to neighborhood data.

        Args:
            kernel (Optional[Callable], optional): Kernel (if set to None defaults to `data.weights.exponential_kernel`). Defaults to None.
            kernel_width (Union[int, float], optional): Width of kernel. Defaults to 25.
        """
        if kernel is None:
            kernel = exponential_kernel
        self.kernel_fn = lambda d: kernel(d, kernel_width)

    def weigh_samples(self, a, b=None, metric='cosine'):
        if b is None:
            b = a[0]
        return self.kernel_fn(pairwise_distances(a, b, metric=metric))


class LIME(LocalExplanation, WeightedExplanation):
    def __init__(self,
                 dataset: Optional[TextEnvironment] = None,
                 local_model: Optional[LinearSurrogate] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 kernel: Optional[Callable] = None,
                 kernel_width: Union[int, float] = 25,
                 seed: int = 0):
        LocalExplanation.__init__(self, dataset=dataset, augmenter=augmenter, seed=seed)
        WeightedExplanation.__init__(self, kernel=kernel, kernel_width=kernel_width)
        if local_model is None:
            local_model = LinearSurrogate(Ridge(alpha=1, fit_intercept=True, random_state=self._seed))
        self.local_model = local_model

    def __call__(self,
                 sample,
                 model,
                 labels=(1,),
                 n_samples=50,
                 n_features=10,
                 feature_selection_method='auto',
                 weigh_samples=True,
                 distance_metric='cosine'):
        provider, perturbed, y = self.augment_sample(sample, model, sequential=False,
                                                     contiguous=False, n_samples=n_samples)
        perturbed = self.binarize(perturbed)  # flatten all n replacements into one

        if weigh_samples:
            weights = self.weigh_samples(perturbed, metric=distance_metric)

        # Get the most important features
        if feature_selection_method == 'auto':
            feature_selection_method = 'forward_selection' if n_features <= 6 else 'highest_weights'
        used_features = FeatureSelector(self.local_model)(perturbed, y,
                                                          weights=weights,
                                                          n_features=n_features,
                                                          method=feature_selection_method)

        # Fit explanation model
        self.local_model.alpha_reset()
        self.local_model.fit(perturbed[:, used_features], y, weights=weights)

        return FeatureAttribution(provider, used_features, self.local_model.feature_importances, labels=labels)#[(self.local_model.feature_importances[label], self.local_model.intercept[label]) for label in labels]


class KernelSHAP(LocalExplanation):
    def __init__(self,
                 dataset: Optional[TextEnvironment] = None,
                 augmenter: LocalTokenPertubator = None,
                 local_model: Optional[LinearSurrogate] = None,
                 seed: int = 0):
        super().__init__(dataset, augmenter, seed)
        if local_model is None:
            local_model = LinearSurrogate(Ridge(alpha=1, fit_intercept=True, random_state=self._seed))
        self.local_model = local_model
        pass

    def select_features(X: np.ndarray, y: np.ndarray, default_features: int = 1,
                        l1_reg: Union[int, float, str] = 'auto') -> np.ndarray:
        """Select features for data X and corresponding output y.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for y.
            default_features (int, optional): Default number of features, when returning all features. Defaults to 1.
            l1_reg (Union[int, float, str], optional): Method for regularization, either `auto`, `n_features({int})`,
            `{int}`, `{float}`, `aic` or `bic`. Defaults to 'auto'.

        Raises:
            Exception: Unknown value for `l1_reg`

        Returns:
            np.ndarray: Feature indices to include.
        """
        feature_selector = FeatureSelector()
        nonzero = np.arange(default_features)

        if isinstance(l1_reg, str) and l1_reg.startswith('n_features('):
            l1_reg = int(l1_reg[len('n_features('):-1])
        if isinstance(l1_reg, int):
            nonzero = feature_selector(X, y, n_features=l1_reg, method='l1_reg')
        elif isinstance(l1_reg, float):
            nonzero = feature_selector(X, y, method='l1_reg', alpha=l1_reg)
        elif l1_reg in ['auto', 'aic', 'bic']:
            if l1_reg == 'auto':
                l1_reg = 'aic'
            nonzero = feature_selector(X, y, method='l1_reg')
        else:
            raise Exception(f'Unknown value "{l1_reg}" for l1_reg')
        return nonzero

    def __call__(self, sample, model, n_samples: int = None, l1_reg: Union[int, float, str] = 'auto'):
        # https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
        sample_len = len(sample.tokenized)
        if n_samples is None:
            n_samples = 2 * sample_len + 2 ** 11
        n_samples = min(n_samples, 2 ** 30)

        provider, perturbed, y = self.augment_sample(sample, model, sequential=True,
                                                     contiguous=False, n_samples=n_samples)

        # Run local model
        self.local_model.fit(self.binarize(perturbed), y)

        # Solve
        # Feature selection
        mask_aug = None
        eyAdj_aug = None
        nonzero = self.select_features(mask_aug, eyAdj_aug, default_features=sample_len, l1_reg=l1_reg)

        if len(nonzero) == 0:
            return provider, np.zeros(sample_len), np.ones(sample_len)

        # calculate ey


class Anchor(LocalExplanation):
    def __call__(self, sample, model, n_samples: int = 50):
        # https://github.com/marcotcr/anchor/blob/master/anchor/anchor_text.py
        pass


class LocalTree(LocalExplanation, WeightedExplanation):
    def __init__(self,
                 dataset: Optional[TextEnvironment] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 local_model: Optional[TreeSurrogate] = None,
                 kernel: Optional[Callable] = None,
                 kernel_width: Union[int, float] = 25,
                 explanation_type: str='multiclass',
                 seed: int = 0):
        LocalExplanation.__init__(self, dataset=dataset, augmenter=augmenter, seed=seed)
        WeightedExplanation.__init__(self, kernel=kernel, kernel_width=kernel_width)
        if local_model is None:
            local_model = TreeSurrogate(DecisionTreeClassifier(max_depth=3))
        self.local_model = local_model
        self.explanation_type = explanation_type

    def __call__(self, sample, model, n_samples: int = 50, weigh_samples=True, distance_metric='cosine', **sample_kwargs):
        provider, perturbed, y = self.augment_sample(sample, model, n_samples=n_samples, avoid_proba=True, **sample_kwargs)
        perturbed = self.binarize(perturbed)  # flatten all n replacements into one

        # Sample weights?
        weights = self.weigh_samples(perturbed, metric=distance_metric) if weigh_samples else None
        self.local_model.fit(perturbed, y, weights=weights)

        return self.local_model.feature_importances, self.local_model
