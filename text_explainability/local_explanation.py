"""TO-DO:
- Implement Anchors
- Implement Foil Trees
"""

import math
import numpy as np

from instancelib import AbstractEnvironment, Instance, TextInstance, InstanceProvider
from instancelib.labels import LabelProvider
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

from typing import Callable, Tuple, Optional, Union, Sequence

from text_explainability.data.augmentation import LocalTokenPertubator, TokenReplacement
from text_explainability.data.weights import pairwise_distances, exponential_kernel
from text_explainability.generation.surrogate import LinearSurrogate, TreeSurrogate
from text_explainability.generation.feature_selection import FeatureSelector
from text_explainability.generation.return_types import FeatureAttribution
from text_explainability.default import Readable
from text_explainability.utils import default_detokenizer, binarize


class LocalExplanation(Readable):
    def __init__(self,
                 dataset: AbstractEnvironment = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 label_names: Optional[Union[Sequence[str], LabelProvider]] = None,
                 seed: int = 0):
        super().__init__()
        self.dataset = dataset
        if augmenter is None:
            augmenter = TokenReplacement(detokenizer=default_detokenizer)
        if isinstance(label_names, LabelProvider) and hasattr(label_names, 'labelset'):
            label_names = list(label_names.labelset)
        elif label_names is None and self.dataset is not None:
            if hasattr(self.dataset.labels, 'labelset'):
                label_names = list(self.dataset.labels.labelset)
        self.label_names = label_names
        self.augmenter = augmenter
        self._seed = seed

    def augment_sample(self,
                       sample: Instance,
                       model,
                       sequential: bool = False,
                       contiguous: bool = False,
                       n_samples: int = 50,
                       add_background_instance: bool = False,
                       predict: bool = True,
                       avoid_proba: bool = False
                       ) -> Union[Tuple[InstanceProvider, np.ndarray], \
                                  Tuple[InstanceProvider, np.ndarray, np.ndarray]]:
        provider = self.dataset.create_empty_provider()

        sample.vector = np.ones(len(sample.tokenized), dtype=int)
        provider.add(sample)

        # Do sampling
        augmenter = self.augmenter(sample,
                                   sequential=sequential,
                                   contiguous=contiguous,
                                   n_samples=n_samples,
                                   add_background_instance=add_background_instance)
        for perturbed_sample in augmenter:
            provider.add(perturbed_sample)
            provider.add_child(sample, perturbed_sample)

        # Perform prediction
        if predict:
            if avoid_proba:
                y = model.predict(provider.bulk_get_all(), return_labels=False)
            else:
                y = model(provider.bulk_get_all(), return_labels=False)

        # Mapping to which instances were perturbed
        perturbed = np.stack([instance.vector for instance in provider.get_all()])

        if predict:
            return provider, perturbed, y
        return provider, perturbed


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
                 dataset: AbstractEnvironment = None,
                 label_names: Optional[Union[Sequence[str], LabelProvider]]  = None,
                 local_model: Optional[LinearSurrogate] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 kernel: Optional[Callable] = None,
                 kernel_width: Union[int, float] = 25,
                 seed: int = 0):
        LocalExplanation.__init__(self, dataset=dataset, augmenter=augmenter, label_names=label_names, seed=seed)
        WeightedExplanation.__init__(self, kernel=kernel, kernel_width=kernel_width)
        if local_model is None:
            local_model = LinearSurrogate(Ridge(alpha=1, fit_intercept=True, random_state=self._seed))
        self.local_model = local_model

    def __call__(self,
                 sample: TextInstance,
                 model,
                 labels: Optional[Union[Sequence[int], Sequence[str]]] = None,
                 n_samples: int = 50,
                 n_features: int = 10,
                 feature_selection_method: str = 'auto',
                 weigh_samples: bool = True,
                 distance_metric: str = 'cosine') -> FeatureAttribution:
        if labels is not None:
            n_labels = sum(1 for _ in iter(labels))
            if n_labels > 0 and isinstance(next(iter(labels)), str):
                assert self.label_names is not None, 'can only provide label names when such a list exists'
                labels = [self.label_names.index(label) for label in labels]
        provider, perturbed, y = self.augment_sample(sample, model, sequential=False,
                                                     contiguous=False, n_samples=n_samples)
        perturbed = binarize(perturbed)  # flatten all n replacements into one

        if weigh_samples:
            weights = self.weigh_samples(perturbed, metric=distance_metric)

        # Get the most important features
        if feature_selection_method == 'auto':
            feature_selection_method = 'forward_selection' if n_features <= 6 else 'highest_weights'
        used_features = FeatureSelector(self.local_model)(perturbed,
                                                          y,
                                                          weights=weights,
                                                          n_features=n_features,
                                                          method=feature_selection_method)

        # Fit explanation model
        self.local_model.alpha_reset()
        self.local_model.fit(perturbed[:, used_features], y, weights=weights)

        if labels is None:
            labels = np.arange(y.shape[1])

        return FeatureAttribution(provider, used_features, self.local_model.feature_importances, labels=labels, label_names=self.label_names)


class KernelSHAP(LocalExplanation):
    def __init__(self,
                 dataset: AbstractEnvironment,
                 label_names: Optional[Union[Sequence[str], LabelProvider]]  = None,
                 augmenter: LocalTokenPertubator = None,
                 seed: int = 0):
        super().__init__(dataset=dataset, augmenter=augmenter, label_names=label_names, seed=seed)

    @staticmethod
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
            nonzero = feature_selector(X, y, method=l1_reg)
        else:
            raise Exception(f'Unknown value "{l1_reg}" for l1_reg')
        return nonzero

    def __call__(self, sample: TextInstance, model, n_samples: Optional[int] = None, l1_reg: Union[int, float, str] = 'auto'):
        # https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
        sample_len = len(sample.tokenized)
        if n_samples is None:
            n_samples = 2 * sample_len + 2 ** 11
        n_samples = min(n_samples, 2 ** 30)

        provider, perturbed, y = self.augment_sample(sample, model, sequential=True,
                                                     contiguous=False, n_samples=n_samples,
                                                     add_background_instance=True)

        # To-do: exclude non-varying feature groups
        y_null, y = y[-1], y[1:-1]
        y -= y_null
        used_features = np.arange(perturbed.shape[1])
        phi = np.zeros([sample_len, y.shape[1]])
        phi_var = np.zeros(sample_len)

        if perturbed.shape[1] == 1:
            phi = np.mean(y - y_null, axis=0).reshape(1, -1)
        elif perturbed.shape[1] > 1:
            # Weigh samples
            M = perturbed.shape[1]
            Z = np.sum(perturbed[1:-1], axis=1).astype(int)
            weight_vector = np.array([(M - 1) / (math.comb(M, m) * m * (M - m)) for m in range(1, M)])
            weight_vector /= np.sum(weight_vector)
            kernel_weights = weight_vector[Z - 1]

            nonzero = KernelSHAP.select_features(perturbed[1:-1], y, default_features=sample_len, l1_reg=l1_reg)
            used_features = nonzero
            phi_var = np.ones(sample_len)
            if len(used_features) > 0:
                X = perturbed[1:-1]
                X_W = np.dot(X.T, np.diag(kernel_weights))
                try:
                    tmp2 = np.linalg.inv(np.dot(X_W, X))
                except np.linalg.LinAlgError:
                    tmp2 = np.linalg.pinv(np.dot(X_W, X))
                phi = np.dot(tmp2, np.dot(X_W, y)).T
        return FeatureAttribution(provider, used_features,
                                  scores=phi,
                                  scores_stddev=phi_var,
                                  base_score=y_null,
                                  labels=np.arange(y.shape[1]),
                                  label_names=self.label_names)

class Anchor(LocalExplanation):
    def __init__(self,
                 dataset: Optional[AbstractEnvironment] = None,
                 label_names: Optional[Union[Sequence[str], LabelProvider]]  = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 seed: int = 0):
        super().__init__(dataset=dataset, augmenter=augmenter, label_names=label_names, seed=seed)

    @staticmethod
    def kl_bernoulli(p, q):
        p = float(min(1 - 1e-15, max(1e-15, p)))
        q = float(min(1 - 1e-15, max(1e-15, q)))
        return (p * np.log(p / q) + (1 - p) *
                np.log((1 - p) / (1 - q)))

    @staticmethod
    def dlow_bernoulli(p, level):
        lm = max(min(1, p - np.sqrt(level / 2.0)), 0.0)
        qm = (p + lm) / 2.0
        if Anchor.kl_bernoulli(p, qm) > level:
            lm = qm
        return lm

    def generate_candidates(self,):
        pass

    def best_candidate(self):
        pass

    @staticmethod
    def beam_search(provider,
                    perturbed: np.ndarray,
                    model,
                    beam_size: int = 1,
                    min_confidence: float = 0.95,
                    delta: float = 0.05,
                    epsilon: float = 0.1,
                    max_anchor_size: Optional[int] = None,
                    batch_size: int = 20):
        assert beam_size >= 1, f'beam size should be at least 1, but is {beam_size}'
        assert 0.0 <= min_confidence <= 0.95, f'min_confidence should be a value in [0, 1], but is {min_confidence}'
        assert 0.0 <= delta <= 0.95, f'delta should be a value in [0, 1], but is {delta}'
        assert 0.0 <= epsilon <= 0.95, f'epsilon should be a value in [0, 1], but is {epsilon}'
        assert batch_size > 2, f'requires positive batch size'

        y = [provider[i] for i in range(batch_size + 1)]
        y_true, y = y[0], y[1:]

        beta = np.log(1.0 / delta)
        mean = y.mean()
        lb = Anchor.dlow_bernoulli(mean, beta / perturbed.shape[0])

        batch = 1
        while mean > min_confidence and lb < min_confidence - epsilon:
            
            batch += 1
        pass

    def __call__(self,
                 sample: TextInstance,
                 model,
                 n_samples: int = 100,
                 beam_size: int = 1,
                 min_confidence: float = 0.95,
                 delta: float = 0.05,
                 epsilon: float = 0.1,
                 max_anchor_size: Optional[int] = None):
        raise NotImplementedError('Only partially implemented')
        # https://github.com/marcotcr/anchor/blob/master/anchor/anchor_text.py
        # https://github.com/marcotcr/anchor/blob/master/anchor/anchor_base.py
        provider, perturbed = self.augment_sample(sample, None, sequential=False,
                                                  contiguous=False, n_samples=n_samples,
                                                  predict=False)
        perturbed = binarize(perturbed[1:])  # flatten all n replacements into one
        y_true = model(provider[0])
        y_true = np.argmax(y_true) if y_true.ndim > 1 else y_true

        # Use beam from https://homes.cs.washington.edu/~marcotcr/aaai18.pdf (Algorithm 2)
        anchor = Anchor.beam_search(provider,
                                    perturbed,
                                    model,
                                    beam_size=beam_size,
                                    min_confidence=min_confidence,
                                    delta=delta,
                                    epsilon=epsilon,
                                    max_anchor_size=max_anchor_size,
                                    batch_size=n_samples // 10 if n_samples >= 1000 else n_samples // 5)
        pass


class LocalTree(LocalExplanation, WeightedExplanation):
    def __init__(self,
                 dataset: AbstractEnvironment,
                 label_names: Optional[Union[Sequence[str], LabelProvider]]  = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 local_model: Optional[TreeSurrogate] = None,
                 kernel: Optional[Callable] = None,
                 kernel_width: Union[int, float] = 25,
                 explanation_type: str = 'multiclass',
                 seed: int = 0):
        LocalExplanation.__init__(self, dataset=dataset, augmenter=augmenter, label_names=label_names, seed=seed)
        WeightedExplanation.__init__(self, kernel=kernel, kernel_width=kernel_width)
        if local_model is None:
            local_model = TreeSurrogate(DecisionTreeClassifier(max_depth=3))
        self.local_model = local_model
        self.explanation_type = explanation_type

    def __call__(self,
                 sample: TextInstance,
                 model,
                 n_samples: int = 50,
                 weigh_samples: bool = True,
                 distance_metric: str = 'cosine',
                 **sample_kwargs):
        provider, perturbed, y = self.augment_sample(sample, model, n_samples=n_samples, avoid_proba=True, **sample_kwargs)
        perturbed = binarize(perturbed)  # flatten all n replacements into one

        # Sample weights?
        weights = self.weigh_samples(perturbed, metric=distance_metric) if weigh_samples else None
        self.local_model.fit(perturbed, y, weights=weights)

        return self.local_model.feature_importances, self.local_model
