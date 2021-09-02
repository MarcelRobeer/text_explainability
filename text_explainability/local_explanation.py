"""Local explanations explain why a model made a prediction for a single instance.

Todo:
    * Implement Anchors
"""

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from instancelib import (AbstractEnvironment, Instance, InstanceProvider,
                         LabelProvider, MemoryLabelProvider, TextEnvironment)
from instancelib.machinelearning import AbstractClassifier
from instancelib.instances.text import MemoryTextInstance, TextInstanceProvider
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

from text_explainability.data.augmentation import (LocalTokenPertubator,
                                                   TokenReplacement)
from text_explainability.data.weights import (exponential_kernel,
                                              pairwise_distances)
from text_explainability.default import Readable
from text_explainability.generation.feature_selection import FeatureSelector
from text_explainability.generation.return_types import FeatureAttribution
from text_explainability.generation.surrogate import (LinearSurrogate,
                                                      TreeSurrogate)
from text_explainability.generation.target_encoding import FactFoilEncoder
from text_explainability.utils import binarize, default_detokenizer


def default_env(env: Optional[AbstractEnvironment] = None) -> AbstractEnvironment:
    """If no environment is supplied, an empty Enviroment is created for text data.

    Args:
        env (Optional[AbstractEnvironment], optional): If a environment is supplied, it is used, otherwise.

    Returns:
        AbstractEnvironment: The default/supplied environment.
    """
    if env is not None:
        return env
    empty_dataset = TextInstanceProvider([])
    empty_labels = MemoryLabelProvider([], {})
    empty_env = TextEnvironment(empty_dataset, empty_labels)
    return empty_env


class LocalExplanation(Readable):
    def __init__(self,
                 env: Optional[AbstractEnvironment] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 labelset: Optional[Union[Sequence[str], LabelProvider]] = None,
                 seed: int = 0):
        """Generate explanation for a single decision.

        Args:
            env (Optional[AbstractEnvironment], optional): Environment to save local perturbations in. Defaults to None.
            augmenter (Optional[LocalTokenPertubator], optional):
                Function to augment data with perturbations, to generate neighborhood data. Defaults to None.
            labelset (Optional[Union[Sequence[str], LabelProvider]], optional): Sequence of label names or 
                LabelProvider containing named labels. When not supplied, it uses identifiers for labels. 
                Defaults to None.
            seed (int, optional): Seed for reproducibility. Defaults to 0.
        """
        super().__init__()
        self.env = default_env(env)
        if augmenter is None:
            augmenter = TokenReplacement(env=self.env, detokenizer=default_detokenizer)
        if isinstance(labelset, LabelProvider) and hasattr(labelset, 'labelset'):
            labelset = list(labelset.labelset)
        elif labelset is None and self.env is not None:
            if hasattr(self.env.labels, 'labelset'):
                labelset = list(self.env.labels.labelset)
        self.labelset = labelset
        self.augmenter = augmenter
        self._seed = seed

    def augment_sample(self,
                       sample: Instance,
                       model: AbstractClassifier,
                       sequential: bool = False,
                       contiguous: bool = False,
                       n_samples: int = 50,
                       add_background_instance: bool = False,
                       predict: bool = True,
                       avoid_proba: bool = False
                       ) -> Union[Tuple[InstanceProvider, np.ndarray], 
                                  Tuple[InstanceProvider, np.ndarray, np.ndarray]]:
        """Augment a single sample to generate neighborhood data.

        Args:
            sample (Instance): Instance to perturb.
            model (AbstractClassifier): Model to provide predictions for neighborhood data.
            sequential (bool, optional):
                Whether to sequentially sample based on length (first length 1, then 2, ...). Defaults to False.
            contiguous (bool, optional):
                Whether to apply perturbations on contiguous stretches of text. Defaults to False.
            n_samples (int, optional): Number of neighborhood samples to generate. Defaults to 50.
            add_background_instance (bool, optional):
                Add an additional instance with all tokens replaced. Defaults to False.
            predict (bool, optional):  Defaults to True.
            avoid_proba (bool, optional): Model predictions als labels (True) or probabilities when available (False). 
                Defaults to False.

        Returns:
            Union[Tuple[InstanceProvider, np.ndarray], Tuple[InstanceProvider, np.ndarray, np.ndarray]]:
                Provider, how instances were perturbed and optionally the corresponding predictions for each instance.
        """
        provider = self.env.create_empty_provider()

        sample.map_to_original = np.ones(len(sample.tokenized), dtype=int)
        provider.add(sample)

        # Do sampling
        augmenter = self.augmenter(sample,
                                   sequential=sequential,
                                   contiguous=contiguous,
                                   n_samples=n_samples,
                                   add_background_instance=add_background_instance)
        for perturbed_sample in augmenter.bulk_get_all():
            provider.add(perturbed_sample)

        # Perform prediction
        if predict:
            ys = model.predict_proba_raw(provider)
            y = np.stack([y_ for id, y_ in ys]).squeeze()
            if avoid_proba:
                y = np.argmax(y, axis=1)

        # Mapping to which instances were perturbed
        perturbed = np.stack([instance.map_to_original for instance in provider.get_all()])

        if predict:
            return provider, perturbed, y
        return provider, perturbed


class WeightedExplanation:
    def __init__(self, kernel: Optional[Callable] = None, kernel_width: Union[int, float] = 25):
        """Add weights to neighborhood data.

        Args:
            kernel (Optional[Callable], optional):  Kernel to determine similarity of perturbed instances to original 
                instance (if set to None defaults to `data.weights.exponential_kernel`). Defaults to None.
            kernel_width (Union[int, float], optional): Hyperparameter for similarity function of kernel. 
                Defaults to 25.
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
                 env: Optional[AbstractEnvironment],
                 local_model: Optional[LinearSurrogate] = None,
                 kernel: Optional[Callable] = None,
                 kernel_width: Union[int, float] = 25,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 labelset: Optional[Union[Sequence[str], LabelProvider]] = None,
                 seed: int = 0):
        """Local Interpretable Model-Agnostic Explanations (`LIME`_).

        Implementation of local linear surrogate model on (weighted) perturbed text data, 
        to get feature attribution scores for an example instance.

        Args:
            env (Optional[AbstractEnvironment]): Environment to save local perturbations in. Defaults to None.
            local_model (Optional[LinearSurrogate], optional): Local linear model. Defaults to None.
            kernel (Optional[Callable], optional):
                Kernel to determine similarity of perturbed instances to original instance. Defaults to None.
            kernel_width (Union[int, float], optional):
                Hyperparameter for similarity function of kernel. Defaults to 25.
            augmenter (Optional[LocalTokenPertubator], optional):
                Function to augment data with perturbations, to generate neighborhood data. Defaults to None.
            labelset (Optional[Union[Sequence[str], LabelProvider]], optional): 
                Sequence of label names or LabelProvider containing named labels. When not supplied, it uses 
                identifiers for labels. Defaults to None.
            seed (int, optional): Seed for reproducibility. Defaults to 0.

        .. _LIME:
            https://github.com/marcotcr/lime
        """
        LocalExplanation.__init__(self, env=env, augmenter=augmenter, labelset=labelset, seed=seed)
        WeightedExplanation.__init__(self, kernel=kernel, kernel_width=kernel_width)
        if local_model is None:
            local_model = LinearSurrogate(Ridge(alpha=1, fit_intercept=True, random_state=self._seed))
        self.local_model = local_model

    def __call__(self,
                 sample: MemoryTextInstance,
                 model: AbstractClassifier,
                 labels: Optional[Union[Sequence[int], Sequence[str]]] = None,
                 n_samples: int = 50,
                 n_features: int = 10,
                 feature_selection_method: str = 'auto',
                 weigh_samples: bool = True,
                 distance_metric: str = 'cosine') -> FeatureAttribution:
        """Calculate feature attribution scores using `LIME Text`_.

        Args:
            sample (MemoryTextInstance): Instance to explain.
            model (AbstractClassifier): Model to explain.
            labels (Optional[Union[Sequence[int], Sequence[str]]], optional): [description]. Defaults to None.
            n_samples (int, optional): Number of neighborhood samples to generate. Defaults to 50.
            n_features (int, optional): Maximum number of features to include (explanation length). Defaults to 10.
            feature_selection_method (str, optional): Method for limiting number of features, either
                `forward_selection`, `highest_weights` or `auto`. Defaults to 'auto'.
            weigh_samples (bool, optional): Whether to locally weigh samples based on their similarity to the original 
                instance. Defaults to True.
            distance_metric (str, optional): Distance metric for local weighting. Defaults to 'cosine'.

        Returns:
            FeatureAttribution: [description]

        .. _LIME Text:
            https://github.com/marcotcr/lime/blob/master/lime/lime_text.py
        """
        if labels is not None:
            if isinstance(labels, (int, str)):
                labels = [labels]

            n_labels = sum(1 for _ in iter(labels))
            if n_labels > 0 and isinstance(next(iter(labels)), str):
                assert self.labelset is not None, 'can only provide label names when such a list exists'
                labels = [self.labelset.index(label) for label in labels]

        # Generate neighborhood samples
        provider, perturbed, y = self.augment_sample(sample, model, sequential=False,
                                                     contiguous=False, n_samples=n_samples)
        perturbed = binarize(perturbed)  # flatten all n replacements into one

        if weigh_samples:
            weights = self.weigh_samples(perturbed, metric=distance_metric)
        if feature_selection_method == 'auto':
            feature_selection_method = 'forward_selection' if n_features <= 6 else 'highest_weights'

        feature_importances, used_features = [], {}

        if labels is None:
            labels = np.arange(y.shape[1])

        for label in labels:
            # Look at output for label
            y_label = y[:, label].copy()

            # Get the most important features
            features = FeatureSelector(self.local_model)(perturbed,
                                                         y_label,
                                                         weights=weights,
                                                         n_features=n_features,
                                                         method=feature_selection_method)
            # Fit explanation model
            self.local_model.alpha_reset()
            self.local_model.fit(perturbed[:, features], y_label, weights=weights)

            feature_importances.append(self.local_model.feature_importances)
            used_features[label] = features

        return FeatureAttribution(provider,
                                  used_features,
                                  feature_importances,
                                  labels=labels,
                                  labelset=self.labelset)


class KernelSHAP(LocalExplanation):
    def __init__(self,
                 env: Optional[AbstractEnvironment] = None,
                 labelset: Optional[Union[Sequence[str], LabelProvider]] = None,
                 augmenter: LocalTokenPertubator = None,
                 seed: int = 0):
        """Calculates `Shapley values`_ for an instance to explain, assuming the model is a black-box.

        Calculates Shapley values (local, additive feature attribution scores) for an instance 
        to explain, by calculating the average contribution of changing combinations of feature 
        values.

        Args:
            env (Optional[AbstractEnvironment], optional): Environment to save local perturbations in. Defaults to None.
            augmenter (Optional[LocalTokenPertubator], optional): Function to augment data with perturbations, 
                to generate neighborhood data. Defaults to None.
            labelset (Optional[Union[Sequence[str], LabelProvider]], optional): Sequence of label names or 
                LabelProvider containing named labels. When not supplied, it uses identifiers for labels. 
                Defaults to None.
            seed (int, optional): Seed for reproducibility. Defaults to 0.

        .. _Shapley values:
            https://github.com/slundberg/shap
        """
        super().__init__(env=env, augmenter=augmenter, labelset=labelset, seed=seed)

    @staticmethod
    def select_features(X: np.ndarray, y: np.ndarray, default_features: int = 1,
                        l1_reg: Union[int, float, str] = 'auto') -> np.ndarray:
        """Select features for data X and corresponding output y.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Prediction / ground-truth value for X.
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

    def __call__(self,
                 sample: MemoryTextInstance,
                 model: AbstractClassifier,
                 n_samples: Optional[int] = None,
                 l1_reg: Union[int, float, str] = 'auto') -> FeatureAttribution:
        """Calculate feature attribution scores using `KernelShap`_.

        Args:
            sample (MemoryTextInstance): Instance to explain.
            model (AbstractClassifier): Model to explain.
            n_samples (Optional[int], optional): Number of neighborhood samples to generate (if None defaults 
                to `2 * sample_len + 2 ** 11`). Defaults to None.
            l1_reg (Union[int, float, str], optional): Method for regularization (limiting number of features), 
                either `auto`, `n_features({int})`, `{int}`, `{float}`, `aic` or `bic`. Defaults to 'auto'.

        Returns:
            FeatureAttribution: [description]

        .. _KernelShap:
            https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
        """
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
        return FeatureAttribution(provider,
                                  used_features,
                                  scores=phi,
                                  scores_stddev=phi_var,
                                  base_score=y_null,
                                  labels=np.arange(y.shape[1]),
                                  labelset=self.labelset)


class Anchor(LocalExplanation):
    def __init__(self,
                 env: Optional[AbstractEnvironment],
                 labelset: Optional[Union[Sequence[str], LabelProvider]] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 seed: int = 0):
        super().__init__(env=env, augmenter=augmenter, labelset=labelset, seed=seed)

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
        assert batch_size > 2, 'requires positive batch size'

        y = [provider[i] for i in range(batch_size + 1)]
        y_true, y = y[0], y[1:]  # noqa: F841

        beta = np.log(1.0 / delta)
        mean = y.mean()
        lb = Anchor.dlow_bernoulli(mean, beta / perturbed.shape[0])

        batch = 1
        while mean > min_confidence and lb < min_confidence - epsilon:
            batch += 1
        pass

    def __call__(self,
                 sample: MemoryTextInstance,
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
        anchor = Anchor.beam_search(provider,  # noqa: F841
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
                 env: Optional[AbstractEnvironment] = None,
                 labelset: Optional[Union[Sequence[str], LabelProvider]] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 local_model: Optional[TreeSurrogate] = None,
                 kernel: Optional[Callable] = None,
                 kernel_width: Union[int, float] = 25,
                 explanation_type: str = 'multiclass',
                 seed: int = 0):
        LocalExplanation.__init__(self, env=env, augmenter=augmenter, labelset=labelset, seed=seed)
        WeightedExplanation.__init__(self, kernel=kernel, kernel_width=kernel_width)
        if local_model is None:
            local_model = TreeSurrogate(DecisionTreeClassifier(max_depth=3))
        self.local_model = local_model
        self.explanation_type = explanation_type

    def __call__(self,
                 sample: MemoryTextInstance,
                 model,
                 n_samples: int = 50,
                 weigh_samples: bool = True,
                 distance_metric: str = 'cosine',
                 **sample_kwargs):
        _, perturbed, y = self.augment_sample(sample,
                                              model,
                                              n_samples=n_samples,
                                              avoid_proba=True,
                                              **sample_kwargs)
        perturbed = binarize(perturbed)  # flatten all n replacements into one

        # Sample weights?
        weights = self.weigh_samples(perturbed, metric=distance_metric) if weigh_samples else None
        self.local_model.fit(perturbed, y, weights=weights)

        return self.local_model.feature_importances, self.local_model


class FoilTree(LocalExplanation, WeightedExplanation):
    def __init__(self,
                 env: Optional[AbstractEnvironment] = None,
                 labelset: Optional[Union[Sequence[str], LabelProvider]] = None,
                 augmenter: Optional[LocalTokenPertubator] = None,
                 local_model: Optional[TreeSurrogate] = DecisionTreeClassifier(),
                 kernel: Optional[Callable] = None,
                 kernel_width: Union[int, float] = 25,
                 explanation_type: str = 'multiclass',
                 seed: int = 0):
        LocalExplanation.__init__(self, env=env, augmenter=augmenter, labelset=labelset, seed=seed)
        WeightedExplanation.__init__(self, kernel=kernel, kernel_width=kernel_width)
        self.local_model = local_model
        self.explanation_type = explanation_type

    def __call__(self,
                 sample: MemoryTextInstance,
                 model: AbstractClassifier,
                 foil_fn: Union[FactFoilEncoder, int, str],
                 n_samples: int = 50,
                 weigh_samples: bool = True,
                 distance_metric: str = 'cosine',
                 **sample_kwargs):
        _, perturbed, y = self.augment_sample(sample,
                                              model,
                                              n_samples=n_samples,
                                              avoid_proba=True,
                                              **sample_kwargs)

        # Encode foil as 0 and rest as 1
        labelset = self.labelset if self.labelset else model
        if isinstance(foil_fn, str):
            foil_fn = FactFoilEncoder.from_str(foil_fn, labelset)
        elif isinstance(foil_fn, int):
            foil_fn = FactFoilEncoder(foil_fn, labelset)
        y_ = foil_fn(y)

        perturbed = binarize(perturbed)  # flatten all n replacements into one

        # weights = self.weigh_samples(perturbed, metric=distance_metric) if weigh_samples else None
        # self.local_model.fit(perturbed, y_, weights=weights)
        self.local_model.fit(perturbed, y_)

        return self.local_model, self.local_model.tree_.max_depth
