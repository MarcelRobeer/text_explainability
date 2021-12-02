"""Sample an (informative) subset from the data.

Todo:

    * Sample (informative?) subset from data
    * Refactor to make sampling base class
    * Add ability to perform MMD critic on a subset (e.g. single class)
"""

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
from genbase import Readable, SeedMixin
from instancelib.instances.memory import DataPoint, MemoryBucketProvider
from instancelib.labels.base import LabelProvider
from instancelib.labels.memory import MemoryLabelProvider
from instancelib.machinelearning.base import AbstractClassifier

from .embedding import Embedder, TfidfVectorizer
from .weights import exponential_kernel


class PrototypeSampler(Readable):
    def __init__(self,
                 instances: MemoryBucketProvider,
                 embedder: Embedder = TfidfVectorizer):
        """Generic class for sampling prototypes (representative samples) based on embedding distances.

        Args:
            instances (MemoryBucketProvider): Instances to select from (e.g. training set, all instance from class 0).
            embedder (Embedder, optional): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to TfidfVectorizer.
        """
        self.embedder = embedder() if isinstance(embedder, type) else embedder
        self.instances = self.embedder(instances) if any(instances[i].vector is None for i in instances) \
                         else instances

    @property
    def embedded(self) -> np.ndarray:
        return np.stack(self.instances.bulk_get_vectors(list(self.instances))[-1])

    def _select_from_provider(self, keys: Sequence[int]) -> Sequence[DataPoint]:
        """Select instances from provider by keys."""
        id_map = np.array(self.instances)
        return [self.instances[id_map[i]] for i in keys]

    def prototypes(self, n: int = 5) -> Sequence[DataPoint]:
        """Select `n` prototypes.

        Args:
            n (int, optional): Number of prototypes to select. Defaults to 5.

        Returns:
            Sequence[DataPoint]: List of prototype instances.
        """
        raise NotImplementedError('Implemented in subclasses')

    def __call__(self, *args, **kwargs):
        return self.prototypes(*args, **kwargs)


class KMedoids(PrototypeSampler, SeedMixin):
    def __init__(self,
                 instances: MemoryBucketProvider,
                 embedder: Embedder = TfidfVectorizer,
                 seed: int = 0):
        """Sampling prototypes (representative samples) based on embedding distances using `k-Medoids`_.

        Args:
            instances (MemoryBucketProvider): Instances to select from (e.g. training set, all instance from class 0).
            embedder (Embedder, optional): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to TfidfVectorizer.
            seed (int, optional): Seed for reproducibility. Defaults to 0.

        .. _k-Medoids:
            https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
        """
        super().__init__(instances, embedder)
        self._seed = self._original_seed = seed

    def prototypes(self,
                   n: int = 5,
                   metric: Union[str, Callable] = 'cosine',
                   **kwargs) -> Sequence[DataPoint]:
        """Select `n` prototypes (most representative samples) using `k-Medoids`_.

        Args:
            n (int, optional): Number of prototypes to select. Defaults to 5.
            metrics (Union[str, Callable], optional): Distance metric used to calculate medoids (e.g. 'cosine', 
                'euclidean' or your own function). See `pairwise distances` for a full list. Defaults to 'cosine'.
            **kwargs: Optional arguments passed to `k-Medoids`_ constructor.

        Returns:
            Sequence[DataPoint]: List of prototype instances.

        .. _k-Medoids:
            https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
        .. _pairwise distances:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        """
        from sklearn_extra.cluster import KMedoids
        kmedoids = KMedoids(n_clusters=n, metric=metric, random_state=self.seed, **kwargs).fit(self.embedded)
        return self._select_from_provider(kmedoids.medoid_indices_)


class MMDCritic(PrototypeSampler):
    def __init__(self,
                 instances: MemoryBucketProvider,
                 embedder: Embedder = TfidfVectorizer,
                 kernel: Callable = exponential_kernel):
        """Select prototypes and criticisms based on embedding distances using `MMD-Critic`_.

        Args:
            instances (MemoryBucketProvider): Instances to select from (e.g. training set, all instance from class 0).
            embedder (Embedder, optional): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to TfidfVectorizer.
            kernel (Callable, optional): Kernel to calculate distances. Defaults to exponential_kernel.

        .. _MMD-critic:
            https://christophm.github.io/interpretable-ml-book/proto.html
        """
        super().__init__(instances, embedder)
        self.kernel = kernel
        self._calculate_kernel()
        self._prototypes = None
        self._criticisms = None

    def _calculate_kernel(self):
        """Calculate kernel `K` and column totals `colsum`."""
        self.K = self.kernel(self.embedded, 1.0 / self.embedded.shape[1])
        self.colsum = np.sum(self.K, axis=0) / self.embedded.shape[1]

    def prototypes(self, n: int = 5) -> Sequence[DataPoint]:
        """Select `n` prototypes (most representatitve instances), using `MMD-critic implementation`_.

        Args:
            n (int, optional): Number of prototypes to select. Defaults to 5.

        Raises:
            ValueError: Cannot select more instances than the total number of instances.

        Returns:
            Sequence[DataPoint]: List of prototype instances.

        .. _MMD-critic implementation:
            https://github.com/maxidl/MMD-critic/blob/main/mmd_critic.py
        """
        if n > len(self.instances):
            raise ValueError(f'Cannot select more than all instances ({len(self.instances)}.')

        K = self.K
        colsum = self.colsum.copy() * 2
        sample_indices = np.arange(0, len(self.instances))
        is_selected = np.zeros_like(sample_indices)
        selected = sample_indices[is_selected > 0]

        for i in range(n):
            candidate_indices = sample_indices[is_selected == 0]
            s1 = colsum[candidate_indices]

            diag = np.diagonal(K)[candidate_indices]
            if selected.shape[0] == 0:
                s1 -= np.abs(diag)
            else:
                temp = K[selected, :][:, candidate_indices]
                s2 = np.sum(temp, axis=0) * 2 + diag
                s2 /= (selected.shape[0] + 1)
                s1 -= s2

            best_sample_index = candidate_indices[np.argmax(s1)]
            is_selected[best_sample_index] = i + 1
            selected = sample_indices[is_selected > 0]

        selected_in_order = selected[is_selected[is_selected > 0].argsort()]
        self._prototypes = self._select_from_provider(selected_in_order)
        return self._prototypes

    def criticisms(self, n: int = 5, regularizer: Optional[str] = None) -> Sequence[DataPoint]:
        """Select `n` criticisms (instances not well represented by prototypes), using `MMD-critic implementation`_. 

        Args:
            n (int, optional): Number of criticisms to select. Defaults to 5.
            regularizer (Optional[str], optional): Regularization method. Choose from [None, 'logdet', 'iterative']. 
                Defaults to None.

        Raises:
            Exception: `MMDCritic.prototypes()` must first be run before being able to determine the criticisms.
            ValueError: Unknown regularizer or requested more criticisms than there are samples left.

        Returns:
            Sequence[DataPoint]: List of criticism instances.

        .. _MMD-critic implementation:
            https://github.com/maxidl/MMD-critic/blob/main/mmd_critic.py
        """
        if self._prototypes is None:
            raise Exception('Calculating criticisms requires prototypes. Run `MMDCritic.prototypes()` first.')
        regularizers = {None, 'logdet', 'iterative'}

        if regularizer not in regularizers:
            raise ValueError(f'Unknown {regularizer=}. Choose from {regularizers}.')
        if n > (len(self.instances) - len(self._prototypes)):
            raise ValueError('Cannot select more than instances excluding prototypes ',
                             f'({len(self.instances) - len(self._prototypes)})')

        id_map = {instance: id for id, instance in enumerate(self.instances)}
        prototypes = np.array([id_map[p.identifier] for p in self._prototypes])

        K = self.K
        colsum = self.colsum
        sample_indices = np.arange(0, len(self.instances))
        is_selected = np.zeros_like(sample_indices)
        selected = sample_indices[is_selected > 0]
        is_selected[prototypes] = n + 1

        inverse_of_prev_selected = None
        for i in range(n):
            candidate_indices = sample_indices[is_selected == 0]
            s1 = colsum[candidate_indices]

            temp = K[prototypes, :][:, candidate_indices]
            s2 = np.sum(temp, axis=0)
            s2 /= prototypes.shape[0]
            s1 -= s2
            s1 = np.abs(s1)

            if regularizer == 'logdet':
                diag = np.diagonal(K + 1)[candidate_indices]
                if inverse_of_prev_selected is not None:
                    temp = K[selected, :][:, candidate_indices]
                    temp2 = np.dot(inverse_of_prev_selected, temp) 
                    reg = temp2 * temp
                    regcolsum = np.sum(reg, axis=0)
                    with np.errstate(divide='ignore'):
                        reg = np.log(np.abs(diag - regcolsum))
                    s1 += reg
                else:
                    with np.errstate(divide='ignore'):
                        s1 -= np.log(np.abs(diag))

            best_sample_index = candidate_indices[np.argmax(s1)]
            is_selected[best_sample_index] = i + 1

            selected = sample_indices[(is_selected > 0) & (is_selected != (n + 1))]

            if regularizer == 'iterative':
                prototypes = np.concatenate([prototypes, np.expand_dims(best_sample_index, 0)])

            if regularizer == 'logdet':
                inverse_of_prev_selected = np.linalg.pinv(K[selected, :][:, selected])

        selected_in_order = selected[is_selected[(is_selected > 0) & (is_selected != (n + 1))].argsort()]      
        self._criticisms = self._select_from_provider(selected_in_order)
        return self._criticisms

    def __call__(self,
                 n_prototypes: int = 5,
                 n_criticisms: int = 5,
                 regularizer: Optional[str] = None) -> Dict[str, Sequence[DataPoint]]:
        """Calculate prototypes and criticisms for the provided instances.

        Args:
            n_prototypes (int, optional): Number of prototypes. Defaults to 5.
            n_criticisms (int, optional): Number of criticisms. Defaults to 5.
            regularizer (Optional[str], optional): Regularization method. Choose from [None, 'logdet', 'iterative']. 
                Defaults to None.

        Returns:
            Dict[str, Sequence[DataPoint]]: Dictionary containing prototypes and criticisms.
        """
        return {'prototypes': self.prototypes(n=n_prototypes),
                'criticisms': self.criticisms(n=n_criticisms, regularizer=regularizer)}


class LabelwisePrototypeSampler(Readable):
    def __init__(self,
                 sampler: PrototypeSampler,
                 instances: MemoryBucketProvider,
                 labels: Union[Sequence[str], Sequence[int], LabelProvider, AbstractClassifier],
                 embedder: Embedder = TfidfVectorizer,
                 **kwargs):
        """Apply `PrototypeSampler()` for each label.

        Args:
            sampler (PrototypeSampler): Prototype sampler to construct (e.g. `KMedoids`, `MMDCritic`)
            instances (MemoryBucketProvider): Instances to select from (e.g. training set, all instance from class 0).
            labels (Union[Sequence[str], Sequence[int], LabelProvider, AbstractClassifier]): Ground-truth or predicted 
                labels, providing the groups (e.g. classes) in which to subdivide the instances.
            embedder (Embedder, optional): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to TfidfVectorizer.
            **kwargs: Additional arguments passed to `_setup_instances()` constructor.
        """
        self.sampler = sampler if isinstance(sampler, type) else self.sampler.__class__
        self.instances = instances
        self._get_labels(labels)
        self._setup_samplers(embedder, **kwargs)

    def _get_labels(self,
                    labels: Union[Sequence[str], Sequence[int], LabelProvider, AbstractClassifier]):
        """Transform the labels into a `LabelProvider`."""   
        if not isinstance(labels, LabelProvider):
            if isinstance(labels, AbstractClassifier):
                labels_ = labels.predict(self.instances)
            else:
                labels_ = [(id, frozenset({label})) for id, label in zip(list(self.instances), labels)]
            labels = MemoryLabelProvider.from_tuples(labels_)
        self.labels = labels

    def _setup_samplers(self,
                        embedder: Embedder,
                        **kwargs):
        """Setup a sampler for each label in `self.labels.labelset`.

        Args:
            embedder (Embedder): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to TfidfVectorizer.
            **kwargs: Additional arguments passed to sampler constructor.
        """
        import copy

        def select_by_label(label):
            instances = copy.deepcopy(self.instances)
            keys_to_keep = self.labels.get_instances_by_label(label)
            instances._remove_from_bucket(frozenset(list(instances)).difference(keys_to_keep))
            return instances

        self._samplers = {label: self.sampler(instances=select_by_label(label),
                                              embedder=embedder,
                                              **kwargs)
                          for label in self.labels.labelset}
        self.samplers = self._samplers

    def prototypes(self, n: int = 5) -> Dict[str, Sequence[DataPoint]]:
        """Select `n` prototypes (most representatitve instances).

        Args:
            n (int, optional): Number of prototypes to select. Defaults to 5.

        Returns:
            Dict[str, Sequence[DataPoint]]: Dictionary with labels and corresponding list of prototypes.
        """
        return {label: sampler.prototypes(n=n)
                for label, sampler in self._samplers.items()}

    def __call__(self,
                 n: int = 5) -> Dict[str, Dict[str, Sequence[DataPoint]]]:
        """Generate prototypes for each label.

        Args:
            n (int, optional): Number of prototypes to select. Defaults to 5.

        Returns:
            Dict[str, Dict[str, Sequence[DataPoint]]]: Dictionary with labels and corresponding dictionary 
                containing prototypes.
        """
        return {label: {'prototypes': sampler.prototypes(n=n)}
                for label, sampler in self._samplers.items()}


class LabelwiseKMedoids(LabelwisePrototypeSampler):
    def __init__(self,
                 instances: MemoryBucketProvider,
                 labels: Union[Sequence[str], Sequence[int], LabelProvider],
                 embedder: Embedder = TfidfVectorizer,
                 seed: int = 0):
        """Select prototypes for each label based on embedding distances using `k-Medoids`_.

        Args:
            instances (MemoryBucketProvider): Instances to select from (e.g. training set, all instance from class 0).
            labels (Union[Sequence[str], Sequence[int], LabelProvider]): Ground-truth or predicted labels, providing 
                the groups (e.g. classes) in which to subdivide the instances.
            embedder (Embedder, optional): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to TfidfVectorizer.
            seed (int, optional): Seed for reproducibility. Defaults to 0.

        .. _k-Medoids:
            https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
        """
        super().__init__(KMedoids,
                         instances=instances,
                         labels=labels,
                         embedder=embedder,
                         seed=seed)


class LabelwiseMMDCritic(LabelwisePrototypeSampler):
    def __init__(self,
                 instances: MemoryBucketProvider,
                 labels: Union[Sequence[str], Sequence[int], LabelProvider],
                 embedder: Embedder = TfidfVectorizer,
                 kernel: Callable = exponential_kernel):
        """Select prototypes and criticisms for each label based on embedding distances using `MMD-Critic`_.

        Args:
            instances (MemoryBucketProvider): Instances to select from (e.g. training set, all instance from class 0).
            labels (Union[Sequence[str], Sequence[int], LabelProvider]): Ground-truth or predicted labels, providing 
                the groups (e.g. classes) in which to subdivide the instances.
            embedder (Embedder, optional): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to TfidfVectorizer.
            kernel (Callable, optional): Kernel to calculate distances. Defaults to exponential_kernel.

        .. _MMD-critic:
            https://christophm.github.io/interpretable-ml-book/proto.html
        """
        super().__init__(MMDCritic,
                         instances=instances,
                         labels=labels,
                         embedder=embedder,
                         kernel=kernel)

    def criticisms(self, n: int = 5, regularizer: Optional[str] = None) -> Dict[str, Sequence[DataPoint]]:
        """Select `n` criticisms (instances not well represented by prototypes).

        Args:
            n (int, optional): Number of criticisms to select. Defaults to 5.
            regularizer (Optional[str], optional): Regularization method. Choose from [None, 'logdet', 'iterative']. 
                Defaults to None.

        Raises:
            Exception: `MMDCritic.prototypes()` must first be run before being able to determine the criticisms.

        Returns:
            Dict[str, Sequence[DataPoint]]: Dictionary with labels and corresponding list of criticisms.
        """
        return {label: sampler.criticisms(n=n, regularizer=regularizer)
                for label, sampler in self._samplers.items()}

    def __call__(self,
                 n_prototypes: int = 5,
                 n_criticisms: int = 5,
                 regularizer: Optional[str] = None) -> Dict[str, Dict[str, Sequence[DataPoint]]]:
        """Generate prototypes and criticisms for each label.

        Args:
            n_prototypes (int, optional): Number of prototypes to select. Defaults to 5.
            n_criticisms (int, optional): Number of criticisms to select. Defaults to 5.
            regularizer (Optional[str], optional): Regularization method. Choose from [None, 'logdet', 'iterative']. 
                Defaults to None.

        Returns:
            Dict[str, Dict[str, Sequence[DataPoint]]]: Dictionary with labels and corresponding dictionary 
                containing prototypes and criticisms.
        """
        return {label: sampler(n_prototypes=n_prototypes, n_criticisms=n_criticisms, regularizer=regularizer)
                for label, sampler in self._samplers.items()}
