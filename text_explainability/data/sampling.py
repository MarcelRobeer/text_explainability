"""Sample an (informative) subset from the data.

Todo:
- Sample (informative?) subset from data
- Prototype sampling (k-medoids)
- Refactor to make sampling base class
- Add ability to perform MMD critic on a subset (e.g. single class)
"""

from typing import Dict, Sequence, Callable, Optional

import numpy as np
from instancelib.instances.memory import MemoryBucketProvider
from instancelib.instances.text import MemoryTextInstance

from text_explainability.data.embedding import Embedder, SentenceTransformer
from text_explainability.data.weights import exponential_kernel
from text_explainability.default import Readable


class MMDCritic(Readable):
    def __init__(self,
                 instances: MemoryBucketProvider,
                 embedder: Embedder = SentenceTransformer(),
                 kernel: Callable = exponential_kernel):
        """Select prototypes and criticisms based on embedding distances using `MMD-Critic`_.

        Args:
            instances (MemoryBucketProvider): Instances to select from (e.g. training set, all instance from class 0).
            embedder (Embedder, optional): Method to embed instances (if the `.vector` property is not yet set). 
                Defaults to SentenceTransformer().
            kernel (Callable, optional): Kernel to calculate distances. Defaults to exponential_kernel.

        .. _MMD-critic:
            https://christophm.github.io/interpretable-ml-book/proto.html
        """
        self.embedder = embedder
        self.kernel = kernel
        self.instances = self.embedder(instances) if any(instances[i].vector is None for i in instances) \
                         else instances
        self._calculate_kernel()
        self._prototypes = None
        self._criticisms = None

    def _calculate_kernel(self):
        """Calculate kernel `K` and column totals `colsum`."""
        instances = np.stack(self.instances.bulk_get_vectors(list(self.instances))[-1])
        self.K = self.kernel(instances, 1.0 / instances.shape[1])
        self.colsum = np.sum(self.K, axis=0) / instances.shape[1]

    def _select_from_provider(self, keys: Sequence[int]) -> Sequence[MemoryTextInstance]:
        """Select instances from provider by keys."""
        return [self.instances[i] for i in keys]

    def prototypes(self, n: int = 5) -> Sequence[MemoryTextInstance]:
        """Select `n` prototypes (most representatitve instances), using `MMD-critic implementation`_.

        Args:
            n (int, optional): Number of prototypes to select. Defaults to 5.

        Returns:
            Sequence[MemoryTextInstance]: List of prototype instances.

        .. _MMD-critic implementation:
            https://github.com/maxidl/MMD-critic/blob/main/mmd_critic.py
        """
        assert n <= len(self.instances), f'Cannot select more than all instances ({len(self.instances)}.'
        K = self.K
        colsum = self.colsum.copy() * 2
        sample_indices = np.array(list(self.instances))
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

    def criticisms(self, n: int = 5, regularizer: Optional[str] = None) -> Sequence[MemoryTextInstance]:
        """Select `n` criticisms (instances not well represented by prototypes), using `MMD-critic implementation`_. 

        Args:
            n (int, optional): Number of criticisms to select. Defaults to 5.
            regularizer (Optional[str], optional): Regularization method. Choose from [None, 'logdet', 'iterative']. 
                Defaults to None.

        Raises:
            Exception: `MMDCritic.prototypes()` must first be run before being able to determine the criticisms.

        Returns:
            Sequence[MemoryTextInstance]: List of criticism instances.

        .. _MMD-critic implementation:
            https://github.com/maxidl/MMD-critic/blob/main/mmd_critic.py
        """
        if self._prototypes is None:
            raise Exception('Calculating criticisms requires prototypes. Run `MMDCritic.prototypes()` first.')
        regularizers = {None, 'logdet', 'iterative'}
        assert regularizer in regularizers, \
            f'Unknown regularizer "{regularizer}", choose from {regularizers}.'
        assert n <= (len(self.instances) - len(self._prototypes)), \
            f'Cannot select more than instances excluding prototypes ({len(self.instances) - len(self._prototypes)})'

        prototypes = np.array([p.identifier for p in self._prototypes])

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
                diag = np.diagonal(K)[candidate_indices]
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
                 regularizer: Optional[str] = None) -> Dict[str, Sequence[MemoryTextInstance]]:
        """Calculate prototypes and criticisms for the provided instances.

        Args:
            n_prototypes (int, optional): Number of prototypes. Defaults to 5.
            n_criticisms (int, optional): Number of criticisms. Defaults to 5.
            regularizer (Optional[str], optional): Regularization method. Choose from [None, 'logdet', 'iterative']. 
                Defaults to None.

        Returns:
            Dict[str, Sequence[MemoryTextInstance]]: Dictionary containing prototypes and criticisms.
        """
        return {'prototypes': self.prototypes(n=n_prototypes),
                'criticisms': self.criticisms(n=n_criticisms, regularizer=regularizer)}
