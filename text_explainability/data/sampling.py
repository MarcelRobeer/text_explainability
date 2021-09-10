"""Sample an (informative) subset from the data.

Todo:
- Sample (informative?) subset from data
- Prototype sampling
- Refactor to make sampling base class
- Add ability to perform MMD critic on a subset (e.g. single class)
"""

from typing import Sequence, Callable, Optional

import numpy as np
from instancelib.instances.memory import MemoryBucketProvider

from text_explainability.data.embedding import Embedder, SentenceTransformer
from text_explainability.data.weights import exponential_kernel
from text_explainability.default import Readable


class MMDCritic(Readable):
    def __init__(self,
                 instances: MemoryBucketProvider,
                 embedder: Embedder = SentenceTransformer(),
                 kernel: Callable = exponential_kernel):
        self.embedder = embedder
        self.kernel = kernel
        self.instances = self.embedder(instances) if any(instances[i].vector is None for i in instances) \
                         else instances
        self._calculate_diag()
        self._prototypes = None
        self._criticisms = None

    def _calculate_diag(self):
        instances = np.stack(self.instances.bulk_get_vectors(list(self.instances))[-1])
        self.K = self.kernel(instances, 1.0 / instances.shape[1])
        self.colsum = np.sum(self.K, axis=0) / instances.shape[1]

    def _select_from_provider(self, keys: Sequence[int]):
        return [self.instances[i] for i in keys]

    def prototypes(self, n: int = 5):
        # https://github.com/maxidl/MMD-critic/blob/main/mmd_critic.py
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

    def criticisms(self, n: int = 5, regularizer: Optional[str] = None):
        if self._prototypes is None:
            raise Exception('Calculating criticisms requires prototypes. Run `MMDCritic.prototypes()` first.')
        available_regularizers = {None, 'logdet', 'iterative'}
        assert regularizer in available_regularizers, \
            f'Unknown regularizer "{regularizer}", choose from {available_regularizers}'

        # https://github.com/maxidl/MMD-critic/blob/main/mmd_critic.py
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

    def __call__(self, n_prototypes: int = 5, n_criticisms: int = 5, regularizer: Optional[str] = None):
        return {'prototypes': self.prototypes(n=n_prototypes),
                'criticisms': self.criticisms(n=n_criticisms, regularizer=regularizer)}
