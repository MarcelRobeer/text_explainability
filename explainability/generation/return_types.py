"""TO-DO:
- add rule-based explanations
- add named label support
"""

import numpy as np

from typing import Optional


class FeatureAttribution:
    def __init__(self, provider, used_features, scores, scores_stddev = None, base_score = None, labels: Optional[int] = None, sampled: bool = False):
        self._provider = provider
        self._used_features = used_features
        self._base_score = base_score
        self._scores = scores
        self._scores_stddev = scores_stddev
        self._labels = labels
        self._original_instance = self._provider[next(iter(self._provider))]
        self._sampled_instances = self._provider.get_children(self._original_instance) if sampled else None
        self._perturbed_instances = None if sampled else self._provider.get_children(self._original_instance) 

    @property
    def used_features(self):
        return self._used_features

    @property
    def original_instance(self):
        return self._original_instance

    @property
    def perturbed_instances(self):
        return self._perturbed_instances

    @property
    def sampled_instances(self):
        return self._sampled_instances

    @property
    def neighborhood_instances(self):
        return self.sampled_instances if self.sampled_instances is not None else self.perturbed_instances

    @property
    def labels(self):
        return list(self._labels)

    @property
    def used_features(self):
        if hasattr(self.original_instance, 'tokenized'):
            return [self.original_instance.tokenized[i] for i in self._used_features]
        return list(self._used_features)

    @property
    def scores(self):
        all_scores = self.get_raw_scores(normalize=True)
        return {label: {feature: score_ 
                for feature, score_ in zip(self.used_features, all_scores[label])}
                for label in self.labels}

    def __str__(self) -> str:
        return self.scores

    def __repr__(self) -> str:
        sampled_or_perturbed = 'sampled' if self.sampled_instances is not None else 'perturbed'
        n = sum(1 for _ in self.neighborhood_instances)
        return f'{self.__class__.__name__}(labels={self.labels}, used_features={self.used_features}, n_{sampled_or_perturbed}_instances={n})'

    def get_raw_scores(self, normalize: bool = False) -> np.ndarray:
        scores = np.array(self._scores) if not isinstance(self._scores, np.ndarray) else self._scores

        if normalize:
            return scores / scores.sum(axis=1)[:, np.newaxis]
        return scores
