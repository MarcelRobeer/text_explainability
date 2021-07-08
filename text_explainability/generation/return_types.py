"""General return types for ...

Todo:
    * add rule-based explanations
    * add named label support
"""

import numpy as np

from typing import Union, Optional, Sequence
from instancelib import InstanceProvider


class FeatureList:
    def __init__(self,
                 used_features: Union[Sequence[str], Sequence[int]],
                 scores: Union[Sequence[int], Sequence[float]],
                 labels: Optional[Sequence[int]] = None,
                 label_names: Optional[Sequence[str]] = None):
        self._used_features = used_features
        self._label_names = label_names
        self._labels = labels
        self._scores = np.array(scores)

    @property
    def labels(self):
        if self._labels is None:
            return self._labels
        return list(self._labels)

    @property
    def label_names(self):
        return self._label_names

    @property
    def used_features(self):
        return self._used_features

    def label_by_index(self, idx):
        if self.label_names is not None:
            return self.label_names[idx]
        return idx

    def get_raw_scores(self, normalize: bool = False) -> np.ndarray:
        scores = np.array(self._scores) if not isinstance(self._scores, np.ndarray) else self._scores

        if normalize:
            return scores / scores.sum(axis=1)[:, np.newaxis]
        return scores

    def get_scores(self, normalize: bool = False):
        all_scores = self.get_raw_scores(normalize=normalize)
        if self.labels is None:
            return {'all': [(feature, score_)
                    for feature, score_ in zip(self.used_features, all_scores)]}
        return {self.label_by_index(label): [(feature, score_)
                for feature, score_ in zip(self.used_features, all_scores[label])]
                for label in self.labels}

    @property
    def scores(self):
        return self.get_scores(normalize=False)

    def __repr__(self) -> str:
        labels = [self.label_by_index(l) for l in self.labels] if self.labels is not None else None
        return f'{self.__class__.__name__}(labels={labels}, used_features={self.used_features})'


class FeatureAttribution(FeatureList):
    def __init__(self,
                 provider: InstanceProvider,
                 used_features: Union[Sequence[str], Sequence[int]],
                 scores: Sequence[float],
                 scores_stddev: Sequence[float] = None,
                 base_score: float = None,
                 labels: Optional[Sequence[int]] = None,
                 label_names: Optional[Sequence[str]] = None,
                 sampled: bool = False):
        """[summary]

        Args:
            provider (InstanceProvider): Sampled or generated data, including original instance.
            used_features (Union[Sequence[str], Sequence[int]]): Which features were selected for the explanation.
            scores (Sequence[float]): Scores corresponding to the selected features.
            scores_stddev (Sequence[float], optional): Standard deviation of each feature attribution score. 
                Defaults to None.
            base_score (float, optional): Base score, to which all scores are relative. Defaults to None.
            labels (Optional[Sequence[int]], optional): Labels for outputs (e.g. classes). Defaults to None.
            label_names (Optional[Sequence[str]], optional): Label names corresponding to labels. Defaults to None.
            sampled (bool, optional): Whether the data in the provider was sampled (True) or generated (False). 
                Defaults to False.
        """
        super().__init__(used_features=used_features,
                         scores=scores,
                         labels=labels,
                         label_names=label_names)
        self._provider = provider
        self._base_score = base_score
        self._scores_stddev = scores_stddev
        self._original_instance = self._provider[next(iter(self._provider))]
        self._sampled_instances = self._provider.get_children(self._original_instance) if sampled else None
        self._perturbed_instances = None if sampled else self._provider.get_children(self._original_instance) 

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
    def used_features(self):
        if hasattr(self.original_instance, 'tokenized'):
            return [self.original_instance.tokenized[i] for i in self._used_features]
        return list(self._used_features)

    @property
    def scores(self):
        return self.get_scores(normalize=True)

    def __str__(self) -> str:
        return '\n'.join([f'{a}: {str(b)}' for a, b in self.scores.items()])

    def __repr__(self) -> str:
        sampled_or_perturbed = 'sampled' if self.sampled_instances is not None else 'perturbed'
        n = sum(1 for _ in self.neighborhood_instances)
        labels = [self.label_by_index(l) for l in self.labels] if self.labels is not None else None
        return f'{self.__class__.__name__}(labels={labels}, ' + \
            'used_features={self.used_features}, n_{sampled_or_perturbed}_instances={n})'
