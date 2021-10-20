"""General return types for global/local explanations.

Todo:

    * Add rule-based explanations
    * Add named label support
    * Test for bugs
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from instancelib import InstanceProvider

from .surrogate import RuleSurrogate, TreeSurrogate


class BaseReturnType:
    def __init__(self,
                 used_features: Union[Sequence[str], Sequence[int]],
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None):
        """Base return type.

        Args:
            used_features (Union[Sequence[str], Sequence[int]]): Used features per label.
            labels (Optional[Sequence[int]], optional): Label indices to include, if none provided 
                defaults to 'all'. Defaults to None.
            labelset (Optional[Sequence[str]], optional): Lookup for label names. Defaults to None.
        """
        self._used_features = used_features
        self._labels = labels
        self._labelset = labelset

    @property
    def labels(self):
        """Get labels property."""
        if self._labels is None:
            return self._labels
        return list(self._labels)

    @property
    def labelset(self):
        """Get label names property."""
        return self._labelset

    @property
    def used_features(self):
        """Get used features property."""
        return self._used_features

    def label_by_index(self, idx: int) -> Union[str, int]:
        """Access label name by index, if `labelset` is set.

        Args:
            idx (int): Lookup index.

        Raises:
            IndexError: `labelset` is set but the element index is
                not in `labelset` (index out of bounds).

        Returns:
            Union[str, int]: Label name (if available) else index.
        """
        if self.labelset is not None:
            return self.labelset[idx]
        return idx

    def __repr__(self) -> str:
        labels = [self.label_by_index(label) for label in self.labels] if self.labels is not None else None
        return f'{self.__class__.__name__}(labels={labels}, used_features={self.used_features})'


class FeatureList(BaseReturnType):
    def __init__(self,
                 used_features: Union[Sequence[str], Sequence[int]],
                 scores: Union[Sequence[int], Sequence[float]],
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None):
        """Save scores per feature, grouped per label.

        Examples of scores are feature importance scores, or counts of features in a dataset.

        Args:
            used_features (Union[Sequence[str], Sequence[int]]): Used features per label.
            scores (Union[Sequence[int], Sequence[float]]): Scores per label.
            labels (Optional[Sequence[int]], optional): Label indices to include, if none provided 
                defaults to 'all'. Defaults to None.
            labelset (Optional[Sequence[str]], optional): Lookup for label names. Defaults to None.
        """
        super().__init__(used_features=used_features,
                         labels=labels,
                         labelset=labelset)
        self._scores = scores

    def get_raw_scores(self, normalize: bool = False) -> np.ndarray:
        """Get saved scores per label as `np.ndarray`.

        Args:
            normalize (bool, optional): Normalize scores (ensure they sum to one). Defaults to False.

        Returns:
            np.ndarray: Scores.
        """
        def feature_scores(scores):
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
            if normalize:
                return scores / scores.sum(axis=0)
            return scores

        if isinstance(self._scores, dict):
            return {k: feature_scores(v) for k, v in self._scores.items()}
        return feature_scores(self._scores)

    def get_scores(self, normalize: bool = False) -> Dict[Union[str, int], Tuple[Union[str, int], Union[float, int]]]:
        """Get scores per label.

        Args:
            normalize (bool, optional): Whether to normalize the scores (sum to one). Defaults to False.

        Returns:
            Dict[Union[str, int], Tuple[Union[str, int], Union[float, int]]]: Scores per label, if no `labelset`
                is not set, defaults to 'all'
        """
        all_scores = self.get_raw_scores(normalize=normalize)
        if self.labels is None:
            return {'all': [(feature, score_)
                    for feature, score_ in zip(self.used_features, all_scores)]}
        elif isinstance(self.used_features, dict):
            return {self.label_by_index(label): [(feature, score_)
                    for feature, score_ in zip(self.used_features[label], all_scores[i])]
                    for i, label in enumerate(self.labels)}
        return {self.label_by_index(label): [(feature, score_)
                for feature, score_ in zip(self.used_features, all_scores[i])]
                for i, label in enumerate(self.labels)}

    @property
    def scores(self):
        """Saved scores (e.g. feature importance)."""
        return self.get_scores(normalize=False)

    def __str__(self) -> str:
        return '\n'.join([f'{a}: {str(b)}' for a, b in self.scores.items()])


class DataExplanation:
    def __init__(self,
                 provider: InstanceProvider,
                 sampled: bool = False):
        """Save the sampled/generated instances used to determine an explanation.

        Args:
            provider (InstanceProvider): Sampled or generated data, including original instance.
            sampled (bool, optional): Whether the data in the provider was sampled (True) or generated (False). 
                Defaults to False.
        """
        self._provider = provider
        self._original_instance = self._provider[next(iter(self._provider))]
        self._neighborhood_instances = self._provider.get_children(self._original_instance)
        self.sampled = sampled

    @property
    def original_instance(self):
        """The instance for which the feature attribution scores were calculated."""
        return self._original_instance

    @property
    def perturbed_instances(self):
        """Perturbed versions of the original instance, if `sampled=False` during initialization."""
        return None if self.sampled else self._neighborhood_instances

    @property
    def sampled_instances(self):
        """Sampled instances, if `sampled=True` during initialization."""
        return self._neighborhood_instances if self.sampled else None

    @property
    def neighborhood_instances(self):
        """Instances in the neighborhood (either sampled or perturbed)."""
        return self._neighborhood_instances


class ReadableDataMixin:
    @property
    def used_features(self):
        """Names of features of the original instance."""
        if hasattr(self.original_instance, 'tokenized'):
            return [self.original_instance.tokenized[i] for i in self._used_features]
        return list(self._used_features)

    def __repr__(self) -> str:
        sampled_or_perturbed = 'sampled' if self.sampled else 'perturbed'
        n = sum(1 for _ in self.neighborhood_instances)
        labels = [self.label_by_index(label) for label in self.labels] if self.labels is not None else None
        return f'{self.__class__.__name__}(labels={labels}, ' + \
            f'used_features={self.used_features}, n_{sampled_or_perturbed}_instances={n})'


class FeatureAttribution(ReadableDataMixin, FeatureList, DataExplanation):
    def __init__(self,
                 provider: InstanceProvider,
                 scores: Sequence[float],
                 used_features: Optional[Union[Sequence[str], Sequence[int]]] = None,
                 scores_stddev: Sequence[float] = None,
                 base_score: float = None,
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None,
                 sampled: bool = False):
        """Create a `FeatureList` with additional information saved.

        The additional information contains the possibility to add standard deviations, 
        base scores, and the sampled or generated instances used to calculate these scores.

        Args:
            provider (InstanceProvider): Sampled or generated data, including original instance.
            scores (Sequence[float]): Scores corresponding to the selected features.
            used_features (Optional[Union[Sequence[str], Sequence[int]]]): Selected features for the explanation label. 
                Defaults to None.
            scores_stddev (Sequence[float], optional): Standard deviation of each feature attribution score. 
                Defaults to None.
            base_score (float, optional): Base score, to which all scores are relative. Defaults to None.
            labels (Optional[Sequence[int]], optional): Labels for outputs (e.g. classes). Defaults to None.
            labelset (Optional[Sequence[str]], optional): Label names corresponding to labels. Defaults to None.
            sampled (bool, optional): Whether the data in the provider was sampled (True) or generated (False). 
                Defaults to False.
        """
        DataExplanation.__init__(self,
                                 provider=provider,
                                 sampled=sampled)
        if used_features is None:
            used_features = list(range(len(self.original_instance.tokenized)))
        FeatureList.__init__(self,
                             used_features=used_features,
                             scores=scores,
                             labels=labels,
                             labelset=labelset)
        self._base_score = base_score
        self._scores_stddev = scores_stddev

    @property
    def scores(self):
        """Saved feature attribution scores."""
        return self.get_scores(normalize=False)


class Rules(ReadableDataMixin, BaseReturnType, DataExplanation):
    def __init__(self,
                 provider: InstanceProvider,
                 rules: Union[Sequence[str], TreeSurrogate, RuleSurrogate],
                 used_features: Optional[Union[Sequence[str], Sequence[int]]] = None,
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None,
                 sampled: bool = False):
        """Base return type.

        Args:
            provider (InstanceProvider): Sampled or generated data, including original instance.
            rules (Union[Sequence[str], TreeSurrogate, RuleSurrogate]): Rules applicable.
            used_features (Optional[Union[Sequence[str], Sequence[int]]]): Used features per label. Defaults to None.
            labels (Optional[Sequence[int]], optional): Label indices to include, if none provided 
                defaults to 'all'. Defaults to None.
            labelset (Optional[Sequence[str]], optional): Lookup for label names. Defaults to None.
            sampled (bool, optional): Whether the data in the provider was sampled (True) or generated (False). 
                Defaults to False.
        """
        DataExplanation.__init__(self,
                                 provider=provider,
                                 sampled=sampled)
        if used_features is None:
            used_features = list(range(len(self.original_instance.tokenized)))
        BaseReturnType.__init__(self,
                                used_features=used_features,
                                labels=labels,
                                labelset=labelset)
        self._rules = self._extract_rules(rules)

    def _extract_rules(self, rules: Union[Sequence[str], TreeSurrogate, RuleSurrogate]):
        if isinstance(rules, (TreeSurrogate, RuleSurrogate)):
            from skrules.rule import replace_feature_name
            from skrules.skope_rules import BASE_FEATURE_NAME
            feature_dict = {BASE_FEATURE_NAME + str(i): feat
                            for i, feat in enumerate(self.used_features)}
            return [(replace_feature_name(rule, feature_dict), perf)
                     for rule, perf in rules.rules]
        print(rules)
        raise NotImplementedError('TODO: Support lists of rules')

    @property
    def rules(self):
        return self._rules
