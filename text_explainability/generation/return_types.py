"""General return types for global/local explanations.

Todo:

    * Add rule-based explanations
    * Add named label support
    * Test for bugs
"""

import copy
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from genbase import MetaInfo
from instancelib import InstanceProvider
from instancelib.typehints import LT

from ..ui.notebook import Render
from .surrogate import RuleSurrogate, TreeSurrogate


class BaseReturnType(MetaInfo):
    def __init__(self,
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None,
                 type: Optional[str] = 'base',
                 subtype: Optional[str] = None,
                 callargs: Optional[dict] = None,
                 **kwargs):
        """Base return type.

        Args:
            labels (Optional[Sequence[int]], optional): Label indices to include, if none provided 
                defaults to 'all'. Defaults to None.
            labelset (Optional[Sequence[str]], optional): Lookup for label names. Defaults to None.
            type (Optional[str]): Type description. Defaults to 'base'.
            subtype (Optional[str], optional): Subtype description. Defaults to None.
            callargs (Optional[dict], optional): Call arguments for reproducibility. Defaults to None.
            **kwargs: Optional meta descriptors.
        """
        renderer = kwargs.pop('renderer', Render)
        super().__init__(type=type, subtype=subtype, callargs=callargs, renderer=renderer, **kwargs)
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
        if hasattr(self, 'used_features'):
            return f'{self.__class__.__name__}(labels={labels}, used_features={self.used_features})'
        return f'{self.__class__.__name__}(label={labels})'


class UsedFeaturesMixin:
    @property
    def used_features(self):
        """Get used features property."""
        return self._used_features


class FeatureList(BaseReturnType, UsedFeaturesMixin):
    def __init__(self,
                 used_features: Union[Sequence[str], Sequence[int]],
                 scores: Union[Sequence[int], Sequence[float]],
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None,
                 type: Optional[str] = 'global_explanation',
                 subtype: Optional[str] = 'feature_list',
                 callargs: Optional[dict] = None,
                 **kwargs):
        """Save scores per feature, grouped per label.

        Examples of scores are feature importance scores, or counts of features in a dataset.

        Args:
            used_features (Union[Sequence[str], Sequence[int]]): Used features per label.
            scores (Union[Sequence[int], Sequence[float]]): Scores per label.
            labels (Optional[Sequence[int]], optional): Label indices to include, if none provided 
                defaults to 'all'. Defaults to None.
            labelset (Optional[Sequence[str]], optional): Lookup for label names. Defaults to None.
            type (Optional[str]): Type description. Defaults to 'explanation'.
            subtype (Optional[str], optional): Subtype description. Defaults to 'feature_list'.
            callargs (Optional[dict], optional): Call arguments for reproducibility. Defaults to None.
            **kwargs: Optional meta descriptors.
        """
        super().__init__(labels=labels,
                         labelset=labelset,
                         type=type,
                         subtype=subtype,
                         callargs=callargs,
                         **kwargs)
        self._used_features = copy.deepcopy(used_features)
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
        # TODO: change to IDs
        all_scores = self.get_raw_scores(normalize=normalize)
        if self.labels is None:
            return {'all': [(feature, score_)
                    for feature, score_ in zip(self.used_features, all_scores)]}
        if isinstance(self.used_features, dict):
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

    @property
    def content(self):
        return self.scores

    def __repr__(self) -> str:
        return '\n'.join([f'{a}: {str(b)}' for a, b in self.scores.items()])


class LocalDataExplanation:
    def __init__(self,
                 provider: InstanceProvider,
                 original_id: Optional[LT] = None,
                 sampled: bool = False):
        """Save the sampled/generated instances used to determine an explanation.

        Args:
            provider (InstanceProvider): Sampled or generated data, including original instance.
            original_id (Optional[LT], optional): ID of original instance; picks first if None. Defaults to None.
            sampled (bool, optional): Whether the data in the provider was sampled (True) or generated (False). 
                Defaults to False.
        """
        self._provider = provider
        original_id = next(iter(self._provider)) if original_id is None else original_id
        self._original_instance = copy.deepcopy(self._provider[original_id])
        self._neighborhood_instances = copy.deepcopy(self._provider.get_children(self._original_instance))
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
            if isinstance(self._used_features, dict):
                return {k: [self.original_instance.tokenized[i] for i in v] for k, v in self._used_features.items()}
            return [self.original_instance.tokenized[i] for i in self._used_features]
        return list(self._used_features)

    def __repr__(self) -> str:
        sampled_or_perturbed = 'sampled' if self.sampled else 'perturbed'
        n = sum(1 for _ in self.neighborhood_instances)
        labels = [self.label_by_index(label) for label in self.labels] if self.labels is not None else None
        return f'{self.__class__.__name__}(labels={labels}, ' + \
            f'used_features={self.used_features}, n_{sampled_or_perturbed}_instances={n})'


class FeatureAttribution(ReadableDataMixin, FeatureList, LocalDataExplanation):
    def __init__(self,
                 provider: InstanceProvider,
                 scores: Sequence[float],
                 used_features: Optional[Union[Sequence[str], Sequence[int]]] = None,
                 scores_stddev: Sequence[float] = None,
                 base_score: float = None,
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None,
                 original_id: Optional[LT] = None,
                 sampled: bool = False,
                 type: Optional[str] = 'local_explanation',
                 subtype: Optional[str] = 'feature_attribution',
                 callargs: Optional[dict] = None,
                 **kwargs):
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
            original_id (Optional[LT], optional): ID of original instance; picks first if None. Defaults to None.
            sampled (bool, optional): Whether the data in the provider was sampled (True) or generated (False). 
                Defaults to False.
            type (Optional[str]): Type description. Defaults to 'base'.
            subtype (Optional[str], optional): Subtype description. Defaults to None.
            callargs (Optional[dict], optional): Call arguments for reproducibility. Defaults to None.
            **kwargs: Optional meta descriptors.
        """
        LocalDataExplanation.__init__(self,
                                      provider=provider,
                                      original_id=original_id,
                                      sampled=sampled)
        if used_features is None:
            used_features = list(range(len(self.original_instance.tokenized)))
        FeatureList.__init__(self,
                             used_features=used_features,
                             scores=scores,
                             labels=labels,
                             labelset=labelset,
                             type=type,
                             subtype=subtype,
                             callargs=callargs,
                             **kwargs)
        self._base_score = base_score
        self._scores_stddev = scores_stddev

    @property
    def scores(self):
        """Saved feature attribution scores."""
        return self.get_scores(normalize=False)

    @property
    def content(self):
        return {'features': list(self.original_instance.tokenized),
                'scores': self.scores}


class Rules(ReadableDataMixin, UsedFeaturesMixin, BaseReturnType, LocalDataExplanation):
    def __init__(self,
                 provider: InstanceProvider,
                 rules: Union[Sequence[str], TreeSurrogate, RuleSurrogate],
                 used_features: Optional[Union[Sequence[str], Sequence[int]]] = None,
                 labels: Optional[Sequence[int]] = None,
                 labelset: Optional[Sequence[str]] = None,
                 original_id: Optional[LT] = None,
                 sampled: bool = False,
                 type: Optional[str] = 'local_explanation',
                 subtype: Optional[str] = 'rules',
                 callargs: Optional[dict] = None,
                 **kwargs):
        """Rule-based return type.

        Args:
            provider (InstanceProvider): Sampled or generated data, including original instance.
            rules (Union[Sequence[str], TreeSurrogate, RuleSurrogate]): Rules applicable.
            used_features (Optional[Union[Sequence[str], Sequence[int]]]): Used features per label. Defaults to None.
            labels (Optional[Sequence[int]], optional): Label indices to include, if none provided 
                defaults to 'all'. Defaults to None.
            labelset (Optional[Sequence[str]], optional): Lookup for label names. Defaults to None.
            original_id (Optional[LT], optional): ID of original instance; picks first if None. Defaults to None.
            sampled (bool, optional): Whether the data in the provider was sampled (True) or generated (False). 
                Defaults to False.
            type (Optional[str]): Type description. Defaults to 'base'.
            subtype (Optional[str], optional): Subtype description. Defaults to None.
            callargs (Optional[dict], optional): Call arguments for reproducibility. Defaults to None.
            **kwargs: Optional meta descriptors.
        """
        LocalDataExplanation.__init__(self,
                                      provider=provider,
                                      original_id=original_id,
                                      sampled=sampled)
        BaseReturnType.__init__(self,
                                labels=labels,
                                labelset=labelset,
                                type=type,
                                subtype=subtype,
                                callargs=callargs,
                                **kwargs)
        if used_features is None:
            used_features = list(range(len(self.original_instance.tokenized)))
        self._used_features = copy.deepcopy(used_features)
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

    @property
    def content(self):
        return self.rules


class Instances(BaseReturnType):
    def __init__(self,
                 instances,
                 type: Optional[str] = 'global_explanation',
                 subtype: Optional[str] = 'prototypes',
                 callargs: Optional[dict] = None,
                 **kwargs):
        super().__init__(labels=None,
                         labelset=None,
                         type=type,
                         subtype=subtype,
                         callargs=callargs,
                         **kwargs)
        self.instances = instances

    @property
    def content(self):
        return self.instances if isinstance(self.instances, dict) else {'instances': self.instances}
