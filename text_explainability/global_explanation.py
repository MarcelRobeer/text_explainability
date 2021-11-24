"""Global explanations explain the whole dataset or model behavior on that dataset.

Todo:

    * More support for sampling methods
    * add support for other tasks than classification (e.g. regression, multi-label classification)
    * partial dependence plots? https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection
"""

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
from fastcountvectorizer import FastCountVectorizer
from genbase import Readable, SeedMixin, add_callargs, translate_list
from instancelib import InstanceProvider
from instancelib.instances.text import TextInstance
from instancelib.labels import LabelProvider
from instancelib.machinelearning import AbstractClassifier
from sklearn.feature_selection import mutual_info_classif

from .data.sampling import (KMedoids, LabelwiseKMedoids, LabelwiseMMDCritic,
                            MMDCritic)
from .generation.return_types import FeatureList


class GlobalExplanation(Readable, SeedMixin):
    def __init__(self,
                 provider: InstanceProvider[TextInstance, Any, str, Any, str],
                 seed: int = 0):
        """Generic wrapper from global explanations (explain whole dataset or model).

        Args:
            provider (InstanceProvider[TextInstance, Any, str, Any, str]): Dataset to perform explanation on.
            seed (int, optional): Seed for reproducibility. Defaults to 0.
        """
        super().__init__()
        self.provider = provider
        self._seed = self._original_seed = seed

    def get_data(self) -> InstanceProvider:
        """Easy access to data.

        Returns:
            InstanceProvider: Easily accessible dataset.
        """
        return self.provider

    def predict(self, model: AbstractClassifier) -> Union[Sequence[FrozenSet[str]], np.ndarray]:
        """Apply predict function of model to data.

        Args:
            model (AbstractClassifier): Model to apply predictions with.

        Returns:
            Union[Sequence[FrozenSet[str]], np.ndarray]: Labels for dataset according to model.
        """
        return model.predict(self.get_data())

    def get_instances_labels(self,
                             model: Optional[AbstractClassifier],
                             labelprovider: Optional[LabelProvider],
                             explain_model: bool = True) -> Tuple[InstanceProvider, np.ndarray]:
        """Get corresponding labels of dataset inputs, either from the original data or 
            according to the predict function.

        Args:
            model (Optional[AbstractClassifier]): Model to perform predictions with.
            labelprovider (Optional[LabelProvider]): Ground-truth labels.
            explain_model (bool, optional): Whether to explain using the `model` 
                labels (True) or `labelprovider` labels (False). Defaults to True.

        Raises:
            ValueError: if explain_model = True provide a model, and if False provide a labelprovider.

        Returns:
            Tuple[InstanceProvider, np.ndarray]: Instances and corresponding labels
        """
        if explain_model and model is None:
            raise ValueError('Provide a model to explain its predictions, or set `explain_predictions` to False')
        elif not explain_model and labelprovider is None:
            raise ValueError('Provide a labelprovider to explain ground-truth labels, ',
                             'or set `explain_predictions` to True')

        instances = self.get_data()
        labels = model.predict(instances) if explain_model \
                 else [next(iter(labelprovider.get_labels(k))) for k in instances]
        if len(labels) > 0 and isinstance(labels[0], tuple) and isinstance(labels[0][-1], frozenset):
            labels = ['-'.join(list(x)) for id, x in labels]
        return instances, np.array(labels)

    def explain(self, *args, **kwargs):
        return self(*args, **kwargs)


class TokenFrequency(GlobalExplanation):
    @add_callargs
    def __call__(self,
                 model: Optional[AbstractClassifier] = None,
                 labelprovider: Optional[LabelProvider] = None,
                 explain_model: bool = True,
                 labelwise: bool = True,
                 k: Optional[int] = None,
                 filter_words: List[str] = translate_list('stopwords'),
                 lower: bool = True,
                 **count_vectorizer_kwargs) -> Dict[str, List[Tuple[str, int]]]:
        """Show the top-k number of tokens for each ground-truth or predicted label.

        Args:
            model (Optional[AbstractClassifier], optional): Predictive model to explain. Defaults to None.
            labelprovider (Optional[LabelProvider], optional): Ground-truth labels to explain. Defaults to None.
            explain_model (bool, optional): Whether to explain the model (True) or ground-truth labels (False).
                Defaults to True.
            labelwise (bool, optional): Whether to summarize the counts for each label seperately. Defaults to True.
            k (Optional[int], optional): Limit to the top-k words per label, or all words if None. Defaults to None.
            filter_words (List[str], optional): Words to filter out from top-k. Defaults to ['de', 'het', 'een'].
            lower (bool, optional): Whether to make all tokens lowercase. Defaults to True.
            **count_vectorizer_kwargs: Optional arguments passed to `FastCountVectorizer`.

        Returns:
            Dict[str, List[Tuple[str, int]]]: Each label with corresponding top words and their frequency
        """
        type, subtype = 'global_explanation', 'token_frequency'
        callargs = count_vectorizer_kwargs.pop('__callargs__', None)

        instances, labels = self.get_instances_labels(model, labelprovider, explain_model=explain_model)

        def top_k_counts(instances_to_fit):
            cv = FastCountVectorizer(**count_vectorizer_kwargs)
            counts = cv.fit_transform([str.lower(d) for d in instances_to_fit] if lower else instances_to_fit)
            counts = np.ravel(counts.sum(axis=0))
            return sorted([(w, counts[v]) for w, v in cv.vocabulary_.items() if k not in filter_words],
                            key=lambda x: x[1], reverse=True)[:k]

        if labelwise:  # TO-DO improve beyond classification, e.g. buckets for regression?
            labels = np.unique(labels)
            label_ids = [i for i, _ in enumerate(labels)]

            def counts_by_label(label):
                return zip(*top_k_counts([instances[instances.key_list[idx]].data
                                          for idx in np.where(labels == label)[0]]))

            used_features, scores = zip(*[counts_by_label(label) for label in labels]) 

            return FeatureList(labels=label_ids,
                               labelset=labels,
                               used_features=dict(zip(label_ids, used_features)),
                               scores=dict(zip(label_ids, scores)),
                               type=type,
                               subtype=subtype,
                               callargs=callargs)
        used_features, scores = zip(*top_k_counts(instances.all_data()))
        return FeatureList(used_features=used_features,
                           scores=scores,
                           type=type,
                           subtype=subtype,
                           callargs=callargs)


class TokenInformation(GlobalExplanation):
    @add_callargs
    def __call__(self,
                 model: Optional[AbstractClassifier] = None,
                 labelprovider: Optional[LabelProvider] = None,
                 explain_model: bool = True,
                 # labelwise: bool = True,
                 k: Optional[int] = None,
                 filter_words: List[str] = translate_list('stopwords'),
                 lower: bool = True,
                 **count_vectorizer_kwargs) -> List[Tuple[str, float]]:
        """Show the top-k token mutual information for a dataset or model.

        Args:
            model (Optional[AbstractClassifier], optional): Predictive model to explain. Defaults to None.
            labelprovider (Optional[LabelProvider], optional): Ground-truth labels to explain. Defaults to None.
            explain_model (bool, optional): Whether to explain the model (True) or ground-truth labels (False).
                Defaults to True.
            k (Optional[int], optional): Limit to the top-k words per label, or all words if None. Defaults to None.
            filter_words (List[str], optional): Words to filter out from top-k. Defaults to ['de', 'het', 'een'].
            lower (bool, optional): Whether to make all tokens lowercase. Defaults to True.
            **count_vectorizer_kwargs: Keyword arguments to pass onto `FastCountVectorizer`.

        Returns:
            List[Tuple[str, float]]: k labels, sorted based on their mutual information with 
                the output (predictive model labels or ground-truth labels)
        """
        callargs = count_vectorizer_kwargs.pop('__callargs__', None)

        instances, labels = self.get_instances_labels(model, labelprovider, explain_model=explain_model)

        cv = FastCountVectorizer(**count_vectorizer_kwargs)
        counts = cv.fit_transform([str.lower(d) for d in instances.all_data()] if lower else instances.all_data())

        # TO-DO improve beyond classification
        # see https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
        # #sklearn.feature_selection.mutual_info_regression
        mif = mutual_info_classif(counts, labels, discrete_features=True, random_state=self.seed)
        feature_names = cv.get_feature_names()
        res = list(map(tuple, zip(feature_names, mif)))
        res_sorted = list(sorted([(w, v) for w, v in res if w not in filter_words],
                                 key=lambda x: x[1], reverse=True))[:k]
        used_features, scores = zip(*res_sorted)
        return FeatureList(used_features=used_features,
                           scores=scores,
                           type='global_explanation',
                           subtype='token_information',
                           method='mutual_information',
                           callargs=callargs)


__all__ = [GlobalExplanation, TokenFrequency, TokenInformation,
           KMedoids, MMDCritic, LabelwiseKMedoids, LabelwiseMMDCritic]
