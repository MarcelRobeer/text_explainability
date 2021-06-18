"""TO-DO
- add support for other tasks than classification (e.g. regression, multi-label classification)
- partial dependence plots? https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection
"""

from instancelib import TextInstance, InstanceProvider
import numpy as np

from typing import (Callable, Optional, Text, List, Dict, Tuple, Any)
from instancelib import TextEnvironment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

from text_explainability.utils import default_detokenizer, default_tokenizer
from text_explainability.default import Readable


class GlobalExplanation(Readable):
    def __init__(self,
                 provider: InstanceProvider[TextInstance, Any, str, Any, str],
                 seed: int = 0):
        super().__init__()
        self.provider = provider
        self._seed = 0

    def get_data(self):
        return self.provider

    def predict(self, model):
        return model.predict(self.get_data())

    def get_instances_labels(self, model: Optional[Any], labelprovider, explain_model: bool = True):
        if explain_model:
            assert model is not None, 'Provide a model to explain its predictions, or set `explain_predictions` to False'
        else:
            assert labelprovider is not None, 'Provide a labelprovider to explain ground-truth labels, or set `explain_predictions` to True'

        instances = self.get_data()
        labels = model.predict(instances, return_labels=False) if explain_model \
                 else [next(iter(labelprovider.get_labels(k))) for k in instances]

        return instances, np.array(labels)


class TokenFrequency(GlobalExplanation):
    def __call__(self,
                 model=None,
                 labelprovider=None,
                 explain_model: bool = True,
                 labelwise: bool = True,
                 k: Optional[int] = None,
                 filter_words: List[str] = ['de', 'het', 'een'],
                 tokenizer: Callable = default_tokenizer,
                 **count_vectorizer_kwargs) -> Dict[str, List[Tuple[str, int]]]:
        """Show the top-k number of tokens for each ground-truth or predicted label.

        Args:
            model ([type], optional): Predictive model to explain. Defaults to None.
            labelprovider ([type], optional): Ground-truth labels to explain. Defaults to None.
            explain_model (bool, optional): Whether to explain the model (True) or ground-truth labels (False). Defaults to True.
            labelwise (bool, optional): Whether to summarize the counts for each label seperately. Defaults to True.
            k (Optional[int], optional): Limit to the top-k words per label, or all words if None. Defaults to None.
            filter_words (List[str], optional): Words to filter out from top-k. Defaults to ['de', 'het', 'een'].
            tokenizer (Callable, optional): [description]. Defaults to default_tokenizer.

        Returns:
            Dict[str, List[Tuple[str, int]]]: Each label with corresponding top words and their frequency
        """
        instances, labels = self.get_instances_labels(model, labelprovider, explain_model=explain_model)

        def top_k_counts(instances_to_fit):
            cv = CountVectorizer(tokenizer=tokenizer,
                                 stop_words=filter_words,
                                 max_features=k,
                                 **count_vectorizer_kwargs)
            counts = cv.fit_transform(instances_to_fit)
            counts = np.array(counts.sum(axis=0)).reshape(-1)
            return sorted(((k_, counts[v_]) for k_, v_ in
                            cv.vocabulary_.items()), key=lambda x: x[1], reverse=True)

        if labelwise:  # TO-DO improve beyond classification, e.g. buckets for regression?
            return {label: top_k_counts([instances[idx].data for idx in np.where(labels == label)[0]])
                    for label in np.unique(labels)}
        return {'all': top_k_counts(instances)}


class TokenInformation(GlobalExplanation):
    def __call__(self,
                 model=None,
                 labelprovider=None,
                 explain_model: bool = True,
                 labelwise: bool = True,
                 k: Optional[int] = None,
                 filter_words: List[str] = ['de', 'het', 'een'],
                 tokenizer: Callable = default_tokenizer,
                 **count_vectorizer_kwargs) -> List[Tuple[str, float]]:
        """Show the top-k token mutual information for a dataset or model.

        Args:
            model ([type], optional): Predictive model to explain. Defaults to None.
            labelprovider ([type], optional): Ground-truth labels to explain. Defaults to None.
            explain_model (bool, optional): Whether to explain the model (True) or ground-truth labels (False). Defaults to True.
            k (Optional[int], optional): Limit to the top-k words per label, or all words if None. Defaults to None.
            filter_words (List[str], optional): Words to filter out from top-k. Defaults to ['de', 'het', 'een'].
            tokenizer (Callable, optional): [description]. Defaults to default_tokenizer.

        Returns:
            List[Tuple[str, float]]: k labels, sorted based on their mutual information with 
                the output (predictive model labels or ground-truth labels)
        """
        instances, labels = self.get_instances_labels(model, labelprovider, explain_model=explain_model)

        cv = CountVectorizer(tokenizer=tokenizer,
                             stop_words=filter_words,
                             #max_features=k, # ??
                             **count_vectorizer_kwargs)
        counts = cv.fit_transform(instances.all_data())

        # TO-DO improve beyond classification
        # see https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression
        res = list(map(tuple, zip(cv.get_feature_names(), mutual_info_classif(counts, labels, discrete_features=True, random_state=self._seed))))
        return list(sorted(res, key=lambda x: x[1], reverse=True))[:k]

