"""Global explanations explain the whole dataset or model behavior on that dataset.

Todo:
    * add support for other tasks than classification (e.g. regression, multi-label classification)
    * partial dependence plots? https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection
"""

from instancelib import InstanceProvider
import numpy as np

from typing import (Callable, Optional, List, Dict, Tuple, Any, Sequence, FrozenSet, Union)
from instancelib.instances.text import TextInstance
from instancelib.machinelearning import AbstractClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

from text_explainability.utils import default_tokenizer
from text_explainability.default import Readable
from text_explainability.generation.return_types import FeatureList
from text_explainability.internationalization import translate_list


class GlobalExplanation(Readable):
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
        self._seed = 0

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
                             model: Optional[Any],
                             labelprovider,
                             explain_model: bool = True) -> Tuple[InstanceProvider, np.ndarray]:
        """Get corresponding labels of dataset inputs, either from the original data or 
            according to the predict function.

        Args:
            model (Optional[Any]): Model to perform predictions with.
            labelprovider ([type]): Ground-truth labels.
            explain_model (bool, optional): Whether to explain using the `model` 
                labels (True) or `labelprovider` labels (False). Defaults to True.

        Returns:
            Tuple[InstanceProvider, np.ndarray]: Instances and corresponding labels
        """
        if explain_model:
            assert model is not None, \
                'Provide a model to explain its predictions, or set `explain_predictions` to False'
        else:
            assert labelprovider is not None, \
                'Provide a labelprovider to explain ground-truth labels, or set `explain_predictions` to True'

        instances = self.get_data()
        labels = model.predict(instances) if explain_model \
                 else [next(iter(labelprovider.get_labels(k))) for k in instances]
        if len(labels) > 0 and isinstance(labels[0], tuple) and isinstance(labels[0][-1], frozenset):
            labels = ['-'.join(list(x)) for id, x in labels]
        return instances, np.array(labels)


class TokenFrequency(GlobalExplanation):
    def __call__(self,
                 model=None,
                 labelprovider=None,
                 explain_model: bool = True,
                 labelwise: bool = True,
                 k: Optional[int] = None,
                 filter_words: List[str] = translate_list('stopwords'),
                 tokenizer: Callable = default_tokenizer,
                 **count_vectorizer_kwargs) -> Dict[str, List[Tuple[str, int]]]:
        """Show the top-k number of tokens for each ground-truth or predicted label.

        Args:
            model ([type], optional): Predictive model to explain. Defaults to None.
            labelprovider ([type], optional): Ground-truth labels to explain. Defaults to None.
            explain_model (bool, optional): Whether to explain the model (True) or ground-truth labels (False).
                Defaults to True.
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
            return {label: top_k_counts([instances[instances.key_list[idx]].data
                                         for idx in np.where(labels == label)[0]])
                    for label in np.unique(labels)}
        return FeatureList('all', top_k_counts(instances.all_data()))


class TokenInformation(GlobalExplanation):
    def __call__(self,
                 model=None,
                 labelprovider=None,
                 explain_model: bool = True,
                 # labelwise: bool = True,
                 k: Optional[int] = None,
                 filter_words: List[str] = translate_list('stopwords'),
                 tokenizer: Callable = default_tokenizer,
                 **count_vectorizer_kwargs) -> List[Tuple[str, float]]:
        """Show the top-k token mutual information for a dataset or model.

        Args:
            model ([type], optional): Predictive model to explain. Defaults to None.
            labelprovider ([type], optional): Ground-truth labels to explain. Defaults to None.
            explain_model (bool, optional): Whether to explain the model (True) or ground-truth labels (False).
                Defaults to True.
            k (Optional[int], optional): Limit to the top-k words per label, or all words if None. Defaults to None.
            filter_words (List[str], optional): Words to filter out from top-k. Defaults to ['de', 'het', 'een'].
            tokenizer (Callable, optional): Function for tokenizing strings. Defaults to default_tokenizer.
            **count_vectorizer_kwargs: Keyword arguments to pass onto `CountVectorizer`.

        Returns:
            List[Tuple[str, float]]: k labels, sorted based on their mutual information with 
                the output (predictive model labels or ground-truth labels)
        """
        instances, labels = self.get_instances_labels(model, labelprovider, explain_model=explain_model)

        cv = CountVectorizer(tokenizer=tokenizer,
                             stop_words=filter_words,
                             **count_vectorizer_kwargs)
        counts = cv.fit_transform(instances.all_data())

        # TO-DO improve beyond classification
        # see https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
        # #sklearn.feature_selection.mutual_info_regression
        mif = mutual_info_classif(counts, labels, discrete_features=True, random_state=self._seed)
        feature_names = cv.get_feature_names()
        res = list(map(tuple, zip(feature_names, mif)))
        res_sorted = list(sorted(res, key=lambda x: x[1], reverse=True))[:k]
        return FeatureList(used_features=[a for a, b in res_sorted],
                           scores=[b for a, b in res_sorted])
