import numpy as np

from typing import (Callable, Optional, Text, List, Dict, Tuple)
from instancelib import TextEnvironment
from sklearn.feature_extraction.text import CountVectorizer

from explainability.utils import default_detokenizer, default_tokenizer


class GlobalExplanation:
    def __init__(self,
                 dataset: TextEnvironment):
        self.dataset = dataset

    def predict(self, model):
        return model.predict(self.dataset.bulk_get_all())


class TokenFrequency(GlobalExplanation):
    def __call__(self,
                 model=None,
                 labelprovider=None,
                 explain_model: bool = True,
                 k: Optional[int] = None,
                 filter_words: List[str] = ['de', 'het', 'een'],
                 tokenizer: Callable = default_tokenizer,
                 **count_vectorizer_kwargs) -> Dict[str, List[Tuple[str, int]]]:
        """Show the top-k number of tokens for each ground-truth or predicted label.

        Args:
            model ([type], optional): Predictive model to explain. Defaults to None.
            labelprovider ([type], optional): Ground-truth labels to explain. Defaults to None.
            explain_model (bool, optional): Whether to explain the model (True) or ground-truth labels (False). Defaults to True.
            k (Optional[int], optional): Limit to the top-k words per label, or all words if None. Defaults to None.
            filter_words (List[str], optional): Words to filter out from top-k. Defaults to ['de', 'het', 'een'].
            tokenizer (Callable, optional): [description]. Defaults to default_tokenizer.

        Returns:
            Dict[str, List[Tuple[str, int]]]: Each label with corresponding top words and their frequency
        """
        if explain_model:
            assert model is not None, 'Provide a model to explain its predictions, or set `explain_predictions` to False'
        else:
            assert labelprovider is not None, 'Provide a labelprovider to explain ground-truth labels, or set `explain_predictions` to True'

        instances = self.dataset.bulk_get_all()
        labels = model.predict(instances, return_labels=False) if explain_model \
                 else [next(iter(labelprovider.get_labels(k))) for k in instances]
        labels = np.array(labels)

        per_label = {}

        for label in np.unique(labels):
            instance_of_label = np.where(labels == label)[0]
            cf = CountVectorizer(tokenizer=tokenizer, stop_words=filter_words, **count_vectorizer_kwargs)
            counts = cf.fit_transform([instances[idx].data for idx in instance_of_label])
            counts = np.array(counts.sum(axis=0)).reshape(-1)

            res = sorted(((k, counts[v]) for k, v in
                           cf.vocabulary_.items()), key=lambda x: x[1], reverse=True)
            
            if k is not None:
                res = res[:k]

            per_label[label] = res

        return per_label
