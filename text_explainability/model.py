import numpy as np
import itertools

from typing import (Optional, Tuple, FrozenSet, Sequence, Callable, List, Union)

from instancelib.utils import SaveableInnerModel
from instancelib.instances import Instance, InstanceProvider

from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError


InstanceType = Union[str, Sequence[str], Instance, InstanceProvider]


class SklearnModel(SaveableInnerModel):
    _name = "Sklearn"

    def __init__(self,
                 estimator,
                 storage_location: Optional[str] = None,
                 filename: Optional[str] = None):
        # TO-DO what if tokenizer, transformer and predictor are separate parts?
        if isinstance(estimator, (list, tuple)):
            estimator = Pipeline([(e.__class__.__name__, e) for e in estimator])
        SaveableInnerModel.__init__(self, estimator, storage_location, filename)
        assert self.fitted, 'You can only explain models after they have been fitted'

    @property
    def estimator_type(self) -> str:
        return getattr(self.innermodel, '_estimator_type', None)

    @property
    def name(self) -> str:
        return f"{self._name} :: {self.innermodel.__class__}"
    
    @property
    def fitted(self) -> bool:
        try:
            self.innermodel.predict(None)
        except NotFittedError:
            return False
        except:
            pass
        return True

    @property
    def labels(self) -> List[str]:
        assert self.estimator_type in ['classifier', 'clusterer']
        if self.estimator_type == 'classifier':
            return self.innermodel.classes_.tolist()
        else:
            n_clusters = self.innermodel.get_params()[f'{self.innermodel.steps[-1][0]}__n_clusters']
            return [f'cluster_{i}' for i in range(n_clusters)]
    
    def __predict_fn(self, instances: InstanceType, inner_fn: Callable, *args, **kwargs):
        if isinstance(instances, Instance):
            instances = instances.data
        if isinstance(instances, str):
            instances = [instances]
        if isinstance(instances, InstanceProvider):
            results = instances.vectorized_data_map(inner_fn)
            chained = list(itertools.chain.from_iterable(results))
            return chained
        if isinstance(instances, list) and len(instances) > 0 and isinstance(instances[0], Instance):
            instances = [i.data for i in instances]
        return inner_fn(instances, *args, **kwargs)

    def predict_proba(self, instances: InstanceType) -> np.ndarray:
        # Q: If model only has .predict() perhaps transform that to a one-hot score?
        assert self.innermodel is not None and self.fitted
        return self.__predict_fn(instances, self.innermodel.predict_proba)
    
    def predict(self,
                instances: InstanceType,
                return_labels: bool = True) -> Union[Sequence[FrozenSet[str]], np.ndarray]:
        # Q: Should we include the labels for `clusterer`?
        assert self.innermodel is not None and self.fitted
        y_pred = self.__predict_fn(instances, self.innermodel.predict)
        return y_pred if self.estimator_type != 'classifier' or not return_labels \
             else [frozenset([y]) for y in y_pred]

    def predict_proba_instances(self, instances: InstanceType) -> Sequence[FrozenSet[Tuple[str, float]]]:
        y_pred = self.predict_proba(instances).tolist()
        y_labels: List[FrozenSet[Tuple[str, float]]] = [
            frozenset(zip(self.labels, y_vec)) # type: ignore
            for y_vec in y_pred
        ]
        return y_labels

    def __inner_call(self, instances: InstanceType, return_labels: bool = True):
        # Q: Should we also include the score() function for `clusterer`?
        if self.estimator_type == 'classifier':
            if return_labels:
                return self.predict_proba_instances(instances)
            return self.predict_proba(instances)
        return self.predict(instances)
    
    def __call__(self,
                 instances: InstanceType,
                 batch_size=None,
                 return_labels=True) -> Union[FrozenSet[Tuple[str, float]], Sequence[FrozenSet[str]], np.ndarray]:
        if (isinstance(batch_size, int) and batch_size > 0
            and isinstance(instances, InstanceProvider)
            and batch_size < len(instances)
           ):
            # Split up in batches, and combine the predictions based on their shape
            batches = instances.data_chunker(batch_size)
            ret = [self.__inner_call(batch, return_labels=return_labels) for batch in batches]
            if isinstance(ret[0], np.ndarray):
                if ret[0].ndim > 1:
                    return np.vstack(ret)
                return np.hstack(ret)
            return list(itertools.chain(*ret))
        else:
            return self.__inner_call(instances, return_labels=return_labels)
