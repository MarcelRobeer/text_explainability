"""Wrap models using instancelib."""

from typing import Union

from instancelib.environment.text import TextEnvironment
from instancelib.instances.text import TextInstanceProvider
from instancelib.machinelearning import AbstractClassifier, SkLearnDataClassifier

from text_explainability.data import train_test_split


def from_sklearn(model,
                 environment: TextEnvironment,
                 train: Union[float, int, TextInstanceProvider] = 0.7) -> AbstractClassifier:
    """Wrap a scikit-learn model.

    Args:
        model ([type]): Model or pipeline from scikit-learn.
        environment (TextEnvironment): Instancelib environment holding instances and corresponding ground-truth labels.
        train (Union[float, int, TextInstanceProvider], optional): Training dataset (TextInstanceProvider), 
            fraction of training dataset size (float, [0, 1]) or instances in training set (int, > 0). Defaults to 0.7.

    Returns:
        AbstractClassifier: Machine learning model able to be used by text_explainability.
    """
    if isinstance(train, (float, int)):
        train, _ = train_test_split(environment, train_size=train)

    # TODO: go beyond classification
    model = SkLearnDataClassifier.build(model, environment)
    model.fit_provider(train, environment.labels)
    return model
