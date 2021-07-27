"""Encode targets into binary labels for contrastive explanation."""


import numpy as np

from typing import Optional, Sequence, Union, Generator, List
from instancelib.machinelearning import AbstractClassifier


class TargetEncoder:
    def __init__(self,
                 labels: Optional[Union[Sequence[str], AbstractClassifier]] = None):
        """Encode model predictions based on encoding rule.

        Args:
            labels (Optional[Union[Sequence[str], AbstractClassifier]], optional): Labelset for mapping labels onto.
                Defaults to None.
        """
        self.labelset = labels

    @property
    def labelset(self):
        return self.__labelset

    @labelset.setter
    def labelset(self, labelset):
        if isinstance(labelset, AbstractClassifier):
            labelset = labelset.encoder.labelset
        if isinstance(labelset, frozenset):
            labelset = list(labelset)
        self.__labelset = labelset

    def get_label(self, y,
                  proba_to_labels: bool = True,
                  label_to_index: bool = True) -> Union[List[int], List[str]]:
        """Get prediction label as probability, string or class index.

        Args:
            y: Predictions with optional indices.
            proba_to_labels (bool, optional): Whether to convert probability to highest scoring class. Defaults to True.
            label_to_index (bool, optional): Convert string to index in labelset. Defaults to True.

        Returns:
            Union[List[int], List[str]]: Label names (if label_to_index is False) or label indices (otherwise).
        """
        if isinstance(y, Generator):
            y = list(y)
        if len(y) > 0:
            if isinstance(y[0], tuple):
                y = [y_[1] for y_ in y]
            if isinstance(y[0], frozenset):
                y = [list(y_) for y_ in y]
            if isinstance(y[0], list) and len(y[0]) > 0 and isinstance(y[0][0], str):
                y = [y_[0] for y_ in y]
        if proba_to_labels:
            # model.predict_proba
            if len(y) > 0 and isinstance(y[0], list) and len(y[0]) > 0 and isinstance(y[0][0], tuple):
                y = [sorted(y_, key=lambda x: x[1], reverse=True)[0][0] for y_ in y]

            # model.predict_proba_raw
            if len(y) > 0 and isinstance(y[0], np.ndarray):
                y = np.hstack([np.argmax(y_, axis=1) for y_ in y])
        if label_to_index:
            return [self.labelset.index(y_) if isinstance(y_, str) else y_ for y_ in y]
        return y

    def encode(self, y):
        """Encode a single instance."""
        return y

    def __call__(self, y) -> List[int]:
        """Encode multiple predicted labels.

        Args:
            y: Predictions with optional indices.

        Returns:
            List[int]: Encoded labels as indices.
        """
        return self.encode(self.get_label(y, proba_to_labels=True, label_to_index=True))


class FactFoilEncoder(TargetEncoder):
    def __init__(self, foil: int, labelset: Optional[Sequence[str]] = None):
        super().__init__(labelset)
        self.foil = foil

    @classmethod
    def from_str(cls, label: str, labelset: Union[AbstractClassifier, Sequence[str]]):
        """Instantiate FactFoilEncoder with a string as foil.

        Args:
            label (str): Foil (expected outcome) label.
            labelset (Union[AbstractClassifier, Sequence[str]]): Labelset containing the foil.

        Returns:
            FactFoilEncoder: Initialized FactFoilEncoder.
        """
        if isinstance(labelset, AbstractClassifier):
            labelset = labelset.encoder.labelset
        if isinstance(labelset, frozenset):
            labelset = list(labelset)
        foil = labelset.index(label)
        return cls(foil, labelset)

    def encode(self, y):
        """Encode a single instance into foil (0) or not foil (1)."""
        if isinstance(self.foil, int):
            return [0 if y_ == self.foil else 1 for y_ in y]
        return [0 if y_ in self.foil else 1 for y_ in y]
