"""Data imports, sampling and generation."""

import os
from typing import Callable, Sequence
from uuid import uuid4

from genbase.data import import_data, train_test_split
from instancelib.environment.text import TextEnvironment
from instancelib.instances.text import MemoryTextInstance
from instancelib.typehints import LT

from ..utils import default_tokenizer


def from_list(instances: Sequence[str], labels: Sequence[LT]) -> TextEnvironment:
    """Create a TextEnvironment from a list of instances, and list of labels

    Example:
        >>> from_list(instances=['A positive test.', 'A negative test.', 'Another positive test'],
        >>>           labels=['pos', 'neg', 'pos'])

    Args:
        instances (Sequence[str]): List of instances.
        labels (Sequence[LT]): List of corresponding labels.

    Returns:
        TextEnvironment: Environment holding data (`.dataset`) and labelprovider (`.labels`).
    """
    instances, labels = list(instances), list(labels)

    return TextEnvironment.from_data(indices=list(range(len(instances))),
                                     data=instances,
                                     target_labels=list(set(labels)),
                                     ground_truth=[[label] for label in labels],
                                     vectors=[])


def from_string(string: str, tokenizer: Callable[[str], Sequence[str]] = default_tokenizer) -> MemoryTextInstance:
    """Create a MemoryTextInstance from a string.

    Example:
        >>> from_string('This is a test example.')

    Args:
        string (str): Input string.
        tokenizer (Callable[[str], Sequence[str]], optional): Tokenizer that converts string into list of tokens 
            (e.g. words or characters). Defaults to default_tokenizer.

    Returns:
        MemoryTextInstance: Holds information on the string, and its tokenized representation.
    """
    return MemoryTextInstance(str(uuid4()), data=string, vector=None, tokenized=tokenizer(string))


__all__ = ['import_data', 'from_list', 'from_string', 'train_test_split']
