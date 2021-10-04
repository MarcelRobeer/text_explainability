"""Data imports, sampling and generation."""

import os

from typing import Callable, Union, Tuple, Sequence
from uuid import uuid4

from instancelib.environment.text import TextEnvironment
from instancelib.ingest.spreadsheet import read_csv_dataset, read_excel_dataset
from instancelib.instances.text import MemoryTextInstance, TextInstanceProvider

from text_explainability.utils import default_tokenizer


def import_data(filename: str, *args, **kwargs) -> TextEnvironment:
    """Import dataset using instancelib, currently supporting CSV and Excel files.

    Example:
        Read `test.csv` from the folder `datasets`:

        >>> from text_explainability.data import import_data
        >>> env = import_data('./datasets/test.csv', data_cols=['fulltext'], label_cols=['label'])
        >>> env.dataset, env.labels

    Args:
        filename (str): Filename, relative to current file or absolute path.

    Raises:
        ImportError: Cannot import file, unknown filetype.

    Returns:
        TextEnvironment: instancelib TextEnviroment. Access dataset through `.dataset()` and `.labels()`
    """
    filepath = os.path.abspath(filename)
    _, extension = os.path.splitext(filepath)
    extension = str.lower(extension).replace('.', '')

    if extension == 'csv':
        return read_csv_dataset(filepath, *args, **kwargs)
    elif extension.startswith('xls'):
        return read_excel_dataset(filepath, *args, **kwargs)
    else:
        raise ImportError(f'Unknown {extension=} for {filepath=}')


def train_test_split(environment: TextEnvironment,
                     train_size: Union[int, float]) -> Tuple[TextInstanceProvider, TextInstanceProvider]:
    """Split a dataset into training and test data.

    Args:
        environment (TextEnvironment): Environment containing all data (`environment.dataset`), 
            including labels (`environment.labels`).
        train_size (Union[int, float]): Size of training data, as a proportion [0, 1] or number of instances > 1.

    Returns:
        Tuple[TextInstanceProvider, TextInstanceProvider]: Train dataset, test dataset.
    """
    return environment.train_test_split(environment.dataset, train_size=train_size)


def from_string(string: str, tokenizer: Callable[[str], Sequence[str]] = default_tokenizer) -> MemoryTextInstance:
    """Create a MemoryTextInstance from a string.

    Args:
        string (str): Input string.
        tokenizer (Callable[[str], Sequence[str]], optional): Tokenizer that converts string into list of tokens 
            (e.g. words or characters). Defaults to default_tokenizer.

    Returns:
        MemoryTextInstance: Holds information on the string, and its tokenized representation.
    """
    return MemoryTextInstance(str(uuid4()), data=string, vector=None, tokenized=tokenizer(string))
