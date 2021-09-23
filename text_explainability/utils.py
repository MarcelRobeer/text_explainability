"""Utility functions."""

import re
import string
import numpy as np
from typing import Sequence, Iterable


def word_tokenizer(input: str) -> Sequence[str]:
    """Simple regex tokenizer."""
    return re.findall(r"\w+|[^\w\s]+", input)


def word_detokenizer(input: Iterable[str]) -> str:
    """Simple regex detokenizer, ideally resulting in `i = detokenizer(tokenizer(i))`."""
    out = " ".join(input).replace("`` ", '"') \
                         .replace(" ''", '"') \
                         .replace('. . .', '...') \
                         .replace(" ( ", " (") \
                         .replace(" ) ", ") ")
    out = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", out)
    out = re.sub(r' ([.,:;?!%]+)$', r"\1", out)
    out = re.sub(r'(\s+[0-9]+):\s+([0-9]+\s+)', r"\1:\2", out)
    return out.replace(" ` ", " '").strip()


def character_tokenizer(input: str) -> Sequence[str]:
    """Convert a string into a list of characters."""
    return list(input)


def character_detokenizer(input: Iterable[str]) -> str:
    """Convert a list of characters into a string."""
    return ''.join(input)


def binarize(X: np.ndarray):
    return (X > 0).astype(int)


default_tokenizer = word_tokenizer
default_detokenizer = word_detokenizer

PUNCTUATION = list(string.punctuation) + ['...']
