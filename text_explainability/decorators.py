"""Function decorators to ensure functions are fool-proof en readable."""

import inspect
from functools import wraps, partial

from uuid import uuid4
from instancelib.instances.text import MemoryTextInstance

from text_explainability.utils import default_tokenizer


def text_instance(func=None, *, tokenize: bool = False):
    """Decorator to convert an accidentally passed string to a TextInstance."""
    if func is None:
        return partial(text_instance, tokenize=tokenize)

    def str_to_text_instance(arg):
        if isinstance(arg, str):
            arg = MemoryTextInstance(str(uuid4()), arg, None)
        if tokenize and not arg.tokenized:
            arg.tokenized = default_tokenizer(arg.data)
        return arg

    @wraps(func)
    def inner(*args, **kwargs):
        possible_args = [i for i, t in enumerate(inspect.signature(func).parameters.values())
                         if 'TextInstance' in str(t)]            
        if possible_args:
            args = tuple(str_to_text_instance(a) if j in possible_args else a for j, a in enumerate(list(args)))
        return func(*args, **kwargs)
    return inner
