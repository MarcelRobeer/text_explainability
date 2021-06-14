"""TO-DO:
- Add more complex sampling methods (e.g. top-k replacement by contextual language model, WordNet, ...)
- Replacement with k tokens at each index
- Ensure inactive[i] is set to 0 if the replacement token is the same as the original token[i]
"""

from instancelib.environment.base import AbstractEnvironment
import numpy as np
import math
import itertools

from typing import (Callable, Iterable, Sequence, Any, Iterator, Text, Tuple, Optional, List, Union)

from instancelib.instances.text import TextInstance
from instancelib.pertubations.base import MultiplePertubator, ChildGenerator
from instancelib.typehints.typevars import VT, KT

from text_explainability.default import Readable


class LocalTokenPertubator(MultiplePertubator[TextInstance], 
                           ChildGenerator[TextInstance], 
                           Readable):
    def __init__(self,
                 env: AbstractEnvironment[TextInstance, Any, Any, Any, Any, Any],
                 detokenizer: Callable[[Iterable[str]], str]):
        """Perturb a single instance into neighborhood samples.

        Args:
            detokenizer (Callable[[Iterable[str]], str]): Mapping back from a tokenized instance to a string used in a predictor.
        """
        super().__init__()
        self.env = env
        self.detokenizer = detokenizer

    @staticmethod
    def binary_inactive(inactive, length) -> np.ndarray:
        res = np.ones(length, dtype=int)
        inactive = [res for res in inactive]
        res[inactive] = 0
        return res

    def perturb(tokenized_instance: Iterable[str], 
                *args: Any, **kwargs: Any) -> Iterator[Tuple[Iterable[str], Iterable[int]]]:
        raise NotImplementedError

    def __call__(self, instance: TextInstance[KT, VT], *args, **kwargs) -> Iterator[TextInstance[KT, VT]]:
        """Apply perturbations to an instance to generate neighborhood data.

        Args:
            instance (TextInstance[Any, VT]): Tokenized instance to perturb.

        Yields:
            Iterator[Sequence[TextInstance[Any, VT]]]: Neighborhood data instances.
        """
        assert hasattr(instance, 'tokenized'), 'Tokenize your instance before applying a perturbation'

        for new_tokenized, map_to_original in self.perturb(instance.tokenized, *args, **kwargs):
            new_data = self.detokenizer(new_tokenized)
            new_instance = self.env.create(
                data=new_data, 
                vector=map_to_original, 
                representation=new_data,
                tokenized=new_tokenized
                )
            self.register_child(instance, new_instance)
            yield new_instance


class TokenReplacement(LocalTokenPertubator):
    def __init__(self,
                 env: AbstractEnvironment[TextInstance, Any, Any, Any, Any, Any],
                 detokenizer: Callable[[Iterable[str]], str], 
                 replacement: Optional[Union[str, List[str]]] = 'UNKWRDZ',
                 seed: int = 0):
        """[summary]

        Args:
            detokenizer (Callable[[Iterable[str]], str]): Mapping back from a tokenized instance to a string used in a predictor.
            replacement (Optional[Union[str, List[str]]], optional): Replacement string, or set to None
                if you want to delete the word entirely. Defaults to 'UNKWRDZ'.
            seed (int, optional): Seed for reproducibility. Defaults to 0.
        """
        super().__init__(env= env, detokenizer=detokenizer)
        self.replacement = replacement
        self._seed = seed

    def _replace(self,
                 tokenized_instance: Iterable[str],
                 keep: Iterable[int]):
        """[summary]

        Args:
            tokenized_instance (Iterable[str]): Tokenized instance.
            keep (Iterable[int]): Binary indicator whether to keep (1) or replace (0) a token.

        Returns:
            [type]: [description]
        """
        if not self.replacement or self.replacement is None:
            return [token for token, i in zip(tokenized_instance, keep) if i == 1]
        if isinstance(self.replacement, list):
            instance_len = sum(1 for _ in tokenized_instance)
            replacement_len = len(self.replacement)
            assert replacement_len >= instance_len, f'Too few replacements in `self.replacement`, got {replacement_len} and expected {instance_len}'
            return [self.replacement[i] if j == 0 else token for i, (token, j) in enumerate(zip(tokenized_instance, keep))]
        return [self.replacement if i == 0 else token for token, i in zip(tokenized_instance, keep)]

    def perturb(self,
                tokenized_instance: Iterable[str],
                n_samples: int = 50,
                sequential: bool = True,
                contiguous: bool = False,
                min_changes: int = 1,
                max_changes: int = 10000,
                add_background_instance: bool = False) -> Iterator[Tuple[Iterable[str], Iterable[int]]]:
        """Perturb a tokenized instance by replacing it with a single replacement token (e.g. 'UNKWRDZ'), 
        which is assumed not to be part of the original tokens.

        Args:
            tokenized_instance (Iterable[str]): [description]
            n_samples (int, optional): Number of samples to return. Defaults to 50.
            sequential (bool, optional): Whether to sample sequentially based on length (first length one, then two, etc.). Defaults to True.
            contiguous (bool, optional): Whether to remove contiguous sequences of tokens (n-grams). Defaults to False.
            min_changes (int, optional): Minimum number of tokens changes (1+). Defaults to 1.
            max_changes (int, optional): Maximum number of tokens changed. Defaults to 10000.
            add_background_instance (bool, optional): Add an additional instance with all tokens replaced. Defaults to False.

        Yields:
            Iterator[Sequence[Iterable[str], Iterable[int]]]: [description]
        """
        instance_len = sum(1 for _ in tokenized_instance)
        min_changes = min(max(min_changes, 1), instance_len)
        max_changes = min(instance_len, max_changes)
        assert min_changes <= max_changes, f'Unable to produce any perturbations since min_changes={min_changes} and max_changes={max_changes}'
        rand = np.random.RandomState(self._seed)

        def get_inactive(inactive_range):
            inactive = TokenReplacement.binary_inactive(inactive_range, instance_len)
            return self._replace(tokenized_instance, inactive), inactive

        if sequential:
            if contiguous:  # n-grams of length size, up to n_samples
                for size in range(min_changes, max_changes + 1):
                    n_contiguous = instance_len - size
                    if n_contiguous <= n_samples:
                        n_samples -= n_contiguous
                        for start in range(instance_len - size + 1):
                            yield get_inactive(range(start, start + size))
                    else:
                        for start in rand.choice(instance_len - size + 1, replace=False):
                            yield get_inactive(range(start, start + size))
                        break
            else:  # used by SHAP
                for size in range(min_changes, max_changes + 1):
                    n_choose_k = math.comb(instance_len, size)
                    if n_choose_k <= n_samples:  # make all combinations of length size
                        n_samples -= n_choose_k
                        for disable in itertools.combinations(range(instance_len), size):
                            yield get_inactive(disable)
                    else:  # fill up remainder with random samples of length size
                        for _ in range(n_samples):
                            yield get_inactive(rand.choice(instance_len, size, replace=False))
                        break
        else:
            sample = rand.randint(min_changes, max_changes + 1, n_samples)

            for size in sample:
                if contiguous: # use n-grams
                    start = rand.choice(max_changes - size + 1, replace=False)
                    inactive = TokenReplacement.binary_inactive(range(start, start + size), instance_len)
                else: # used by LIME, https://github.com/marcotcr/lime/blob/a2c7a6fb70bce2e089cb146a31f483bf218875eb/lime/lime_text.py#L436
                    inactive = TokenReplacement.binary_inactive(rand.choice(instance_len, size, replace=False), instance_len)
                yield self._replace(tokenized_instance, inactive), inactive
        
        if add_background_instance:
            inactive = np.zeros(instance_len)
            yield self._replace(tokenized_instance, inactive), inactive


class LeaveOut(TokenReplacement):
    def __init__(self,
                 env: AbstractEnvironment[TextInstance, Any, Any, Any, Any, Any],
                 detokenizer: Callable[[Iterable[str]], str], 
                 seed: int = 0):
        """Leave tokens out of the tokenized sequence.

        Args:
            detokenizer (Callable[[Iterable[str]], str]): Mapping back from a tokenized instance to a string used in a predictor.
            seed (int, optional): Seed for reproducibility. Defaults to 0.
        """
        super().__init__(env=env, detokenizer=detokenizer, replacement=None, seed=seed)
