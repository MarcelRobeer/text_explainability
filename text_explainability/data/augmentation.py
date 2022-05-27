"""Augment a single instance to generate neighborhood data.

Todo:

    * Add more complex sampling methods (e.g. top-k replacement by contextual language model, WordNet, ...)
    * Replacement with k tokens at each index
    * Ensure inactive[i] is set to 0 if the replacement token is the same as the original token[i]
"""

import itertools
import math
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from genbase import Readable, SeedMixin
from instancelib.environment.base import AbstractEnvironment
from instancelib.environment.text import TextEnvironment
from instancelib.instances.base import InstanceProvider
from instancelib.instances.text import TextInstance
from instancelib.pertubations.base import ChildGenerator, MultiplePertubator

from text_explainability.decorators import text_instance
from text_explainability.utils import default_detokenizer


class LocalTokenPertubator(MultiplePertubator[TextInstance], 
                           ChildGenerator[TextInstance], 
                           Readable):
    def __init__(self,
                 env: Optional[AbstractEnvironment[TextInstance, Any, Any, Any, Any, Any]] = None,
                 detokenizer: Optional[Callable[[Iterable[str]], str]] = default_detokenizer):
        """Perturb a single instance into neighborhood samples.

        Args:
            detokenizer (Callable[[Iterable[str]], str]): Mapping back from a tokenized instance 
                to a string used in a predictor.
        """
        super().__init__()
        if env is None:
            env = TextEnvironment.from_data([], [], [], [], [])
        self.env = env
        self.detokenizer = detokenizer

    @staticmethod
    def binary_inactive(inactive, length) -> np.ndarray:
        res = np.zeros(length, dtype=int)
        inactive = [res for res in inactive]
        res[inactive] = 1
        return res

    def perturb(self, tokenized_instance: Iterable[str], 
                *args: Any, **kwargs: Any) -> Iterator[Tuple[Iterable[str], Iterable[int]]]:
        raise NotImplementedError('Implemented in subclasses')

    def discard_children(self, parent: TextInstance) -> None:
        """Discard the generated children for a given parent."""
        self.env.discard_children(parent)

    def get_children(self, parent: TextInstance) -> InstanceProvider[TextInstance, Any, Any, Any, Any]:
        """Get the children of a given parent."""
        return self.env.get_children(parent)

    @text_instance(tokenize=True)
    def __call__(self,
                 instance: TextInstance,
                 discard_children: bool = True,
                 *args,
                 **kwargs) -> Iterator[TextInstance]:
        """Apply perturbations to an instance to generate neighborhood data.

        Args:
            instance (TextInstance): Tokenized instance to perturb.
            discard_children (bool, optional): Remove children from previous passes. Defaults to True.
            *args: Arguments to be passed on to `perturb()` function.
            **kwargs: Keyword arguments to be passed on to `perturb()` function.

        Yields:
            Iterator[Sequence[TextInstance]]: Neighborhood data instances.
        """
        if instance.data not in self.env.all_instances.all_data():
            provider = self.env.create_empty_provider()
            provider.add(instance)

        if discard_children:
            self.discard_children(instance)

        for new_tokenized, map_to_original in self.perturb(instance.tokenized, *args, **kwargs):
            new_data = self.detokenizer(new_tokenized)
            new_instance = self.env.create(
                data=new_data,
                vector=None,
                map_to_original=map_to_original, 
                representation=new_data,
                tokenized=new_tokenized
                )
            self.register_child(instance, new_instance)
        return self.get_children(instance)


class TokenReplacement(LocalTokenPertubator, SeedMixin):
    def __init__(self,
                 env: Optional[AbstractEnvironment[TextInstance, Any, Any, Any, Any, Any]] = None,
                 detokenizer: Optional[Callable[[Iterable[str]], str]] = default_detokenizer,
                 replacement: Optional[Union[str, List[str], Dict[int, Optional[Union[str, List[str]]]]]] = 'UNKWRDZ',
                 seed: int = 0):
        """Perturb a tokenized instance by replacing with a set token (e.g. 'UNKWRDZ') or deleting it.

        Examples:
            Randomly replace at least two tokens with the replacement word 'UNK':

            >>> from text_explainability.augmentation import TokenReplacement
            >>> TokenReplacement(replacement='UNK').perturb(['perturb', 'this', 'into', 'multiple'],
            >>>                                             n_samples=3,
            >>>                                             min_changes=2)

            Perturb each token with ['UNK', None]:

            >>> from text_explainability.augmentation import TokenReplacement
            >>> TokenReplacement(replacement=['UNK', None]).perturb(['perturb', 'this', 'into', 'multiple'], ...)

            Perturb with synonyms:

            >>> from text_explainability.augmentation import TokenReplacement
            >>> replacement = {0: ['change', 'adjust'], 1: None, 2: 'to', 3: 'more'}
            >>> TokenReplacement(replacement=replacement).perturb(['perturb', 'this', 'into', 'multiple'], ...)

        Args:
            detokenizer (Callable[[Iterable[str]], str]): Mapping back from a tokenized instance to a string used in a 
                predictor.
            replacement (Optional[Union[str, List[str], Dict[int, Optional[Union[str, List[str]]]]]], optional): 
                Replacement string, list or dictionary, or set to None if you want to delete the word entirely. 
                Defaults to 'UNKWRDZ'.
            seed (int, optional): Seed for reproducibility. Defaults to 0.

        """
        super().__init__(env=env, detokenizer=detokenizer)
        self.replacement = replacement
        self.__update_replacement()
        self._seed = self._original_seed = seed

    def __update_replacement(self, tokenized_instance: Optional[Iterable[str]] = None):
        if not hasattr(self, '_replacement'):
            self._replacement = None
        elif not isinstance(self._replacement, dict) or len(self._replacement) != len(tokenized_instance):
            if not self.replacement or self.replacement is None:
                self._replacement = {i: [None] for i in range(len(tokenized_instance))}
            elif isinstance(self.replacement, str):
                self._replacement = {i: [self.replacement] for i in range(len(tokenized_instance))}
            elif isinstance(self.replacement, list):
                self._replacement = {i: self.replacement for i in range(len(tokenized_instance))}
            elif isinstance(self.replacement, dict) and tokenized_instance is not None:
                instance_len = sum(1 for _ in tokenized_instance)
                replacement_len = len(self.replacement)
                if not (replacement_len >= instance_len):
                    raise ValueError(f'Too few replacements in `self.replacement`, got {replacement_len} ',
                                     f'and expected {instance_len}')
                self._replacement = {k: v if isinstance(v, list) else [v] for k, v in self.replacement.items()}

    def __required_changes(self,
                           tokenized_instance: Iterable[str],
                           contiguous: bool,
                           min_changes: int,
                           max_changes: int) -> int:
        instance_len = len(list(tokenized_instance))
        if contiguous:  # (T+1)(B-A+1) + A(A-1)/2 - B(B+1)/2
            return int((instance_len + 1) * (max_changes - min_changes + 1) +
                       min_changes * (min_changes - 1) / 2 -
                       max_changes * (max_changes + 1) / 2)
        return sum(math.comb(instance_len, i) for i in range(min_changes, max_changes + 1))

    def _replace(self,
                 tokenized_instance: Iterable[str],
                 keep: Iterable[int]) -> Iterable[str]:
        """Apply replacement/deletion to tokenized instance.

        Args:
            tokenized_instance (Iterable[str]): Tokenized instance.
            keep (Iterable[int]): Binary indicator whether to keep (1) or replace (0) a token.

        Raises:
            ValueError: Too few replacements in self.replacement.

        Returns:
            Iterable[str]: Tokenized instance with perturbation applied.
        """        
        for idx, (token, i) in enumerate(zip(tokenized_instance, keep)):
            if i == 0:
                yield token
            else:
                repl = self._replacement[idx]
                res = self._rand.choice(repl) if i < 0 or i > len(repl) else repl[i - 1]
                if res is not None:
                    yield res 

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

        Example:
            Randomly replace at least two tokens with the replacement word 'UNK':

            >>> from text_explainability.augmentation import TokenReplacement
            >>> TokenReplacement(replacement='UNK').perturb(['perturb', 'this', 'into', 'multiple'],
            >>>                                             n_samples=3,
            >>>                                             min_changes=2)

        Args:
            tokenized_instance (Iterable[str]): Tokenized instance to apply perturbations to.
            n_samples (int, optional): Number of samples to return. Defaults to 50.
            sequential (bool, optional): Whether to sample sequentially based on length (first length one, then two, 
                etc.). Defaults to True.
            contiguous (bool, optional): Whether to remove contiguous sequences of tokens (n-grams). Defaults to False.
            min_changes (int, optional): Minimum number of tokens changes (1+). Defaults to 1.
            max_changes (int, optional): Maximum number of tokens changed. Defaults to 10000.
            add_background_instance (bool, optional): Add an additional instance with all tokens replaced. 
                Defaults to False.

        Raises:
            ValueError: min_changes cannot be greater than max_changes.

        Yields:
            Iterator[Sequence[Iterable[str], Iterable[int]]]: Perturbed text instances and indices where
                perturbation were applied.
        """
        instance_len = sum(1 for _ in tokenized_instance)
        min_changes = min(max(min_changes, 1), instance_len)
        max_changes = min(instance_len, max_changes)
        self._rand = np.random.RandomState(self.seed)
        self.__update_replacement(tokenized_instance)

        # avoid duplication in case n_samples >= required_changes
        required_changes = self.__required_changes(tokenized_instance, contiguous, min_changes, max_changes)

        def get_inactive(inactive_range):
            inactive = TokenReplacement.binary_inactive(inactive_range, instance_len)
            return list(self._replace(tokenized_instance, np.where(inactive > 0, -1, inactive))), inactive

        if sequential or n_samples >= required_changes:
            if contiguous:  # n-grams of length size, up to n_samples 
                for size in range(min_changes, max_changes + 1):
                    n_contiguous = instance_len - size
                    if n_contiguous <= n_samples:
                        n_samples -= n_contiguous
                        for start in range(instance_len - size + 1):
                            yield get_inactive(range(start, start + size))
                    else:
                        for start in self._rand.choice(instance_len - size + 1, size=n_samples, replace=False):
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
                            yield get_inactive(self._rand.choice(instance_len, size, replace=False))
                        break
        else:
            sample = self._rand.randint(min_changes, max_changes + 1, n_samples)

            for size in sample:
                if contiguous:  # use n-grams
                    start = self._rand.choice(max_changes - size + 1, replace=False)
                    inactive = TokenReplacement.binary_inactive(range(start, start + size), instance_len)
                else:  # used by LIME
                    inactive = TokenReplacement.binary_inactive(self._rand.choice(instance_len, size, replace=False),
                                                                instance_len)
                yield self._replace(tokenized_instance, inactive), inactive

        if add_background_instance:
            inactive = np.zeros(instance_len)
            yield list(self._replace(tokenized_instance, inactive)), inactive


class LeaveOut(TokenReplacement):
    def __init__(self,
                 env: Optional[AbstractEnvironment[TextInstance, Any, Any, Any, Any, Any]] = None,
                 detokenizer: Optional[Callable[[Iterable[str]], str]] = default_detokenizer,
                 seed: int = 0):
        """Leave tokens out of the tokenized sequence.

        Args:
            detokenizer (Callable[[Iterable[str]], str]): Mapping back from a tokenized 
                instance to a string used in a predictor.
            seed (int, optional): Seed for reproducibility. Defaults to 0.
        """
        super().__init__(env=env, detokenizer=detokenizer, replacement=None, seed=seed)
