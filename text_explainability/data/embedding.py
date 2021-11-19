"""Embed text instances into numerical vectors.

Todo:

    * Add more sentence embedding methods
"""

from typing import Callable, Union

import numpy as np
from genbase import Readable
from instancelib.instances.memory import MemoryBucketProvider


def as_n_dimensional(vectors: Union[np.ndarray, list, MemoryBucketProvider],
                     n: int = 2,
                     method: str = 'pca',
                     **kwargs) -> np.ndarray:
    """Summarize vectors into n dimensions.

    Args:
        vectors (Union[np.ndarray, list, MemoryBucketProvider]): Vectors or BucketProvider with vectorized instances.
        n (int, optional): Number of dimensions (should be low, e.g. 2 or 3). Defaults to 2.
        method (str, optional): Method used for dimensionality reduction. Choose from ['pca', 'kernel_pca', 
            'incremental_pca', 'nmf', 'tsne']. Defaults to 'pca'.
        **kwargs: Optional arguments passed to method constructor.

    Raises:
        ValueError: Unknown method selected.

    Returns:
        np.ndarray: Vectors summarized in n dimensions.
    """
    from sklearn.decomposition import NMF, PCA, IncrementalPCA, KernelPCA
    from sklearn.manifold import TSNE

    methods = {'pca': PCA,
               'kernel_pca': KernelPCA,
               'incremental_pca': IncrementalPCA,
               'nmf': NMF,
               'tsne': TSNE}

    # Default to `init='pca'` for tsne to ensure stability
    if method == 'tsne' and 'init' not in kwargs:
        kwargs['init'] = 'pca'

    if method not in methods.keys():
        raise ValueError(f'Unknown {method=}. Choose from {list(methods.keys())}.')

    if isinstance(vectors, MemoryBucketProvider):
        vectors = vectors.bulk_get_vectors(list(vectors))[-1]
    return methods[method](n_components=n, **kwargs).fit_transform(vectors)


def as_2d(vectors: Union[np.ndarray, list, MemoryBucketProvider], method: str = 'pca', **kwargs) -> np.ndarray:
    """Summarize vectors in 2 dimensions."""
    return as_n_dimensional(vectors=vectors, n=2, method=method, **kwargs)


def as_3d(vectors: Union[np.ndarray, list, MemoryBucketProvider], method: str = 'pca', **kwargs) -> np.ndarray:
    """Summarize vectors in 3 dimensions."""
    return as_n_dimensional(vectors=vectors, n=3, method=method, **kwargs)


class Embedder(Readable):
    def __init__(self, model_fn: Callable):
        """Embedding model base class to transform instances into vectors.

        Args:
            model_fn (Callable): Model that embeds instances (transforms into vectors).
        """
        self.model_fn = model_fn

    def embed(self,
              instances: Union[np.ndarray, list, MemoryBucketProvider]) -> Union[np.ndarray, MemoryBucketProvider]:
        """Embed instances (transform into numerical vectors).

        Args:
            instances (Union[np.ndarray, list, MemoryBucketProvider]): Sequence of instances.

        Returns:
            Union[np.ndarray, MemoryBucketProvider]: Embedded instances (provided back into the BucketProvider if it
                was originally passed as a BucketProvider).
        """
        is_provider = isinstance(instances, MemoryBucketProvider)

        if is_provider:
            instances_ = instances
            instances = list(instances.all_data())

        embeddings = self.model_fn(instances)

        if is_provider:
            for idx, embedding in zip(instances_, embeddings):
                instances_[idx].vector = embedding
            return instances_

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        return embeddings

    def __call__(self,
                 instances: Union[np.ndarray, list, MemoryBucketProvider]) -> Union[np.ndarray, MemoryBucketProvider]:
        """Calls the `self.embed()` function."""
        return self.embed(instances)


class SentenceTransformer(Embedder):
    def __init__(self, model_name: str = 'distiluse-base-multilingual-cased-v1', **kwargs):
        """Embed sentences using the `Sentence Transformers`_ package.

        By default requires and active internet connection, or provide the name of a local `model_name`.

        Args:
            model_name (str, optional): Name of Sentence Transformer model. See 
                https://www.sbert.net/docs/pretrained_models.html for model names. Defaults to 
                'distiluse-base-multilingual-cased-v1'.
            **kwargs: Optional arguments to be passed to `SentenceTransformer.encode()` function. See
                https://www.sbert.net/examples/applications/computing-embeddings/README.html

        .. _Sentence Transformers:
            https://github.com/UKPLab/sentence-transformers
        """
        from sentence_transformers import \
            SentenceTransformer as SentTransformer
        self.model = SentTransformer(model_name)
        super().__init__(lambda x: self.model.encode(x, **kwargs))


class CountVectorizer(Embedder):
    def __init__(self, **kwargs):
        """Embed sentences using `sklearn.CountVectorizer`_.

        Args:
            **kwargs: Optional arguments passed for `sklearn.CountVectorizer()` construction.

        .. _sklearn.TfidfVectorizer:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        """
        from sklearn.feature_extraction.text import CountVectorizer as Count
        self.model = Count(**kwargs)
        super().__init__(lambda x: self.model.fit_transform(x).toarray())


class TfidfVectorizer(Embedder):
    def __init__(self, **kwargs):
        """Embed sentences using `sklearn.TfidfVectorizer`_.

        Args:
            **kwargs: Optional arguments passed for `sklearn.TfidfVectorizer()` construction.

        .. _sklearn.TfidfVectorizer:
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        """
        from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
        self.model = Tfidf(**kwargs)
        super().__init__(lambda x: self.model.fit_transform(x).toarray())
