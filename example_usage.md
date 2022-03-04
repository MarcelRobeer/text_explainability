# Example Usage

## Dependencies
`text_explainability` uses instances and machine learning models wrapped with the [InstanceLib](https://pypi.org/project/instancelib/) library. For your convenience, we wrap some `instancelib` functions in `text_explainability.data` and `explainability.model`.
```python
from text_explainability.data import import_data, train_test_split, from_string
from text_explainability.model import import_model
```

## Dataset and model
As a dummy black-box model, we use the example dataset in `./datasets/test.csv` and train a machine learning model on it with `scikit-learn`.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Create train/test dataset
env = import_data('./datasets/test.csv', data_cols='fulltext', label_cols='label')
env = train_test_split(env, train_size=0.70)

# Create sklearn model with pipeline
pipeline = Pipeline([('tfidf', TfidfVectorizer(use_idf=True)),
                     ('rf', RandomForestClassifier(random_state=0))])

# Build and fit (train) model
model = import_model(pipeline, environment=env)
```

## Using Text Explainability
Text Explainability is used for _local explanations_ (explaining a single prediction) or _global explanations_ (explaining general dataset/model behavior).

### Local explanations
Popular local explanations include `LIME`, `KernelSHAP` , local decion trees (`LocalTree`), local decision rules (`LocalRules`) and `FoilTree`. First, let us create a sample to explain:

```python
from text_explainability.data import from_string

sample = from_string('Dit is zeer positieve of negatieve proef... Of toch negatief?')
```

Next, the prediction of `model` on `sample` can be explained by generating neighborhood data (`text_explainability.data.augmentation.TokenReplacement`), used by `LIME`, `LocalTree`, `FoilTree` and `KernelSHAP`:

```python
from text_explainability import LIME, LocalTree, FoilTree, KernelSHAP

# LIME explainer for `sample` on `model`
explainer = LIME(env)
explainer(sample, model, labels=['neutraal', 'positief']).scores

# SHAP explanation for `sample` on `model`, limited to 4 features
KernelSHAP(label_names=labelprovider)(sample, model, n_samples=50, l1_reg=4)

# Local tree explainer for `sample` on `model` (non-weighted neighborhood data)
LocalTree()(sample, model, weigh_samples=False)

# Contrastive local tree explainer for `sample` on `model` (why not 'positief'?)
FoilTree()(sample, model, foil_fn='positief').rules

# LocalRules on `model` (why 'positief'?)
LocalRules()(sample, model, foil_fn='negatief', n_samples=100).rules
```

### Global explanations
Global explanations provide information on the dataset and its ground-truth labels, or the dataset and corresponding predictions by the `model`. Example global explanations are `TokenFrequency` (the frequency of each token per label/class/bucket) or `TokenInformation` (how informative each token is for predicting the various labels).

```python
from text_explainability import TokenFrequency, TokenInformation

# Global word frequency explanation on ground-truth labels
tf = TokenFrequency(env.dataset)
tf(labelprovider=env.labels, explain_model=False, k=10).scores

# Global word frequency explanation on model predictions
tf(model=model, explain_model=True, k=3, filter_words=PUNCTUATION)

# Token information for dataset
ti = TokenInformation(env.dataset)
ti(labelprovider=env.labels, explain_model=False, k=50).scores

# Token information for model
ti(model=model, explain_model=True, k=50, filter_words=PUNCTUATION)
```

#### Global explanation: Explanation by example
Explanations by example provide information on a dataset (e.g. the test set) or subsets thereof (e.g. all training instances with label 0) by showing representative instances. Examples of representative instances are prototypes (`n` most representative instances, e.g. of a class) and criticsms (`n` instances not well represented by prototypes). Example explanations by example are `KMedoids` (using the _k-Medoids_ algorithm to extract prototypes) and `MMDCritic` (extracting prototypes and corresponding criticisms). In addition, each of these can be performed labelwise (e.g. for the ground-truth labels in a `labelprovider` or for each models' predicted class).

```python
from text_explainability import KMedoids, MMDCritic, LabelwiseMMDCritic

# Extract top-2 prototypes with KMedoids
KMedoids(env.dataset).prototypes(n=2)

# Extract top-2 prototypes and top-2 criticisms label with MMDCritic
MMDCritic(env.dataset)(n_prototypes=2, n_criticisms=2)

# Extract 1 prototype for each ground-truth label with MMDCritic
LabelwiseMMDCritic(env.dataset, labelprovider).prototypes(n=1)

# Extract 1 prototype and 2 criticisms for each predicted label with MMDCritic
LabelwiseMMDCritic(env.dataset, model)(n_prototypes=1, n_criticisms=2)
```
