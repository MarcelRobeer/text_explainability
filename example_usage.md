# Example Usage

## Dependencies
`text_explainability` uses instances and machine learning models wrapped with the [InstanceLib](https://pypi.org/project/instancelib/) library.
```python
import os

from instancelib.ingest.spreadsheet import read_csv_dataset
from instancelib.instances.text import MemoryTextInstance
```

## Dataset and model
As a dummy black-box model, we use the example dataset in `./datasets/test.csv` and train a machine learning model on it with `scikit-learn`.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from instancelib.machinelearning import SkLearnDataClassifier

# Create train/test dataset
path = os.path.join(os.path.dirname(__file__), './datasets/test.csv')
test_env = read_csv_dataset(path, data_cols=['fulltext'], label_cols=['label'])
instanceprovider = test_env.dataset
labelprovider = test_env.labels
train, test = test_env.train_test_split(instanceprovider, train_size=0.70)

# Create sklearn model with pipeline
p = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer(use_idf=False)),
              ('rf', RandomForestClassifier())
             ])

# Build and fit (train) model
model = SkLearnDataClassifier.build(p, test_env)
model.fit_provider(train, labelprovider)
```

## Using Text Explainability
Text Explainability is mainly used for local explanations (explaining a single prediction) or global explanations (explaining general model behavior).

### Local explanations
Popular local explanations include `LIME`, local decion trees (`LocalTree`), `KernelSHAP` and `FoilTree`. First, let us create a sample to explain:

```python
from text_explainability.utils import default_tokenizer

data = 'Dit is zeer positieve proef...'
sample = MemoryTextInstance(0, data, None, tokenized = default_tokenizer(data))
```

Next, the prediction of `model` on `sample` can be explained by generating neighborhood data (`text_explainability.data.augmentation.TokenReplacement`), used by `LIME`, `LocalTree`, `FoilTree` and `KernelSHAP`:

```python
from text_explainability.local_explanation import LIME, LocalTree, FoilTree, KernelSHAP

# LIME explainer for `sample` on `model`
explainer = LIME(test_env)
explainer(sample, model, labels=['neutraal', 'positief']).scores

# Local tree explainer for `sample` on `model` (non-weighted neighborhood data)
LocalTree()(sample, model, weigh_samples=False)

# Contrastive local tree explainer for `sample` on `model` (why not 'positief'?)
FoilTree()(sample, model, foil_fn='positief')

# SHAP explanation for `sample` on `model`, limited to 4 features
KernelSHAP(label_names=labelprovider)(sample, model, n_samples=50, l1_reg=4)
```

### Global explanations
Global explanations provide information on the dataset and its ground-truth labels, or the dataset and corresponding predictions by the `model`. Example global explanations are `TokenFrequency` (the frequency of each token per label/class/bucket) or `TokenInformation` (how informative each token is for predicting the various labels).

```python
from text_explainability.global_explanation import TokenFrequency, TokenInformation

# Global word frequency explanation on ground-truth labels
tf = TokenFrequency(instanceprovider)
tf(labelprovider=labelprovider, explain_model=False, k=10)

# Global word frequency explanation on model predictions
tf(model=model, explain_model=True, k=3, filter_words=PUNCTUATION)

# Token information for dataset
ti = TokenInformation(instanceprovider)
ti(labelprovider=labelprovider, explain_model=False, k=50).scores

# Token information for model
ti(model=model, explain_model=True, k=50, filter_words=PUNCTUATION)
```
