# %% General imports
import os

from instancelib.ingest.spreadsheet import read_csv_dataset
from instancelib.instances.text import MemoryTextInstance

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

# %% Create train/test dataset
path = os.path.join(os.path.dirname(__file__), './datasets/test.csv')
test_env = read_csv_dataset(path, data_cols=['fulltext'], label_cols=['label'])
instanceprovider = test_env.dataset
labelprovider = test_env.labels
train, test = test_env.train_test_split(instanceprovider, train_size=0.70)

# %% Create sklearn model with pipeline
p = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer(use_idf=False)),
              ('rf', RandomForestClassifier())
             ])

# %% Imports
from instancelib.machinelearning import SkLearnDataClassifier

from text_explainability.local_explanation import LIME, LocalTree, Anchor, KernelSHAP, FoilTree
from text_explainability.global_explanation import TokenFrequency, TokenInformation
from text_explainability.data.augmentation import TokenReplacement, LeaveOut
from text_explainability.utils import default_detokenizer, default_tokenizer, PUNCTUATION

# %% Wrap sklearn model
model = SkLearnDataClassifier.build(p, test_env)
model.fit_provider(train, labelprovider)

# %% Create example instance
data = 'Dit is zeer positieve proef...'
sample = MemoryTextInstance(0, data, None, tokenized = default_tokenizer(data))

# %% 
repl = TokenReplacement(test_env, default_detokenizer)

# %% Sequential replacement, 10 samples
print(list(repl(sample, n_samples=10).all_data()))

# %% Non-sequential replacement, 10 samples
print(list(repl(sample, n_samples=10, sequential=False).all_data()))

# %% Non-sequential, contiguous replacement, 10 samples
print(list(repl(sample, n_samples=10, sequential=False, contiguous=True).all_data()))

# %% Sequential deletion, 10 samples
print(list(LeaveOut(test_env, default_detokenizer)(sample, n_samples=10).all_data()))

# %% LIME explainer for `sample` on `model`
explainer = LIME(test_env)
explainer(sample, model, labels=['neutraal', 'positief']).scores

# %% Local tree explainer for `sample` on `model` (non-weighted neighborhood data)
LocalTree()(sample, model, weigh_samples=False)

# %% SHAP explanation for `sample` on `model`, limited to 4 features
KernelSHAP(labelset=labelprovider)(sample, model, n_samples=50, l1_reg=4)

# %% Anchor explanation for `sample` on `model`
#Anchor(label_names=['neg', 'net', 'pos'])(sample, model)

# %% FoilTree explanation for `sample` on `model` (why not 'neg'?)
FoilTree()(sample, model, 'positief')

# %% Global word frequency explanation on ground-truth labels
tf = TokenFrequency(instanceprovider)
tf(labelprovider=labelprovider, explain_model=False, k=10)

# %% Global word frequency explanation on model predictions
tf(model=model, explain_model=True, k=3, filter_words=PUNCTUATION)

# %% Token information for dataset
ti = TokenInformation(instanceprovider)
ti(labelprovider=labelprovider, explain_model=False, k=25).scores

# %% Token information for model
ti(model=model, explain_model=True, k=25, filter_words=PUNCTUATION)

