# %% General imports
from instancelib.ingest.spreadsheet import read_csv_dataset
from instancelib.instances.text import TextInstance

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

# %% Create train/test dataset
test_env = read_csv_dataset('./datasets/test.csv', data_cols=['fulltext'], label_cols=['label'])
instanceprovider = test_env.dataset
labelprovider = test_env.labels
train, test = test_env.train_test_split(instanceprovider, train_size=0.70)

# %% Fit sklearn model
p = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer(use_idf=False)),
              ('rf', RandomForestClassifier())
             ])
p.fit([t.data for t in train.bulk_get_all()],
      [list(labelprovider.get_labels(k))[0] for k in train])

# %% Imports
from text_explainability.model import SklearnModel
from text_explainability.local_explanation import LIME, LocalTree, Anchor, KernelSHAP
from text_explainability.global_explanation import TokenFrequency, TokenInformation
from text_explainability.data.augmentation import TokenReplacement, LeaveOut
from text_explainability.utils import default_detokenizer, default_tokenizer, PUNCTUATION

# %% Wrap sklearn model
model = SklearnModel(p)

# %% Create example instance
sample = TextInstance(0, 'Dit is zeer positieve proef...', None)
sample.tokenized = default_tokenizer(sample.data)

# %% 
repl = TokenReplacement(test_env, default_detokenizer)

# %% Sequential replacement, 10 samples
print([i.data for i in repl(sample, n_samples=10)])

# %% Non-sequential replacement, 10 samples
print([i.data for i in repl(sample, n_samples=10, sequential=False)])

# %% Non-sequential, contiguous replacement, 10 samples
print([i.data for i in repl(sample, n_samples=10, sequential=False, contiguous=True)])

# %% Sequential deletion, 10 samples
print([i.data for i in LeaveOut(test_env, default_detokenizer)(sample, n_samples=10)])

# %% LIME explainer for `sample` on `model`
explainer = LIME(test_env)
explainer(sample, model, labels=['neutraal', 'positief']).scores

# %% Local tree explainer for `sample` on `model` (non-weighted neighborhood data)
LocalTree()(sample, model, weigh_samples=False)

# %% SHAP explanation for `sample` on `model`, limited to 4 features
KernelSHAP(label_names=labelprovider)(sample, model, n_samples=50, l1_reg=4)

# %% Anchor explanation for `sample` on `model`
#Anchor(label_names=['neg', 'net', 'pos'])(sample, model)

# %% Global word frequency explanation on ground-truth labels
tf = TokenFrequency(instanceprovider)
tf(labelprovider=labelprovider, explain_model=False, k=10)

# %% Global word frequency explanation on model predictions
tf(model=model, explain_model=True, k=3, filter_words=PUNCTUATION)

# %% Token information for dataset
ti = TokenInformation(instanceprovider)
ti(labelprovider=labelprovider, explain_model=False, k=50)

# %% Token information for model
ti(model=model, explain_model=True, k=50, filter_words=PUNCTUATION)

#%%