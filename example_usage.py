# %% General imports
from text_explainability.data import import_data, train_test_split
from text_explainability.model import from_sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# %% Create train/test dataset
env = import_data('./datasets/test.csv', data_cols=['fulltext'], label_cols=['label'])
train, test = train_test_split(env, train_size=0.70)
labelprovider = env.labels

# %% Create sklearn model with pipeline
pipeline = Pipeline([('tfidf', TfidfVectorizer(use_idf=True)),
                     ('rf', RandomForestClassifier(random_state=0))])

# %% Wrap sklearn model
model = from_sklearn(pipeline, environment=env, train=train)

# %% Imports
from text_explainability.data import from_string

from text_explainability.local_explanation import LIME, LocalTree, Anchor, KernelSHAP, FoilTree, LocalRules
from text_explainability.global_explanation import (TokenFrequency, TokenInformation,
                                                    MMDCritic, KMedoids, LabelwiseMMDCritic)
from text_explainability.data.augmentation import TokenReplacement, LeaveOut
from text_explainability.utils import default_detokenizer, PUNCTUATION

# %% Create example instance
sample = from_string('Dit is zeer positieve of negatieve proef... Of toch negatief?')

# %% 
repl = TokenReplacement(env, default_detokenizer)

# %% Sequential replacement, 10 samples
print(list(repl(sample, n_samples=10).all_data()))

# %% Non-sequential replacement, 10 samples
print(list(repl(sample, n_samples=10, sequential=False).all_data()))

# %% Non-sequential, contiguous replacement, 10 samples
print(list(repl(sample, n_samples=10, sequential=False, contiguous=True).all_data()))

# %% Sequential deletion, 10 samples
print(list(LeaveOut(env, default_detokenizer)(sample, n_samples=10).all_data()))

# %% LIME explainer for `sample` on `model`
explainer = LIME(env)
explainer(sample, model, labels=['neutraal', 'positief']).scores

# %% Local tree explainer for `sample` on `model` (non-weighted neighborhood data)
LocalTree()(sample, model, weigh_samples=False).rules

# %% SHAP explanation for `sample` on `model`, limited to 4 features
KernelSHAP(labelset=labelprovider)(sample, model, n_samples=100, l1_reg=4)

# %% Anchor explanation for `sample` on `model`
#Anchor(label_names=['neg', 'net', 'pos'])(sample, model)

# %% FoilTree explanation for `sample` on `model` (why not 'neg'?)
FoilTree()(sample, model, 'positief').rules

# %% LocalRules on `model` (why 'positief'?)
LocalRules()(sample, model, 'negatief', n_samples=100).rules

# %% Global word frequency explanation on ground-truth labels
tf = TokenFrequency(train)
tf(labelprovider=labelprovider, explain_model=False, k=10)

# %% Global word frequency explanation on model predictions
tf(model=model, explain_model=True, k=3, filter_words=PUNCTUATION)

# %% Token information for dataset
ti = TokenInformation(train)
ti(labelprovider=labelprovider, explain_model=False, k=25).scores

# %% Token information for model
ti(model=model, explain_model=True, k=25, filter_words=PUNCTUATION)

# %% Extract top-2 prototypes with KMedoids
KMedoids(env.dataset).prototypes(n=2)

# %% Extract top-2 prototypes and top-2 criticisms label with MMDCritic
MMDCritic(env.dataset)(n_prototypes=2, n_criticisms=2)

# %% Extract 1 prototype for each ground-truth label with MMDCritic
LabelwiseMMDCritic(env.dataset, labelprovider).prototypes(n=1)

# %% Extract 1 prototype and 2 criticisms for each ground-truth label with MMDCritic
LabelwiseMMDCritic(env.dataset, labelprovider)(n_prototypes=1, n_criticisms=2)

# %% Extract 1 prototype and 1 criticism for each predicted label with MMDCritic
LabelwiseMMDCritic(env.dataset, model)(n_prototypes=1, n_criticisms=1)
