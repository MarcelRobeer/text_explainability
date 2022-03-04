*<p align="center">
  <img src="https://git.science.uu.nl/m.j.robeer/text_explainability/-/raw/main/img/TextLogo-Logo%20large.png" alt="T_xt Explainability logo" width="70%">*
</p>

**<h3 align="center">
A generic explainability architecture for explaining text machine learning models**
</h3>

[![PyPI](https://img.shields.io/pypi/v/text_explainability)](https://pypi.org/project/text-explainability/)
[![Python_version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/text-explainability/)
[![Build_passing](https://img.shields.io/badge/build-passing-brightgreen)](https://git.science.uu.nl/m.j.robeer/text_explainability/-/pipelines)
[![License](https://img.shields.io/pypi/l/text_explainability)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![Docs_passing](https://img.shields.io/badge/docs-external-blueviolet)](https://marcelrobeer.github.io/text_explainability)
[![Code style: black](https://img.shields.io/badge/code%20style-flake8-aa0000)](https://github.com/PyCQA/flake8)

---

`text_explainability` provides a **generic architecture** from which well-known state-of-the-art explainability approaches for text can be composed. This modular architecture allows components to be swapped out and combined, to **quickly develop new types of explainability approaches** for (natural language) text, or to **improve a plethora of approaches by improving a single module**.

Several example methods are included, which provide **local explanations** (_explaining the prediction of a single instance_, e.g. `LIME` and `SHAP`) or **global explanations** (_explaining the dataset, or model behavior on the dataset_, e.g. `TokenFrequency` and `MMDCritic`). By replacing the default modules (e.g. local data generation, global data sampling or improved embedding methods), these methods can be improved upon or new methods can be introduced.

&copy; Marcel Robeer, 2021-2022

## Quick tour
**Local explanation**: explain a models' prediction on a given sample, self-provided or from a dataset.
```python
from text_explainability import LIME, LocalTree

# Define sample to explain
sample = 'Explain why this is positive and not negative!'

# LIME explanation (local feature importance)
LIME().explain(sample, model).scores

# List of local rules, extracted from tree
LocalTree().explain(sample, model).rules
``` 

**Global explanation**: explain the whole dataset (e.g. train set, test set), and what they look like for the ground-truth or predicted labels.
```python
from text_explainability import import_data, TokenFrequency, MMDCritic

# Import dataset
env = import_data('./datasets/test.csv', data_cols=['fulltext'], label_cols=['label'])

# Top-k most frequent tokens per label
TokenFrequency(env.dataset).explain(labelprovider=env.labels, explain_model=False, k=3)

# 2 prototypes and 1 criticisms for the dataset
MMDCritic(env.dataset)(n_prototypes=2, n_criticisms=1)
```

## Installation
See the [installation](docs/installation.md) instructions for an extended installation guide.

| Method | Instructions |
|--------|--------------|
| `pip` | Install from [PyPI](https://pypi.org/project/text-explainability/) via `pip3 install text_explainability`. To speed up the explanation generation process use `pip3 install text_explainability[fast]`. |
| Local | Clone this repository and install via `pip3 install -e .` or locally run `python3 setup.py install`.

## Documentation
Full documentation of the latest version is provided at [https://marcelrobeer.github.io/text_explainability/](https://marcelrobeer.github.io/text_explainability/).

## Example usage
See [example usage](example_usage.md) to see an example of how the package can be used, or run the lines in `example_usage.py` to do explore it interactively.

## Explanation methods included
`text_explainability` includes methods for model-agnostic _local explanation_ and _global explanation_. Each of these methods can be fully customized to fit the explainees' needs.

| Type | Explanation method | Description | Paper/link |
|------|--------------------|-------------|-------|
| *Local explanation* | `LIME` | Calculate feature attribution with _Local Intepretable Model-Agnostic Explanations_ (LIME). | [[Ribeiro2016](https://paperswithcode.com/method/lime)], [interpretable-ml/lime](https://christophm.github.io/interpretable-ml-book/lime.html) |
| |  `KernelSHAP` | Calculate feature attribution with _Shapley Additive Explanations_ (SHAP). | [[Lundberg2017](https://paperswithcode.com/paper/a-unified-approach-to-interpreting-model)], [interpretable-ml/shap](https://christophm.github.io/interpretable-ml-book/shap.html) |
| |  `LocalTree` | Fit a local decision tree around a single decision. | [[Guidotti2018](https://paperswithcode.com/paper/local-rule-based-explanations-of-black-box)] |
| | `LocalRules` | Fit a local sparse set of label-specific rules using `SkopeRules`. | [github/skope-rules](https://github.com/scikit-learn-contrib/skope-rules) |
| |  `FoilTree` | Fit a local contrastive/counterfactual decision tree around a single decision. | [[Robeer2018](https://github.com/MarcelRobeer/ContrastiveExplanation)] |
| *Global explanation* | `TokenFrequency` | Show the top-_k_ number of tokens for each ground-truth or predicted label. |
| |  `TokenInformation` | Show the top-_k_ token mutual information for a dataset or model. | [wikipedia/mutual_information](https://en.wikipedia.org/wiki/Mutual_information) |
| | `KMedoids` | Embed instances and find top-_n_ prototypes (can also be performed for each label using `LabelwiseKMedoids`). | [interpretable-ml/prototypes](https://christophm.github.io/interpretable-ml-book/proto.html) |
| | `MMDCritic` | Embed instances and find top-_n_ prototypes and top-_n_ criticisms (can also be performed for each label using `LabelwiseMMDCritic`). | [[Kim2016](https://papers.nips.cc/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html)], [interpretable-ml/prototypes](https://christophm.github.io/interpretable-ml-book/proto.html) |

## Releases
`text_explainability` is officially released through [PyPI](https://pypi.org/project/text-explainability/).

See [CHANGELOG.md](CHANGELOG.md) for a full overview of the changes for each version.

## Extensions
<a href="https://marcelrobeer.github.io/text_sensitivity/" target="_blank"><img src="https://git.science.uu.nl/m.j.robeer/text_sensitivity/-/raw/main/img/TextLogo-Logo_large_sensitivity.png" alt="T_xt sensitivity logo" width="200px"></a><p>`text_explainability` can be extended to also perform _sensitivity testing_, checking for machine learning model robustness and fairness. The `text_sensitivity` package is available through [PyPI](https://pypi.org/project/text-sensitivity/) and fully documented at [https://marcelrobeer.github.io/text_sensitivity/](https://marcelrobeer.github.io/text_sensitivity/).</p>

## Citation
```bibtex
@misc{text_explainability,
  title = {Python package text\_explainability},
  author = {Marcel Robeer},
  howpublished = {\url{https://git.science.uu.nl/m.j.robeer/text_explainability}},
  year = {2021}
}
```

## Maintenance
### Contributors
- [Marcel Robeer](https://www.uu.nl/staff/MJRobeer) (`@m.j.robeer`)
- [Michiel Bron](https://www.uu.nl/staff/MPBron) (`@mpbron-phd`)

### Todo
Tasks yet to be done:

* Implement local post-hoc explanations:
    - Implement Anchors
* Implement global post-hoc explanations:
    - Representative subset
* Add support for regression models
* More complex data augmentation
    - Top-k replacement (e.g. according to LM / WordNet)
    - Tokens to exclude from being changed
    - Bag-of-words style replacements
* Add rule-based return type
* Write more tests

## Credits
- Florian Gardin, Ronan Gautier, Nicolas Goix, Bibi Ndiaye and Jean-Matthieu Schertzer. _[Skope-rules](https://github.com/scikit-learn-contrib/skope-rules)_. 2020.
- Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Dino Pedreschi, Franco Turini and Fosca Gianotti. _[Local Rule-Based Explanations of Black Box Decision Systems](https://paperswithcode.com/paper/local-rule-based-explanations-of-black-box)_. 2018.
- Been Kim, Rajiv Khanna and Oluwasanmi O. Koyejo. [Examples are not Enough, Learn to Criticize! Criticism for Interpretability](https://papers.nips.cc/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html). _Advances in Neural Information Processing Systems (NIPS 2016)_. 2016.
- Scott Lundberg and Su-In Lee. [A Unified Approach to Interpreting Model Predictions](https://paperswithcode.com/paper/a-unified-approach-to-interpreting-model). _31st Conference on Neural Information Processing Systems (NIPS 2017)_. 2017.
- Christoph Molnar. _[Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/)_. 2021.
- Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin. ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://paperswithcode.com/method/lime). _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2016)_. 2016.
- Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin. [Anchors: High-Precision Model-Agnostic Explanations](https://github.com/marcotcr/anchor). _AAAI Conference on Artificial Intelligence (AAAI)_. 2018.
- Jasper van der Waa, Marcel Robeer, Jurriaan van Diggelen, Matthieu Brinkhuis and Mark Neerincx. ["Contrastive Explanations with Local Foil Trees"](https://github.com/MarcelRobeer/ContrastiveExplanation). _2018 Workshop on Human Interpretability in Machine Learning (WHI 2018)_. 2018.
