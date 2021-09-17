<p align="center">
  <img src="https://i.ibb.co/xXtJ23n/Text-Logo-Logo-large.png" alt="T_xt Explainability logo" width="70%">
</p>

[![PyPI](https://img.shields.io/pypi/v/text_explainability)](https://pypi.org/project/text-explainability/)
[![Python_version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)](https://pypi.org/project/text-explainability/)
[![Build_passing](https://img.shields.io/badge/build-passing-brightgreen)](https://git.science.uu.nl/m.j.robeer/text_explainability/-/pipelines)
[![License](https://img.shields.io/pypi/l/text_explainability)](https://www.gnu.org/licenses/lgpl-3.0.en.html)

---

_A generic explainability architecture for explaining text machine learning models._

Marcel Robeer, 2021

## Installation
See [installation.md](docs/installation.md) for an extended installation guide.

| Method | Instructions |
|--------|--------------|
| `pip` | Install from [PyPI](https://pypi.org/project/text-explainability/) via `pip3 install text_explainability`. |
| Local | Clone this repository and install via `pip3 install -e .` or locally run `python3 setup.py install`.

## Example usage
See [example_usage.md](example_usage.md) to see an example of how the package can be used, or run the lines in `example_usage.py` to do explore it interactively.

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

## Maintenance
### Contributors
- [Marcel Robeer](https://www.uu.nl/staff/MJRobeer) (`@m.j.robeer`)
- [Michiel Bron](https://www.uu.nl/staff/MPBron) (`@mpbron-phd`)

### Todo
Tasks yet to be done:

- Implement local post-hoc explanations:
    - Implement Anchors
- Implement global post-hoc explanations:
    - Representative subset
- Add support for regression models
- More complex data augmentation
    - Top-k replacement (e.g. according to LM / WordNet)
    - Tokens to exclude from being changed
    - Bag-of-words style replacements
- Add rule-based return type
- Write more tests
