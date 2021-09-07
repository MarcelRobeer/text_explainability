<p align="center">
  <img src="https://i.ibb.co/xXtJ23n/Text-Logo-Logo-large.png">
</p>

_A generic explainability architecture for explaining text machine learning models._

[![PyPI](https://img.shields.io/pypi/v/text_explainability)](https://pypi.org/project/text-explainability/)
[![Python_version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)](https://pypi.org/project/text-explainability/)
[![Build_passing](https://img.shields.io/badge/build-passing-brightgreen)](https://git.science.uu.nl/m.j.robeer/text_explainability/-/pipelines)
[![License](https://img.shields.io/pypi/l/text_explainability)](https://www.gnu.org/licenses/lgpl-3.0.en.html)

Marcel Robeer, 2021

## Installation
| Method | Instructions |
|--------|--------------|
| `pip` | Install from PyPI via `pip3 install text_explainability`. |
| Local | Clone this repository and install via `pip3 install -e .` or locally run `python3 setup.py install`.

## Example usage
Run lines in `example_usage.py` to see an example of how the package can be used.

## Releases
`text_explainability` is officially released through [PyPI](https://pypi.org/project/text-explainability/).

See [CHANGELOG.md](CHANGELOG.md) for a full overview of the changes for each version.

## Maintenance
### Contributors
- [Marcel Robeer](https://www.uu.nl/staff/MJRobeer) (`@m.j.robeer`)
- [Michiel Bron](https://www.uu.nl/staff/MPBron) (`@mpbron-phd`)

### Todo
Tasks yet to be done:
- Add data sampling methods (e.g. representative subset, prototypes, MMD-critic)
- Implement local post-hoc explanations:
    - Implement Anchors
- Implement global post-hoc explanations
- Add support for regression models
- More complex data augmentation
    - Top-k replacement (e.g. according to LM / WordNet)
    - Tokens to exclude from being changed
    - Bag-of-words style replacements
- Add rule-based return type
- Write more tests
