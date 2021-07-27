<p align="center">
  <img src="https://git.science.uu.nl/m.j.robeer/text_explainability/-/raw/main/img/TextLogo-Logo%20large.png">
</p>

_A generic explainability architecture for explaining text machine learning models._

Marcel Robeer, 2021

## Installation
Install from PyPI via `pip3 install text_explainability`. Alternatively, clone this repository and install via `pip3 install -e .` or locally run `python3 setup.py install`.

## Example usage
Run lines in `example_usage.py` to see an example of how the package can be used.

## Maintenance
### Contributors
- Marcel Robeer (`@m.j.robeer`)
- Michiel Bron (`@mpbron-phd`)

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
