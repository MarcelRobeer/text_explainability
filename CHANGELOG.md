# Changelog
All notable changes to `text_explainability` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.1] - 2022-03-16
### Changed
- Requires `genbase>=0.2.4`
- Requires `instancelib>=0.4.3.1`

### Fixed
- Typo fixes and small bugs

## [0.6.0] - 2022-03-04
### Added
- More tests to increase test coverage

### Changed
- Requires `genbase>=0.2.2`
- Renamed `pyproject.toml` to `.portray` to avoid build errors
- Made `fastcountvectorizer` optional

### Fixed
- Bugfix when installing package, by moving `__version___` to `/_version.py`

## [0.5.8] - 2021-12-02
### Added
- `get_meta_descriptors()` to get type/subtype/method from meta

### Changed
- Requires `genbase>=0.1.13`

### Fixed
- Bugfix in `MMDCritic` for prototype indices
- Bugfix in `TRANSLATION_DICT`

## [0.5.7] - 2021-12-01
### Added
- Return type for `Instances`
- Rendering of `Instances`
- Rendering of `FeatureList`
- Extended rendering of `render_subtitle()`

### Changed
- Ensure `MMDCritic`/`KMedoids` returns `Instances`
- Requires `genbase>=0.1.11`

### Fixed
- Bugfix of instance identifier in `PrototypeSampler._select_from_provider()`

## [0.5.6] - 2021-11-30
### Added
- Added meta information with `genbase.MetaInfo`
- Rendering with and extended `genbase.Render`

### Changed
- Moved `Readable` to `genbase`
- Use `genbase.SeedMixin` for seeds
- Use `genbase.internationalization` for internationalization
- Requires `genbase>=0.1.10`

### Fixed
- Selected features are in order in `FeatureList`

## [0.5.5] - 2021-11-17
### Changed
- `TokenFrequency` and `TokenInformation` now use the faster `fastcountvectorizer` implementation

### Fixed
- Bugfixes in return type of `TokenFrequency` and `TokenInformation`

## [0.5.4] - 2021-10-27
### Fixed
- Bugfixes in local explanation return types

## [0.5.3] - 2021-10-19
### Fixed
- Made `alpha` optional in `LinearSurrogate`
- Added `skope-rules` dependency to `setup.py`

## [0.5.2] - 2021-10-05
### Fixed
- Hotfix in `FeatureSelector._information_criterion()`

## [0.5.1] - 2021-10-05
### Added
- Added `text_explainability.data.from_list`

### Changed
- Added example results in README.md

### Fixed
- Added new methods and classes to `__init__.py`

## [0.5.0] - 2021-10-04
### Added
- Security testing with bandit
- More locale translations
- Wrappers around `instancelib` in `text_explainability.data` and `text_explainability.model`

### Changed
- Extended description in README.md
- Changed example usage to fit workflow changes
- Logo link in README.md

### Fixed
- Bugfixes in MMDCritic
- Bugfixes in KernelSHAP

## [0.4.6] - 2021-10-02
### Added
- External documentation
- Documentation styling
- Citation information

### Changed
- Word tokenizer can now combine tokens in curly bracket when setting `exclude_curly_brackets=True`

## [0.4.5] - 2021-09-24
### Added
- Decorator to allow strings to be converted into TextInstances
- Decorator to ensure TextInstances are tokenized when required

### Fixed
- Typing fixes

## [0.4.4] - 2021-09-23
### Added
- Character-level tokenizer/detokenizer

## [0.4.3] - 2021-09-20
### Added
- New embeddings not requiring internet (`CountVectorizer`, `TfidfVectorizer`)
- `Rules` return type
- First version of local rules using `SkopeRules`
- More test cases

### Changed
- New default embedding method for `MMDCritic` and `KMedoids`
- Version moved to `__init__.py`
- New README.md layout
- Updates to Anchor local explanations
- Added random state in example_usage to ensure reproducibility

## [0.4.2] - 2021-09-13
### Fixed
- Hotfix to fix `predict_proba` usage

## [0.4.1] - 2021-09-13
### Fixed
- Hotfix to make dependency on internet optional

## [0.4.0] - 2021-09-13
### Added
- Initial support for embeddings/vectors
- Support for dimensionality reduction
- Initial implementation of MMD-Critic
- Initial implementation of labelwise MMD-Critic
- Initial implementation of prototype selection using k-Medoids

### Changed
- Updated README.md

## [0.3.8] - 2021-09-07
### Changed
- Support for dimensionality reduction

### Fixed
- Bugfix in including `locale/*.json` files during setup

## [0.3.7] - 2021-09-07
### Added
- Dependencies for package

## [0.3.6] - 2021-09-07
### Added
- PyPI release script to .gitignore
- Badges to README.md
- Added dependencies to `setup.py`

## [0.3.5] - 2021-09-03
### Changed
- Locale changed to .json format, to remove optional dependency

### Fixed
- Bugfix for getting key in TokenFrequency
- Bugfixes in FeatureAttribution return type
- Bugfixes in `i18n`

## [0.3.4] - 2021-08-18
### Changed
- External logo url

### Fixed
- Hotfix in FeatureAttribution

## [0.3.3] - 2021-08-18
### Added
- Updated to support `instancelib==0.3.1.2`
- `i18n` internationalization support
- CHANGELOG.md

### Changed
- Additional samples in example dataset

### Fixed
- Bugfixes for LIME and FeatureAttribution return type

## [0.3.2] - 2021-07-27
### Added
- Initial support for [`Foil Trees`](https://github.com/MarcelRobeer/ContrastiveExplanation)
- Logo in documentation

### Changed
- Improved documentation

## [0.3.1] - 2021-07-23
### Added
- `flake8` linting
- CI/CD Pipeline
- Run test scripts

## [0.3.0] - 2021-07-20
### Added
- Updated to support `instancelib==0.3.0.0`

### Changed
- Improved documentation
- `global_explanation` classes have equal return types

## [0.2] - 2021-06-22
### Added
- LICENSE.md
- Updated to support `instancelib==0.2.3.1`

### Changed
- Module description

## [0.1] - 2021-05-28
### Added
- README.md
- Example usage
- Local explanation classes (LIME, KernelSHAP)
- Global explanation classes
- Data augmentation/sampling
- Feature selection
- Local surrogates
- Tokenization
- `git` setup


[Unreleased]: https://git.science.uu.nl/m.j.robeer/text_explainability
[0.6.1]: https://pypi.org/project/text-explainability/0.6.1/
[0.6.0]: https://pypi.org/project/text-explainability/0.6.0/
[0.5.8]: https://pypi.org/project/text-explainability/0.5.8/
[0.5.7]: https://pypi.org/project/text-explainability/0.5.7/
[0.5.6]: https://pypi.org/project/text-explainability/0.5.6/
[0.5.5]: https://pypi.org/project/text-explainability/0.5.5/
[0.5.4]: https://pypi.org/project/text-explainability/0.5.4/
[0.5.3]: https://pypi.org/project/text-explainability/0.5.3/
[0.5.2]: https://pypi.org/project/text-explainability/0.5.2/
[0.5.1]: https://pypi.org/project/text-explainability/0.5.1/
[0.5.0]: https://pypi.org/project/text-explainability/0.5.0/
[0.4.6]: https://pypi.org/project/text-explainability/0.4.6/
[0.4.5]: https://pypi.org/project/text-explainability/0.4.5/
[0.4.4]: https://pypi.org/project/text-explainability/0.4.4/
[0.4.3]: https://pypi.org/project/text-explainability/0.4.3/
[0.4.2]: https://pypi.org/project/text-explainability/0.4.2/
[0.4.1]: https://pypi.org/project/text-explainability/0.4.1/
[0.4.0]: https://pypi.org/project/text-explainability/0.4.0/
[0.3.8]: https://pypi.org/project/text-explainability/0.3.8/
[0.3.7]: https://pypi.org/project/text-explainability/0.3.7/
[0.3.6]: https://pypi.org/project/text-explainability/0.3.6/
[0.3.5]: https://pypi.org/project/text-explainability/0.3.5/
[0.3.4]: https://pypi.org/project/text-explainability/0.3.4/
[0.3.3]: https://pypi.org/project/text-explainability/0.3.3/
[0.3.2]: https://pypi.org/project/text-explainability/0.3.2/
[0.3.1]: https://pypi.org/project/text-explainability/0.3.1/
[0.3.0]: https://pypi.org/project/text-explainability/0.3.0/
[0.2]: https://pypi.org/project/text-explainability/0.2/
[0.1]: https://pypi.org/project/text-explainability/0.1/
