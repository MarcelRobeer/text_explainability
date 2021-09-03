# Changelog
All notable changes to `text_explainability` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.5] - 2021-09-03
### Changed
- Bugfix for getting key in TokenFrequency
- Locale changed to .json format, to remove optional dependency
- Bugfixes in FeatureAttribution return type
- Bugfixes in `i18n`

## [0.3.4] - 2021-08-18
### Changed
- External logo url
- Hotfix in FeatureAttribution

## [0.3.3] - 2021-08-18
### Added
- Updated to support `instancelib==0.3.1.2`
- `i18n` internationalization support
- CHANGELOG.md

### Changed
- Additional samples in example dataset
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


[unreleased]: https://git.science.uu.nl/m.j.robeer/text_explainability
[0.3.5]: https://pypi.org/project/text-explainability/0.3.5/
[0.3.4]: https://pypi.org/project/text-explainability/0.3.4/
[0.3.3]: https://pypi.org/project/text-explainability/0.3.3/
[0.3.2]: https://pypi.org/project/text-explainability/0.3.2/
[0.3.1]: https://pypi.org/project/text-explainability/0.3.1/
[0.3.0]: https://pypi.org/project/text-explainability/0.3.0/
[0.2]: https://pypi.org/project/text-explainability/0.2/
[0.1]: https://pypi.org/project/text-explainability/0.1/
