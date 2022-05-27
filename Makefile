.PHONY: docs html coverage quality

package := text_explainability
check_dirs := $(package)
docs_dir := docs
build_dir := $(docs_dir)/build
source_dir := $(docs_dir)/source

# Build documentation files
docs:
	cp img/TextLogo.png $(source_dir)/_static
	sphinx-apidoc --module-first --no-toc --force --templatedir=$(source_dir)/_templates/ -o $(source_dir)/api .
	m2r CHANGELOG.md --dry-run > $(source_dir)/changelog.rst
	m2r example_usage.md --dry-run > $(source_dir)/example-usage.rst
	m2r INSTALLATION.md --dry-run > $(source_dir)/installation.rst

# Convert docs to HTML
html:
	sphinx-build -M clean $(source_dir) $(build_dir)
	sphinx-build -M html $(source_dir) $(build_dir)

# Code style quality
quality:
	flake8 --config .flake8 text_explainability
	python3 -m isort --line-length 120 --check-only -diff text_explainability

# Coverage
coverage:
	coverage run -m pytest
	coverage html
	open htmlcov/index.html
