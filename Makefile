.DEFAULT_GOAL := help


PACKAGE_PATH="jax2pytensor"

SPHINXOPTS    =
SPHINXBUILD   = python-msphinx
SPHINXPROJ    = jax2pytensor
SOURCEDIR     = docs/
BUILDDIR      = docs/_build

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY:help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY:clean
clean: ## remove build artifacts, compiled files, and cache
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {}  +
	find . -name '*~' -exec rm -f {} +
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

.PHONY:lint
lint: ## lint the project
	pre-commit run --all-files

.PHONY: test
test: ## run tests quickly with the default Python
	pytest

.PHONY:docs-build
docs-build:
	sphinx-apidoc -o docs/_build ${PACKAGE_PATH}
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: docs-preview
docs-preview: docs-build
	cd docs/_build && python -m http.server

.PHONY:build
build:
	python -m build

