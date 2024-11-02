.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
USING_POETRY=$(shell grep "tool.poetry" pyproject.toml && echo "yes")

# ========================
# |     INFO PROJECT     |
# ========================

.PHONY: help
help:             	## Show the help
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: show
show:             	## Show the current environment
	@echo "Current environment:"
	@if [ "$(USING_POETRY)" ]; then poetry env info && exit; fi
	@echo "Running using $(ENV_PREFIX)"
	@$(ENV_PREFIX)python -V
	@$(ENV_PREFIX)python -m site


.PHONY: virtualenv
virtualenv:       	## Create a virtual environment
	@if [ "$(USING_POETRY)" ]; then poetry install && exit; fi
	@echo "creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@./.venv/bin/pip install -e .[test]
	@echo
	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment !!!"


.PHONY: install
install:          	## Install dependencies of the project
	@if [ "$(USING_POETRY)" ]; then poetry install && exit; fi
	@echo "Don't forget to run 'make virtualenv' if you got errors."
	$(ENV_PREFIX)pip install -e .[test]



# =========================
# |    FORMAT AND LINT    |
# =========================

.PHONY: fmt
fmt:              	## Format code using black & isort
	$(ENV_PREFIX)isort src/rna_si/
	$(ENV_PREFIX)black -l 79 src/rna_si/
	$(ENV_PREFIX)black -l 79 tests/

.PHONY: lint
lint:             	## Run pep8, black, mypy linters
	$(ENV_PREFIX)flake8 --ignore=E501,E203,W503 src/rna_si/
	$(ENV_PREFIX)black -l 79 --check src/rna_si/
	$(ENV_PREFIX)black -l 79 --check tests/
	$(ENV_PREFIX)mypy --ignore-missing-imports src/rna_si/

.PHONY: test
test:         	  	## Run tests and generate coverage report
	$(ENV_PREFIX)pytest -v --cov-config .coveragerc --cov=src/rna_si/ -l --tb=short --maxfail=1 tests/pytest/ 
	$(ENV_PREFIX)coverage html
	$(ENV_PREFIX)coverage json
	$(ENV_PREFIX)coverage xml
	xdg-open .coverage/html/index.html

.PHONY: test-pytest-unittest
test-pytest-unittest:         	  	## Run tests and generate coverage report
	$(ENV_PREFIX)coverage run -m pytest -v tests/pytest/*
	$(ENV_PREFIX)coverage run -m unittest -v tests/utests/*
# $(ENV_PREFIX)coverage combine
	$(ENV_PREFIX)coverage html
	$(ENV_PREFIX)coverage json
	$(ENV_PREFIX)coverage xml
	xdg-open .coverage/html/index.html

.PHONY: watch
watch:            	## Run tests on every change
	ls **/**.py | entr $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 tests/

.PHONY: clean
clean:            	## Clean unused files
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf .coverage/
	@rm -rf .tox/
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf docs/_build


.PHONY: release
release:          	## Create a new tag for release.
	@echo "WARNING: This operation will create s version tag and push to github"
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@echo "$${TAG}" > src/rna_si/VERSION
	@$(ENV_PREFIX)gitchangelog > HISTORY.md
	@git add src/rna_si/VERSION HISTORY.md
	@git commit -m "release: version $${TAG} 🚀"
	@echo "creating git tag : $${TAG}"
	@git tag $${TAG}
	@git push -u origin HEAD --tags
	@echo "Github Actions will detect the new tag and release the new version."



# ==============
# |    DOCS    |
# ==============

.PHONY: docs-serve
docs-serve:    	  	## Enable the serve of mkdocs
	@echo "building documentation ..."
	@$(ENV_PREFIX)mkdocs serve


.PHONY: docs
docs:             	## Open the documentation.
	@echo "building documentation ..."
	@$(ENV_PREFIX)mkdocs build
	URL="site/index.html"; xdg-open $$URL || sensible-browser $$URL || x-www-browser $$URL || gnome-open $$URL || open $$URL



# TODO
.PHONY: switch-to-poetry
switch-to-poetry: 	## Switch to poetry package manager.
	@echo "Switching to poetry ..."
	@if ! poetry --version > /dev/null; then echo 'poetry is required, install from https://python-poetry.org/'; exit 1; fi
	@rm -rf .venv
	@poetry init --no-interaction --name=a_flask_test --author=rochacbruno
	@echo "" >> pyproject.toml
	@echo "[tool.poetry.scripts]" >> pyproject.toml
	@echo "rna_si = 'rna_si.__main__:main'" >> pyproject.toml
	@cat requirements.txt | while read in; do poetry add --no-interaction "$${in}"; done
	@cat requirements-test.txt | while read in; do poetry add --no-interaction "$${in}" --dev; done
	@poetry install --no-interaction
	@mkdir -p .github/backup
	@mv requirements* .github/backup
	@mv setup.py .github/backup
	@echo "You have switched to https://python-poetry.org/ package manager."
	@echo "Please run 'poetry shell' or 'poetry run rna_si'"



# ==============
# |   MANUAL   |
# ==============

.PHONY: manual
manual: manual-build manual-install manual-read 

manual-build:    	## Build the manual
	@echo "building manual ..."
	@pandoc manual/MANUAL.1.md --standalone --mathjax --to man -o manual/man/rna_si.1
	@gzip manual/man/rna_si.1


manual-install:		## Install the manual
	@echo "Installing the manual"
	@sudo cp manual/man/rna_si.1.gz /usr/share/man/man1/
	@sudo mandb


manual-read:		## Read the manual
	@echo "Read manual"
	@pandoc manual/MANUAL.1.md -s -t man -o manual/rna_si.1
	@man -l manual/rna_si.1