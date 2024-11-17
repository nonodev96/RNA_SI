.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

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
	@echo "Running using $(ENV_PREFIX)"
	@$(ENV_PREFIX)python -V
	@$(ENV_PREFIX)python -m site


.PHONY: virtualenv
virtualenv:       	## Create a virtual environment
	@echo "creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment !!!"


.PHONY: install
install:          	## Install dependencies of the project
	@echo "Don't forget to run 'make virtualenv' if you got errors."
	$(ENV_PREFIX)pip install -r requirements.txt



# =========================
# |    FORMAT AND LINT    |
# =========================

.PHONY: fmt
fmt:			## Format code using black & isort
	$(ENV_PREFIX)isort src/
	$(ENV_PREFIX)black -l 79 src/
	$(ENV_PREFIX)black -l 79 tests/

.PHONY: lint
lint:			## Run pep8, black, mypy linters
	$(ENV_PREFIX)flake8 --ignore=E501,E203,W503 src/
	$(ENV_PREFIX)black -l 79 --check src/
	$(ENV_PREFIX)black -l 79 --check tests/
	$(ENV_PREFIX)mypy --ignore-missing-imports src/

# .PHONY: test
# test:			## Run tests and generate coverage report
# 	$(ENV_PREFIX)pytest -v --cov-config .coveragerc --cov=src/ -l --tb=short --maxfail=1 tests/pytest/ 
# 	$(ENV_PREFIX)coverage html
# 	$(ENV_PREFIX)coverage json
# 	$(ENV_PREFIX)coverage xml
# 	xdg-open .coverage/html/index.html

# .PHONY: test-pytest-unittest
# test-pytest-unittest:	## Run tests and generate coverage report
# 	$(ENV_PREFIX)coverage run -m pytest -v tests/pytest/*
# 	$(ENV_PREFIX)coverage run -m unittest -v tests/utests/*
# # $(ENV_PREFIX)coverage combine
# 	$(ENV_PREFIX)coverage html
# 	$(ENV_PREFIX)coverage json
# 	$(ENV_PREFIX)coverage xml
# 	xdg-open .coverage/html/index.html

# .PHONY: watch
# watch:            	## Run tests on every change
# 	ls **/**.py | entr $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 tests/

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


# .PHONY: release
# release:          	## Create a new tag for release.
# 	@echo "WARNING: This operation will create s version tag and push to github"
# 	@read -p "Version? (provide the next x.y.z semver) : " TAG
# 	@echo "$${TAG}" > src/VERSION
# 	@$(ENV_PREFIX)gitchangelog > HISTORY.md
# 	@git add src/VERSION HISTORY.md
# 	@git commit -m "release: version $${TAG} 🚀"
# 	@echo "creating git tag : $${TAG}"
# 	@git tag $${TAG}
# 	@git push -u origin HEAD --tags
# 	@echo "Github Actions will detect the new tag and release the new version."



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


# # ==============
# # |   MANUAL   |
# # ==============

# .PHONY: manual
# manual: manual-build manual-install manual-read 

# manual-build:    	## Build the manual
# 	@echo "building manual ..."
# 	@pandoc manual/MANUAL.1.md --standalone --mathjax --to man -o manual/man/rna_si.1
# 	@gzip manual/man/rna_si.1


# manual-install:		## Install the manual
# 	@echo "Installing the manual"
# 	@sudo cp manual/man/rna_si.1.gz /usr/share/man/man1/
# 	@sudo mandb


# manual-read:		## Read the manual
# 	@echo "Read manual"
# 	@pandoc manual/MANUAL.1.md -s -t man -o manual/rna_si.1
# 	@man -l manual/rna_si.1