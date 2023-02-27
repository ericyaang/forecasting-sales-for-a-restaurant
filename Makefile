setup:
	@echo "Creating virtual environment"
	python -m venv .venv
	.venv\Scripts\Activate.ps1


install:
	@echo "Activating virtual environment"
	python -m pip install --upgrade pip setuptools wheel
	pip install poetry

install_poetry:
	@echo "Installing poetry..."
	poetry install
	poetry run pre-commit install

test:
	pytest


git-status:
	git add .
	git status


git-push:
	git commit -m "update"
	git push
