setup:
	@echo "Creating virtual environment"
	python -m venv .venv
	powershell -noexit -executionpolicy bypass .venv/Scripts/activate.ps1


install:
	@echo "Installing poetry, pip, setuptools, wheel"
	python -m pip install --upgrade pip setuptools wheel
	pip install poetry

install_poetry:
	@echo "Installing poetry..."
	poetry install
	poetry run pre-commit install

install_all: setup install install_poetry

activate:
	powershell -noexit -executionpolicy bypass .venv/Scripts/activate.ps1

test:
	pytest


git-status:
	git add .
	git status


git-push:
	git commit -m "update"
	git push

process: 
	@echo "Processing data..."
	python src/process.py

select_reg:
	@echo "Training regressors with CV"
	python src/select_regressor.py

train_hyperopt:
	@echo "Evaluating hyperopt with CV"
	python src/train_hyperopt.py

train_model:
	@echo "Training model"
	python src/train_model.py

pipeline: process select_reg train_hyperopt train_model

	


