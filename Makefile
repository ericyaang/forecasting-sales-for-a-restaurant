setup:
	python -m venv .venv
	.venv\Scripts\Activate.ps1


install:	
	python -m pip install --upgrade pip setuptools wheel
	pip install poetry

test:
	pytest	


git:
	git add .
	git status