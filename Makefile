setup:
	python -m venv .venv
	.venv\Scripts\Activate.ps1


install:	
	python -m pip install --upgrade pip setuptools wheel
	pip install poetry

test:
	pytest	


git-status:
	git add .
	git status


git-push:
	git commit -m "update"
	git push 	