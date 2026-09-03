.PHONY: venv install data data-quick eda test clean

VENV := .venv
PYTHON := $(VENV)/bin/python

venv:
	python3 -m venv $(VENV)

install: venv
	$(VENV)/bin/pip install --upgrade pip -q
	$(VENV)/bin/pip install -r requirements.txt -q

data:
	$(PYTHON) -m src.data_generation.generate_dataset

data-quick:
	$(PYTHON) -m src.data_generation.generate_dataset --quick

eda:
	$(VENV)/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb --ExecutePreprocessor.timeout=180

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
