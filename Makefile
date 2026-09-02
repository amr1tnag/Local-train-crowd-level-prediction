# Mumbai Harbour Line crowd-level prediction
PY ?= python3

.PHONY: help install data co2 co5 all quick test clean

help:
	@echo "make install   install dependencies"
	@echo "make data      simulate 180 days of Harbour-line operations"
	@echo "make co2       train the asymmetric-loss regressors (CO2)"
	@echo "make co5       cluster the station profiles (CO5)"
	@echo "make all       data + co2 + co5"
	@echo "make quick     a 45-day version of the above, for a laptop"
	@echo "make test      run the test suite"
	@echo "make clean     delete generated data, figures and tables"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/01_generate_data.py --days 180 --monitored 0.08

co2:
	$(PY) scripts/02_train_regression.py

co5:
	$(PY) scripts/03_cluster_stations.py --k 4

all:
	$(PY) scripts/run_all.py

quick:
	$(PY) scripts/run_all.py --quick

test:
	$(PY) -m pytest

clean:
	rm -f data/*.csv data/*.csv.gz reports/figures/*.png reports/tables/*.csv reports/tables/*.json
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
