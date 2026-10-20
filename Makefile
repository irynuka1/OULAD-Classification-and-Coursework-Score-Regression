PY ?= python
export PYTHONPATH := src

.PHONY: eda dev submit all clean

eda:      ## EDA tables and figures
	$(PY) -m oauld eda

dev:      ## fit on train, evaluate on the labelled hold-out (CA_OUALD_test)
	$(PY) -m oauld all --mode dev

submit:   ## fit on train + hold-out, write predictions for CA_private_test
	$(PY) -m oauld all --mode submit

all: dev submit

clean:
	rm -rf catboost_info reports/metrics/* reports/figures/classification/* reports/figures/regression/*
