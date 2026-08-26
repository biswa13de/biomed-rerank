.PHONY: venv install test data retrieve rerank eval all clean

VENV := .venv/bin/python

venv:
	uv venv --python 3.11 .venv

install: venv
	uv pip install --python $(VENV) -r requirements.txt

test:
	$(VENV) -m pytest tests/ -v

data:
	$(VENV) scripts/download_data.py
	$(VENV) -m src.data_prep

retrieve:
	$(VENV) -m src.retrieval.bm25
	$(VENV) -m src.retrieval.dense
	$(VENV) -m src.retrieval.hybrid

rerank:
	$(VENV) -m src.rerank.cross_encoder

eval:
	$(VENV) -m src.eval.metrics

all: test data retrieve rerank eval

clean:
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/__pycache__
