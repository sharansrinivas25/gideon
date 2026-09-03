.PHONY: install test train generate bench figures clean all

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

train:
	python -m gideon.train --n-layer 6 --n-head 6 --n-embd 192 \
		--block-size 128 --batch-size 32 --max-iters 3000 --lr 1e-3

generate:
	python -m gideon.generate --ckpt results/ckpt.pt --tokens 500 --temperature 0.8

bench:
	python -m gideon.benchmark --ckpt results/ckpt.pt --out results/benchmarks.json

figures:
	python scripts/make_figures.py

all: test train bench figures

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache *.egg-info build dist
