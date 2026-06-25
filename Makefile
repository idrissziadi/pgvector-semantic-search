.PHONY: install db-up db-down ingest demo test test-cov lint eval

install:
	pip install -r requirements.txt

db-up:
	docker-compose up -d

db-down:
	docker-compose down

ingest:
	python scripts/ingest_pipeline.py

demo:
	python scripts/demo_search.py

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100

eval:
	python src/evaluation.py
