.PHONY: install test lint format type-check clean dev-install build docs help

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install package in development mode"
	@echo "  dev-install - Install with development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  type-check  - Run type checking"
	@echo "  build       - Build package"
	@echo "  clean       - Clean build artifacts"
	@echo "  docs        - Generate documentation"

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pip install -r requirements/dev.txt

# Testing
test:
	pytest tests/ -v --cov=src/youtube_translator --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	ruff check src/ tests/
	ruff check examples/ scripts/

format:
	black src/ tests/ examples/ scripts/
	isort src/ tests/ examples/ scripts/

format-check:
	black --check src/ tests/ examples/ scripts/
	isort --check-only src/ tests/ examples/ scripts/

type-check:
	mypy src/youtube_translator

# Development
dev: dev-install lint type-check test

# Building
build:
	python -m build

clean:
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.orig" -delete
	find . -name "*.rej" -delete
	find . -name "*.log" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Docker (for future use)
docker-build:
	docker build -t youtube-translator .

docker-run:
	docker run -it --rm youtube-translator

# Environment setup
env-create:
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

env-requirements:
	pip freeze > requirements/current.txt

# Health checks
health-check:
	python scripts/health_check.py

# Example runs
example-basic:
	python examples/basic_usage.py

example-batch:
	python examples/batch_processing.py

# Migration from old structure
migrate-old:
	python scripts/migrate_existing_data.py

# Performance testing
perf-test:
	@echo "Performance testing not yet implemented"

# Security checks
security:
	@echo "Security checks not yet implemented"

# All quality checks
check: format-check lint type-check

# Full development workflow
all: clean dev-install check test build