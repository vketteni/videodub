.PHONY: install test lint format type-check clean clean-output clean-all dev-install build docs help

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
	@echo "  clean-output - Clean all pipeline output files"
	@echo "  clean-all   - Clean everything (build + output)"
	@echo "  docs        - Generate documentation"

# Installation
install:
	poetry install --only main

dev-install:
	poetry install

# Testing
test:
	poetry run pytest tests/ -v --cov=src/videodub --cov-report=term-missing --cov-report=html

test-fast:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-performance:
	poetry run pytest tests/integration/ -m performance -v

test-performance-fast:
	poetry run pytest tests/integration/ -m "performance and fast" -v

test-performance-slow:
	poetry run pytest tests/integration/ -m "performance and slow" -v

# Code quality
lint:
	poetry run ruff check src/ tests/
	poetry run ruff check examples/ scripts/

format:
	poetry run black src/ tests/ examples/ scripts/
	poetry run isort src/ tests/ examples/ scripts/

format-check:
	poetry run black --check src/ tests/ examples/ scripts/
	poetry run isort --check-only src/ tests/ examples/ scripts/

type-check:
	poetry run mypy src/videodub

# Development
dev: dev-install lint type-check test

# Building
build:
	poetry build

clean:
	@echo "🧹 Cleaning build artifacts..."
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
	@echo "✅ Build artifacts cleaned"

clean-output:
	@echo "🗑️  Cleaning all pipeline output files..."
	@echo "Removing output directories..."
	rm -rf ./output/
	rm -rf ./quick_test_output/
	rm -rf ./example_output/
	rm -rf ./multi_lang_output/
	rm -rf ./pipeline_output/
	@echo "Removing old output in project root..."
	rm -rf ./old/pipeline_output/
	@echo "Removing any test output directories..."
	find . -name "*_output" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "scraped" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ All output files cleaned"

clean-all: clean clean-output
	@echo "🧽 Deep clean completed - all artifacts and outputs removed"

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Docker Commands
docker-build:
	docker compose build

docker-setup: docker-build
	@echo "🐳 Setting up Docker environment..."
	@if [ ! -f .env ]; then \
		echo "📋 Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env with your API keys"; \
	fi
	@echo "✅ Docker setup complete"

docker-dev:
	docker compose --profile dev up -d videodub-dev
	@echo "🚀 Development container started"
	@echo "💡 Run: docker compose exec videodub-dev bash"

docker-test:
	docker compose --profile test up videodub-test

docker-prod:
	docker compose --profile prod up -d videodub-prod

docker-run:
	docker compose run --rm videodub

docker-shell:
	docker compose exec videodub bash

docker-logs:
	docker compose logs -f

docker-down:
	docker compose down

docker-clean:
	docker compose down -v --remove-orphans
	docker system prune -f

docker-rebuild: docker-clean docker-build

# Quick Docker workflows
docker-quick-test: docker-build
	docker compose run --rm videodub python examples/quick_test.py

docker-quick-setup: docker-setup docker-dev
	@echo "🎉 Quick Docker setup complete!"
	@echo "🔗 Connect: docker compose exec videodub-dev bash"

# Environment setup
env-create:
	@echo "Poetry manages virtual environments automatically"
	@echo "To activate: poetry shell"
	@echo "To run commands: poetry run <command>"

env-requirements:
	poetry export -f requirements.txt --output requirements/current.txt

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