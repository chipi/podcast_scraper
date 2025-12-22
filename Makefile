PYTHON ?= python3
PACKAGE = podcast_scraper

.PHONY: help init format format-check lint lint-markdown type security security-bandit security-audit test coverage docs build ci clean docker-build docker-test docker-clean install-hooks

help:
	@echo "Common developer commands:"
	@echo "  make init            Install development dependencies"
	@echo "  make format          Apply formatting with black + isort"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make lint            Run flake8 linting"
	@echo "  make lint-markdown   Run markdownlint on markdown files"
	@echo "  make type            Run mypy type checks"
	@echo "  make security        Run bandit & pip-audit security scans"
	@echo "  make test            Run pytest with coverage"
	@echo "  make docs            Build MkDocs site (strict mode, outputs to .build/site/)"
	@echo "  make build           Build source and wheel distributions (outputs to .build/dist/)"
	@echo "  make ci              Run the full CI suite locally"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-test     Build and test Docker image"
	@echo "  make docker-clean    Remove Docker test images"
	@echo "  make install-hooks   Install git pre-commit hook for automatic linting"
	@echo "  make clean           Remove build artifacts (.build/, .mypy_cache/, .pytest_cache/)"

init:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e .[dev,ml]
	@if [ -f docs/requirements.txt ]; then $(PYTHON) -m pip install -r docs/requirements.txt; fi

format:
	black .
	isort .

format-check:
	black --check .
	isort --check-only .

lint:
	flake8 --config .flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 --config .flake8 . --count --exit-zero --statistics

lint-markdown:
	@command -v markdownlint >/dev/null 2>&1 || { echo "markdownlint not found. Install with: npm install -g markdownlint-cli"; exit 1; }
	markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site --config .markdownlint.json

type:
	mypy --config-file pyproject.toml .

security: security-bandit security-audit

security-bandit:
	bandit -r . --exclude ./.venv

security-audit:
	$(PYTHON) -m pip install --upgrade setuptools
	pip-audit --requirement requirements.txt --skip-editable || true

docs:
	mkdocs build --strict

test:
	pytest --cov=$(PACKAGE) --cov-report=term-missing

coverage: test

build:
	$(PYTHON) -m pip install --quiet build
	$(PYTHON) -m build
	@if [ -d dist ]; then mkdir -p .build && rm -rf .build/dist && mv dist .build/ && echo "Moved dist to .build/dist/"; fi

ci: format-check lint lint-markdown type security test docs build

docker-build:
	docker build -t podcast-scraper:test -f docker/Dockerfile .

docker-test: docker-build
	@echo "Running Docker smoke tests..."
	@echo "Test 1: --help command"
	@docker run --rm podcast-scraper:test --help > /dev/null
	@echo "Test 2: --version command"
	@docker run --rm podcast-scraper:test --version
	@echo "Test 3: No args (should error)"
	@docker run --rm podcast-scraper:test 2>&1 | grep -q "required" && echo "[OK] Error handling works"
	@echo "Test 4: Building with multiple Whisper models..."
	@docker build --quiet --build-arg WHISPER_PRELOAD_MODELS="tiny.en,base.en" -t podcast-scraper:multi-model -f docker/Dockerfile . > /dev/null
	@docker run --rm podcast-scraper:multi-model --help > /dev/null
	@echo "[OK] All Docker tests passed"

docker-clean:
	docker rmi podcast-scraper:test podcast-scraper:multi-model 2>/dev/null || true

install-hooks:
	@if [ ! -d .git ]; then echo "Error: Not a git repository"; exit 1; fi
	@echo "Installing git pre-commit hook..."
	@cp .github/hooks/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✓ Pre-commit hook installed!"
	@echo ""
	@echo "The hook will automatically run before each commit to check:"
	@echo "  • Black & isort formatting"
	@echo "  • flake8 linting"
	@echo "  • markdownlint (if installed)"
	@echo "  • mypy type checking"
	@echo ""
	@echo "To skip the hook for a specific commit, use: git commit --no-verify"

clean:
	rm -rf build .build .mypy_cache .pytest_cache
