PYTHON ?= python3
PACKAGE = podcast_scraper

.PHONY: help init init-no-ml format format-check lint lint-markdown type security security-bandit security-audit test-unit test-unit-sequential test-unit-no-ml test-integration test-integration-sequential test-integration-fast test-ci test-ci-fast test-e2e test-e2e-sequential test-e2e-fast test-e2e-data-quality test test-sequential test-fast test-reruns coverage docs build ci ci-fast ci-sequential ci-clean clean clean-cache clean-all docker-build docker-build-fast docker-build-full docker-test docker-clean install-hooks preload-ml-models repair-ml-cache

help:
	@echo "Common developer commands:"
	@echo "  make init            Install development dependencies"
	@echo "  make format          Apply formatting with black + isort"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make lint            Run flake8 linting"
	@echo "  make lint-markdown   Run markdownlint on markdown files"
	@echo "  make fix-md          Auto-fix common markdown linting issues"
	@echo "  make type            Run mypy type checks"
	@echo "  make security        Run bandit & pip-audit security scans"
	@echo ""
	@echo "Test commands:"
	@echo "  make test-unit            Run unit tests with coverage in parallel (default, matches CI)"
	@echo "  make test-unit-sequential Run unit tests sequentially (for debugging, slower but clearer output)"
	@echo "  make test-unit-no-ml Run unit tests without ML dependencies (matches CI)"
	@echo "  make test-integration            Run all integration tests (full suite, parallel)"
	@echo "  make test-integration-sequential Run all integration tests sequentially (for debugging)"
	@echo "  make test-integration-fast       Run fast integration tests (critical path only)"
	@echo "  make test-e2e                   Run all E2E tests (full suite, parallel, 1 episode per test)"
	@echo "  make test-e2e-sequential         Run all E2E tests sequentially (for debugging)"
	@echo "  make test-e2e-fast              Run fast E2E tests (critical path only, 1 episode per test)"
	@echo "  make test-e2e-data-quality      Run data quality E2E tests (multiple episodes, all original mock data, nightly only)"
	@echo "  make test                Run all tests (unit + integration + e2e, full suite, uses multi-episode feed)"
	@echo "  make test-sequential     Run all tests sequentially (for debugging, uses multi-episode feed)"
	@echo "  make test-fast           Run fast tests (unit + critical path integration + critical path e2e, uses fast feed)"
	@echo "  make test-reruns     Run tests with reruns for flaky tests (2 retries, 1s delay)"
	@echo ""
	@echo "Other commands:"
	@echo "  make docs            Build MkDocs site (strict mode, outputs to .build/site/)"
	@echo "  make build           Build source and wheel distributions (outputs to .build/dist/)"
	@echo "  make ci              Run the full CI suite locally (all tests: unit + integration + e2e, uses multi-episode feed)"
	@echo "  make ci-fast         Run fast CI checks (unit + critical path integration + critical path e2e, uses fast feed)"
	@echo "  make ci-sequential   Run the full CI suite sequentially (all tests, slower but clearer output)"
	@echo "  make ci-clean        Run complete CI suite with clean first (same as ci but cleans build artifacts first)"
	@echo "  make docker-build       Build Docker image (default, with model preloading)"
	@echo "  make docker-build-fast  Build Docker image fast (no model preloading, <5min target)"
	@echo "  make docker-build-full  Build Docker image full (with model preloading, matches main)"
	@echo "  make docker-test        Build and test Docker image"
	@echo "  make docker-clean       Remove Docker test images"
	@echo "  make install-hooks   Install git pre-commit hook for automatic linting"
	@echo "  make clean           Remove build artifacts (.build/, .mypy_cache/, .pytest_cache/)"
	@echo "  make clean-cache     Remove ML model caches (Whisper, spaCy) to test network isolation"
	@echo "  make clean-all       Remove both build artifacts and ML model caches"
	@echo "  make preload-ml-models  Pre-download and cache all required ML models locally"
	@echo "  make repair-ml-cache Check and auto-repair corrupted ML model caches"

init:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e .[dev,ml]
	@if [ -f docs/requirements.txt ]; then $(PYTHON) -m pip install -r docs/requirements.txt; fi

init-no-ml:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e .[dev]

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
	markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site --ignore "docs/wip/**" --ignore "tests/fixtures/**" --config .markdownlint.json

fix-md:
	@echo "Fixing common markdown linting issues..."
	@python scripts/fix_markdown.py
	@echo "✓ Markdown fixes applied. Run 'make lint-markdown' to verify."

type:
	mypy --config-file pyproject.toml .

security: security-bandit security-audit

security-bandit:
	bandit -r . --exclude ./.venv --skip B113,B108,B110,B310 --severity-level medium

security-audit:
	$(PYTHON) -m pip install --upgrade setuptools
	# Install ML dependencies to ensure they are audited
	# This ensures production dependencies like torch, transformers, spacy, openai-whisper are audited
	$(PYTHON) -m pip install --quiet -e .[ml] || \
		(echo "⚠️  Editable install failed, using non-editable install" && \
		 $(PYTHON) -m pip install --quiet .[ml])
	# Audit all installed packages (including ML dependencies from pyproject.toml)
	pip-audit --skip-editable

docs:
	mkdocs build --strict

test-unit:
	# Unit tests: parallel execution for faster feedback
	pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n auto

test-unit-sequential:
	# Unit tests: sequential execution (slower but clearer output, useful for debugging)
	pytest --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

test-unit-no-ml: init-no-ml
	@echo "Running unit tests without ML dependencies (matches CI test-unit job)..."
	@echo "This verifies that unit tests work when ML dependencies (spacy, torch) are not installed."
	@echo ""
	@echo "Step 1: Checking if modules can be imported without ML dependencies..."
	@$(PYTHON) scripts/check_unit_test_imports.py
	@echo ""
	@echo "Step 2: Running unit tests..."
	pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

test-integration:
	# Integration tests: parallel execution (3.4x faster, significant benefit)
	# Includes reruns for flaky tests (matches CI behavior)
	pytest tests/integration/ -m integration -n auto --reruns 2 --reruns-delay 1

test-integration-sequential:
	# Integration tests: sequential execution (slower but clearer output, useful for debugging)
	pytest tests/integration/ -m integration

test-integration-fast:
	# Fast integration tests: critical path tests only (includes ML tests if models are cached)
	# Includes reruns for flaky tests (matches CI behavior)
	# Excludes slow tests even if marked critical_path (timeout/retry tests run in full suite)
	pytest tests/integration/ -m "integration and critical_path and not slow" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

test-ci:
	# CI test suite: serial tests first (sequentially), then parallel execution for the rest
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Note: Non-critical path tests run on main branch only
	pytest -m 'serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' --disable-socket --allow-hosts=127.0.0.1,localhost --cov=$(PACKAGE) --cov-report=term-missing || true
	pytest -m 'not serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --cov=$(PACKAGE) --cov-report=term-missing --cov-append

test-ci-fast:
	# Fast CI test suite: serial tests first (sequentially), then parallel execution for the rest
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Note: Coverage is excluded here for faster execution; full validation job includes unified coverage
	# Excludes slow tests even if marked critical_path (timeout/retry tests run in full suite)
	pytest tests/unit/ tests/integration/ tests/e2e/ -m 'serial and ((not integration and not e2e) or (integration and critical_path and not slow) or (e2e and critical_path and not slow))' --disable-socket --allow-hosts=127.0.0.1,localhost || true
	pytest tests/unit/ tests/integration/ tests/e2e/ -m 'not serial and ((not integration and not e2e) or (integration and critical_path and not slow) or (e2e and critical_path and not slow))' -n auto --disable-socket --allow-hosts=127.0.0.1,localhost

test-e2e:
	# E2E tests: serial tests first (sequentially), then parallel execution for the rest
	# Includes reruns for flaky tests (matches CI behavior)
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	@E2E_TEST_MODE=multi_episode pytest tests/e2e/ -m "e2e and serial" --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 || true
	@E2E_TEST_MODE=multi_episode pytest tests/e2e/ -m "e2e and not serial" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

test-e2e-sequential:
	# E2E tests: sequential execution (slower but clearer output, useful for debugging)
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	E2E_TEST_MODE=multi_episode pytest tests/e2e/ -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

test-e2e-fast:
	# Fast E2E tests: serial tests first (sequentially), then parallel execution for the rest
	# Critical path tests only (includes ML tests if models are cached)
	# Includes reruns for flaky tests (matches CI behavior)
	# Uses fast feed (1 episode) - set via E2E_TEST_MODE environment variable
	# Excludes slow tests even if marked critical_path (timeout/retry tests run in full suite)
	@E2E_TEST_MODE=fast pytest tests/e2e/ -m "e2e and critical_path and not slow and serial" --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 || true
	@E2E_TEST_MODE=fast pytest tests/e2e/ -m "e2e and critical_path and not slow and not serial" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

test-e2e-data-quality:
	# Data quality E2E tests: full pipeline validation with multiple episodes
	# Uses all original mock data (not fast fixtures)
	# Runs with 3-5 episodes per test to validate data quality and consistency
	# For nightly builds only - not part of regular CI/CD code quality checks
	@E2E_TEST_MODE=data_quality pytest tests/e2e/ -m "e2e and data_quality" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

test:
	# All tests: serial tests first (sequentially), then parallel execution for the rest
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	@E2E_TEST_MODE=multi_episode pytest tests/ -m serial --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost || true
	@E2E_TEST_MODE=multi_episode pytest tests/ -m "not serial" --cov=$(PACKAGE) --cov-report=term-missing --cov-append -n auto --disable-socket --allow-hosts=127.0.0.1,localhost

test-sequential:
	# All tests: sequential execution (slower but clearer output, useful for debugging)
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	E2E_TEST_MODE=multi_episode pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing

test-fast:
	# Fast tests: serial tests first (sequentially), then parallel execution for the rest
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Uses fast feed for E2E tests (1 episode) - set via E2E_TEST_MODE environment variable
	# Excludes slow tests even if marked critical_path (timeout/retry tests run in full suite)
	@E2E_TEST_MODE=fast pytest -m 'serial and ((not integration and not e2e) or (integration and critical_path and not slow) or (e2e and critical_path and not slow))' --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost || true
	@E2E_TEST_MODE=fast pytest -m 'not serial and ((not integration and not e2e) or (integration and critical_path and not slow) or (e2e and critical_path and not slow))' --cov=$(PACKAGE) --cov-report=term-missing --cov-append -n auto --disable-socket --allow-hosts=127.0.0.1,localhost

test-reruns:
	pytest --reruns 2 --reruns-delay 1 --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

coverage: test-unit

build:
	$(PYTHON) -m pip install --quiet build
	$(PYTHON) -m build
	@if [ -d dist ]; then mkdir -p .build && rm -rf .build/dist && mv dist .build/ && echo "Moved dist to .build/dist/"; fi

ci: format-check lint lint-markdown type security test docs build

ci-fast: format-check lint lint-markdown type security test-fast docs build

ci-sequential: format-check lint lint-markdown type security test-sequential docs build

ci-clean: clean format-check lint lint-markdown type security test docs build

docker-build:
	docker build -t podcast-scraper:test -f Dockerfile .

docker-build-fast:
	@echo "Building Docker image (fast mode - no model preloading, matches PR builds)..."
	@echo "This should complete in under 5 minutes..."
	@echo ""
	@time DOCKER_BUILDKIT=1 docker build \
		--build-arg WHISPER_PRELOAD_MODELS= \
		-t podcast-scraper:test-fast \
		-f Dockerfile .
	@echo ""
	@echo "✓ Fast build complete! Image tagged as: podcast-scraper:test-fast"

docker-build-full:
	@echo "Building Docker image (full mode - with model preloading, matches main builds)..."
	@echo "This will take longer due to Whisper model downloads..."
	@echo ""
	@time DOCKER_BUILDKIT=1 docker build \
		--build-arg WHISPER_PRELOAD_MODELS=base.en \
		-t podcast-scraper:test \
		-f Dockerfile .
	@echo ""
	@echo "✓ Full build complete! Image tagged as: podcast-scraper:test"

docker-test: docker-build
	@echo "Running Docker tests..."
	@echo "Test 1: --help command"
	@docker run --rm podcast-scraper:test --help > /dev/null
	@echo "Test 2: --version command"
	@docker run --rm podcast-scraper:test --version
	@echo "Test 3: No args (should error)"
	@docker run --rm podcast-scraper:test 2>&1 | grep -q "required" && echo "[OK] Error handling works"
	@echo "Test 4: Building with multiple Whisper models..."
	@docker build --quiet --build-arg WHISPER_PRELOAD_MODELS="tiny.en,base.en" -t podcast-scraper:multi-model -f Dockerfile . > /dev/null
	@docker run --rm podcast-scraper:multi-model --help > /dev/null
	@echo "[OK] All Docker tests passed"

docker-clean:
	docker rmi podcast-scraper:test podcast-scraper:test-fast podcast-scraper:multi-model 2>/dev/null || true

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
	# Coverage files: .coverage.* are created during parallel test execution (pytest -n auto)
	rm -f .coverage .coverage.*

clean-cache:
	@echo "Cleaning ML model caches..."
	@if [ -d "$$HOME/.cache/whisper" ]; then \
		echo "  Removing Whisper cache: $$HOME/.cache/whisper"; \
		echo "    (includes tiny.en model)"; \
		rm -rf "$$HOME/.cache/whisper"; \
	fi
	@if [ -d "$$HOME/.cache/spacy" ]; then \
		echo "  Removing spaCy cache: $$HOME/.cache/spacy"; \
		echo "    (Note: spaCy model is installed as dependency, but cache may exist)"; \
		rm -rf "$$HOME/.cache/spacy"; \
	fi
	@if [ -d "$$HOME/.local/share/spacy" ]; then \
		echo "  Removing spaCy user cache: $$HOME/.local/share/spacy"; \
		echo "    (Note: spaCy model is installed as dependency, but cache may exist)"; \
		rm -rf "$$HOME/.local/share/spacy"; \
	fi
	@if [ -d "$$HOME/.cache/huggingface" ]; then \
		echo "  Removing HuggingFace cache: $$HOME/.cache/huggingface"; \
		echo "    (includes all Transformers models: facebook/bart-base, facebook/bart-large-cnn, sshleifer/distilbart-cnn-12-6)"; \
		rm -rf "$$HOME/.cache/huggingface"; \
	fi
	@if [ -d "$$HOME/.cache/torch" ]; then \
		echo "  Removing PyTorch cache: $$HOME/.cache/torch"; \
		rm -rf "$$HOME/.cache/torch"; \
	fi
	@echo "Cache cleaning complete. Run 'make test-unit' to verify network isolation."

clean-all: clean clean-cache
	@echo "All cleaning complete."

preload-ml-models:
	@echo "Preloading ML models for local development..."
	@echo "This will download and cache models to avoid network calls during testing."
	@echo ""
	@echo "Preloading Whisper models..."
	@echo "  - tiny.en (English-only, smallest and fastest model for tests)..."
	@$(PYTHON) -c "import whisper; model = whisper.load_model('tiny.en'); print('  Verifying model loads...'); assert model is not None; print('  Verifying model structure...'); assert hasattr(model, 'dims') and model.dims is not None" || \
		(echo "ERROR: Failed to preload Whisper tiny.en. Install with: pip install openai-whisper" && exit 1)
	@echo "  ✓ Whisper tiny.en cached and verified"
	@echo ""
	@echo "Verifying spaCy model: en_core_web_sm..."
	@echo "  (Model is installed as a dependency, no download needed)"
	@$(PYTHON) -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('  Verifying model loads...'); assert nlp is not None; doc = nlp('Test text'); print('  Verifying model works...'); assert doc is not None and len(doc) > 0" || \
		(echo "ERROR: spaCy model not available. Install with: pip install -e .[ml]" && exit 1)
	@echo "✓ spaCy model verified (installed as dependency)"
	@echo ""
	@echo "Preloading Transformers models..."
	@echo "  - facebook/bart-base..."
	@$(PYTHON) -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; from pathlib import Path; import gc; print('  Downloading...'); tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base'); model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base'); print('  Verifying model loads...'); assert model is not None and tokenizer is not None; print('  Verifying tokenizer works...'); tokens = tokenizer.encode('Test text', return_tensors='pt'); assert tokens is not None; print('  Verifying model structure...'); assert hasattr(model, 'config') and model.config is not None; del model, tokenizer; gc.collect(); cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--facebook--bart-base'; snapshots = cache_path / 'snapshots'; assert snapshots.exists(), 'Model files not cached to disk'; model_files = []; [model_files.extend(list((snapshots / item.name).glob('*.safetensors')) + list((snapshots / item.name).glob('*.bin'))) for item in snapshots.iterdir() if (snapshots / item.name).is_dir()]; assert len(model_files) > 0, 'Model files not found in cache'; print('  ✓ Model files verified on disk')" || \
		(echo "ERROR: Failed to preload facebook/bart-base. Install with: pip install transformers torch" && exit 1)
	@echo "  ✓ facebook/bart-base cached and verified"
	@echo "  - facebook/bart-large-cnn..."
	@$(PYTHON) -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; from pathlib import Path; import gc; print('  Downloading...'); tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn'); model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn'); print('  Verifying model loads...'); assert model is not None and tokenizer is not None; print('  Verifying tokenizer works...'); tokens = tokenizer.encode('Test text', return_tensors='pt'); assert tokens is not None; print('  Verifying model structure...'); assert hasattr(model, 'config') and model.config is not None; del model, tokenizer; gc.collect(); cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--facebook--bart-large-cnn'; snapshots = cache_path / 'snapshots'; assert snapshots.exists(), 'Model files not cached to disk'; model_files = []; [model_files.extend(list((snapshots / item.name).glob('*.safetensors')) + list((snapshots / item.name).glob('*.bin'))) for item in snapshots.iterdir() if (snapshots / item.name).is_dir()]; assert len(model_files) > 0, 'Model files not found in cache'; print('  ✓ Model files verified on disk')" || \
		(echo "ERROR: Failed to preload facebook/bart-large-cnn. Install with: pip install transformers torch" && exit 1)
	@echo "  ✓ facebook/bart-large-cnn cached and verified"
	@echo "  - sshleifer/distilbart-cnn-12-6..."
	@$(PYTHON) -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; from pathlib import Path; import gc; print('  Downloading...'); tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6'); model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6'); print('  Verifying model loads...'); assert model is not None and tokenizer is not None; print('  Verifying tokenizer works...'); tokens = tokenizer.encode('Test text', return_tensors='pt'); assert tokens is not None; print('  Verifying model structure...'); assert hasattr(model, 'config') and model.config is not None; del model, tokenizer; gc.collect(); cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--sshleifer--distilbart-cnn-12-6'; snapshots = cache_path / 'snapshots'; assert snapshots.exists(), 'Model files not cached to disk'; model_files = []; [model_files.extend(list((snapshots / item.name).glob('*.safetensors')) + list((snapshots / item.name).glob('*.bin'))) for item in snapshots.iterdir() if (snapshots / item.name).is_dir()]; assert len(model_files) > 0, 'Model files not found in cache'; print('  ✓ Model files verified on disk')" || \
		(echo "ERROR: Failed to preload sshleifer/distilbart-cnn-12-6. Install with: pip install transformers torch" && exit 1)
	@echo "  ✓ sshleifer/distilbart-cnn-12-6 cached and verified"
	@echo "  - allenai/led-base-16384 (REDUCE model for summarization)..."
	@$(PYTHON) -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; from pathlib import Path; import gc; print('  Downloading...'); tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384'); model = AutoModelForSeq2SeqLM.from_pretrained('allenai/led-base-16384'); print('  Verifying model loads...'); assert model is not None and tokenizer is not None; print('  Verifying tokenizer works...'); tokens = tokenizer.encode('Test text', return_tensors='pt'); assert tokens is not None; print('  Verifying model structure...'); assert hasattr(model, 'config') and model.config is not None; del model, tokenizer; gc.collect(); cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--allenai--led-base-16384'; snapshots = cache_path / 'snapshots'; assert snapshots.exists(), 'Model files not cached to disk'; model_files = []; [model_files.extend(list((snapshots / item.name).glob('*.safetensors')) + list((snapshots / item.name).glob('*.bin'))) for item in snapshots.iterdir() if (snapshots / item.name).is_dir()]; assert len(model_files) > 0, 'Model files not found in cache'; print('  ✓ Model files verified on disk')" || \
		(echo "ERROR: Failed to preload allenai/led-base-16384. Install with: pip install transformers torch" && exit 1)
	@echo "  ✓ allenai/led-base-16384 cached and verified"
	@echo ""
	@echo "All models preloaded and verified successfully!"
	@echo "Models are cached in:"
	@echo "  - Whisper: ~/.cache/whisper/"
	@echo "  - spaCy: Installed as dependency (en_core_web_sm in pyproject.toml)"
	@echo "  - Transformers: ~/.cache/huggingface/hub/"

