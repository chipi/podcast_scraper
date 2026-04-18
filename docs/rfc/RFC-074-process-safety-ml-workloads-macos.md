# RFC-074: Process Safety for ML Workloads on macOS

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: All contributors running CI locally on macOS
- **Related Documents**:
  - `docs/architecture/TESTING_STRATEGY.md`
  - `docs/rfc/ci_ml_stability_guide.md` (predecessor draft, superseded by this RFC)

## Abstract

This RFC addresses recurring system-level instability caused by Python/ML
processes entering macOS kernel uninterruptible wait state during local CI
runs. The problem has caused 3 complete system crashes in 4 months, including
inability to log in after reboot. The RFC proposes concrete changes to the
Makefile, pre-commit hook, agent rules, and operational setup to prevent
process pileup, bound blast radius, and eliminate the conditions that lead
to APFS filesystem corruption.

## Problem Statement

### Incident pattern

Three times in four months, the development machine became completely
unusable during local CI runs involving ML model loading (spaCy,
Transformers, Whisper). The pattern each time:

1. Multiple Python processes loading ML models get stuck
2. Processes enter `UE` (uninterruptible wait) state in the macOS kernel
3. `kill -9` has no effect -- signals are not delivered to processes in
   uninterruptible wait (this is by POSIX design, not a bug)
4. More `make` invocations spawn more Python processes that hit the same
   kernel lock, creating a pileup
5. Eventually even `python3 -c "print('hello')"` hangs -- Python itself
   is blocked system-wide
6. Only resolution: reboot
7. After reboot: Disk Utility reports inode warnings, `Resource Fork xattr
   missing for compressed file`, and in the worst case the user cannot log in
   (requires overnight shutdown or Recovery Mode repair)

### Root cause chain

The failure is not caused by ML libraries themselves. It is caused by
the interaction of four factors:

**Factor 1: APFS global kernel lock.** APFS acquires a global kernel mutex
(`lck_mtx_lock`) during filesystem operations including `readdir()`. When
multiple Python processes perform parallel filesystem I/O -- as happens
during ML model loading -- they contend for this lock. Processes waiting
for the lock enter uninterruptible wait. This is a well-documented APFS
behavior (Gregory Szorc, 2018; Apple Developer Forums thread 800906;
rdar://45648013). It has been partially improved since macOS 10.14 but
remains present.

**Factor 2: ML model loading does heavy filesystem I/O.** Loading an ML
model is not like reading a source file. It involves:

- `spacy.load()`: reads model package directory, loads weights into memory
- `AutoTokenizer.from_pretrained()`: scans HuggingFace cache directory
  structure, reads tokenizer files
- Whisper `load_model()`: reads multi-hundred-MB `.pt` files
- All of these trigger `readdir()`, `lstat()`, `open()`, `mmap()` on
  directories containing thousands of files

**Factor 3: Parse-time Makefile probes.** The current Makefile has a
`$(shell ...)` assignment at parse time (line 1195) that imports
`podcast_scraper.config`, `tests.integration.ml_model_cache_helpers`,
and calls `spacy.load()` -- every time any `make` target is invoked,
including `make help`. This spawns a Python process that does heavy
ML I/O before any recipe runs.

**Factor 4: Agentic tooling multiplies process spawn rate.** A human
developer runs `make` perhaps 5 times per hour. An AI agent in Cursor
runs `make` commands 20-50 times per hour, with rapid retries when
something appears stuck. Each invocation spawns a new ML-loading Python
process. The pileup that might take hours manually happens in minutes
with an agent.

### Cascade to filesystem corruption

When the system is forcibly rebooted (the only option once processes are
in `UE` state), APFS may have incomplete metadata operations in flight.
This produces inode warnings, resource fork inconsistencies, and in the
worst case, corruption that prevents login. The login process itself
requires filesystem operations (reading user preferences, keychain,
launch agents) and can enter uninterruptible wait on corrupted metadata.

### Evidence

- Incident timeline from Cursor session (agent transcript
  `1bbd5b8a-6b1f-4bc7-a365-9dea5c38e49e`): ~12 zombie Python processes
  in `UE` state, all from the `ML_MODELS_CACHED` Makefile probe
- Gregory Szorc's analysis of APFS global kernel locks (2018):
  `readdir()` contention under parallel Python processes
- mlx-lm issue #883: ML workload on macOS causing kernel panic via
  unbounded memory/I/O (same class of problem, different trigger)
- Apple Developer Forums thread 800906: ongoing APFS lock contention
  reports through 2025

## Goals

1. **Eliminate parse-time ML process spawning** so routine `make` targets
   never trigger model loading
2. **Prevent process pileup** by detecting existing processes before
   spawning new ones
3. **Bound blast radius** with timeouts, process groups, and cleanup traps
4. **Reduce APFS lock contention** by minimizing parallel filesystem I/O
   on model cache directories
5. **Codify agent behavior** to prevent the rapid-retry pattern that
   amplifies the problem

## Constraints and Assumptions

**Constraints:**

- All changes must work on macOS (currently 15.7.3 Sequoia)
- Must not break GitHub Actions CI (Linux)
- Must not require Docker or VM setup
- Must not slow down the normal development loop
- `setsid` is not available on macOS without Homebrew; use Python
  `os.setpgrp` or `os.setsid` where needed

**Assumptions:**

- Spotlight is already disabled (`mdutil -s /` confirms this)
- ML model caches are in `~/.cache/huggingface`, `~/.cache/whisper`,
  and `.venv/lib/*/site-packages/en_core_web_*`
- The pre-commit hook is installed via `make install-hooks`

## Design and Implementation

### Group 1: Makefile -- stop spawning unnecessary ML processes

**Change 1.1: Move `ML_MODELS_CACHED` off parse-time.**

Replace the `:=` assignment (line 1195) with a shell block inside the
`ci:` recipe. Only `make ci` and `make ci-nightly` need this check.

```make
ci:
	@cached=$$($(PYTHON) -c "import sys; sys.path.insert(0, 'src'); \
	from tests.integration.ml_model_cache_helpers import _is_whisper_model_cached, _is_transformers_model_cached; \
	from podcast_scraper import config; \
	whisper_ok = _is_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL); \
	transformers_ok = _is_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None); \
	spacy_ok = False; \
	try: \
		import spacy; nlp = spacy.util.get_installed_models(); \
		spacy_ok = config.DEFAULT_NER_MODEL.replace('-', '_') in [m.replace('-', '_') for m in nlp]; \
	except Exception: \
		pass; \
	all_cached = whisper_ok and transformers_ok and spacy_ok; \
	print('1' if not all_cached else '0', end='')" 2>/dev/null || printf '1'); \
	$(MAKE) _ci_impl ML_MODELS_CACHED="$$cached"
```

Key changes from current code:

- No `:=` -- runs only when `make ci` is invoked
- `spacy.load()` replaced with `spacy.util.get_installed_models()` (no
  model loading, just checks installed package list)
- `except:` changed to `except Exception:` (KeyboardInterrupt works)
- `|| echo "1"` changed to `|| printf '1'` (no stray newline)

**Change 1.2: Move `PYTEST_WORKERS` off parse-time.**

Replace the `?=` with `$(shell ...)` (line 40) with a simple default.
Each recipe that needs a worker count already has its own inline
`$(shell ...)` call at recipe time.

```make
PYTEST_WORKERS ?= 2
```

The parse-time `calculate_test_workers.py` call is redundant because
every test recipe already calls it inline with the correct `--test-type`.

**Change 1.3: Fix `_is_transformers_model_cached` to avoid loading
tokenizer.** (Implemented)

In `tests/integration/ml_model_cache_helpers.py`, replace the
`AutoTokenizer.from_pretrained(..., local_files_only=True)` call with
a filesystem check: verify the snapshot directory contains `config.json`
plus at least one weight file (`.safetensors` or `.bin`). This matches
the pure-filesystem pattern used by `_is_whisper_model_cached` and
eliminates the heavy disk I/O that contributed to APFS lock contention.

### Group 2: Pre-commit hook -- timeout and process group cleanup

**Change 2.1: Add master timeout.**

Wrap the hook body in a bash timeout using `SECONDS` (portable, no
`gtimeout` dependency):

```bash
MAX_HOOK_SECONDS=120
(
  # ... existing hook body ...
) &
HOOK_CHILD=$!
while kill -0 $HOOK_CHILD 2>/dev/null; do
  if [ $SECONDS -ge $MAX_HOOK_SECONDS ]; then
    kill -- -$HOOK_CHILD 2>/dev/null || true
    sleep 2
    kill -9 -- -$HOOK_CHILD 2>/dev/null || true
    echo "Pre-commit hook timed out after ${MAX_HOOK_SECONDS}s"
    exit 1
  fi
  sleep 1
done
wait $HOOK_CHILD
exit $?
```

**Change 2.2: Extend cleanup trap to kill child processes.**

```bash
cleanup() {
    kill -- -$$ 2>/dev/null || true
    rm -rf "${LOG_DIR}/precommit_${HOOK_PID}"* 2>/dev/null || true
}
```

### Group 3: Agent rules -- prevent rapid-retry pileup

**Change 3.1: Add to `.cursorrules`:**

```text
### Process Safety (ML workloads)

10. **NEVER retry a `make` command that produced no output**
    - If `make` appears stuck (no output for 60s), do NOT spawn another
      `make`. Report to user: "make appears stuck, check for zombie
      processes with `make check-zombie`"
    - NEVER run multiple `make ci` / `make ci-fast` / `make test`
      concurrently

10a. **Before running `make ci` or `make ci-fast`, check for existing
     runs**
    - Run `pgrep -f "make.*(ci|test)" || true` first
    - If processes found, ask user before proceeding

10b. **After any `make` command that hangs or is killed, run
     `make cleanup-processes`**
    - This prevents accumulation of orphaned Python processes
```

### Group 4: Makefile cleanup infrastructure

**Change 4.1: Extend `cleanup-processes` patterns.**

Add patterns that match the ML probe and model-loading processes:

```make
cleanup-processes:
	@echo "Cleaning up leftover test processes..."
	@pkill -f "pytest" 2>/dev/null || true
	@pkill -f "gw[0-9]" 2>/dev/null || true
	@pkill -f "python.*ml_model_cache_helpers" 2>/dev/null || true
	@pkill -f "python.*spacy.*load" 2>/dev/null || true
	@pkill -f "python.*calculate_test_workers" 2>/dev/null || true
	@echo "Process cleanup complete"
```

**Note:** A historical pattern `python.*podcast_scraper.*test` was removed
from the live `Makefile` because it false-positived on
`python -m podcast_scraper.cli serve --output-dir …/.test_outputs` (the
`.*` could bridge to `test` inside `test_outputs`), killing dev API
servers whenever `cleanup-processes` ran (for example before `ci-fast`).

**Change 4.2: Add `check-zombie` target.**

```make
check-zombie:
	@echo "Checking for unkillable (UE state) Python processes..."
	@zombie_count=$$(ps aux | grep -E '[Pp]ython|[Pp]ytest' | \
		awk '$$8 ~ /U/' | grep -v grep | wc -l | tr -d ' '); \
	if [ "$$zombie_count" -gt 0 ]; then \
		echo "WARNING: $$zombie_count unkillable Python process(es) found:"; \
		ps aux | grep -E '[Pp]ython|[Pp]ytest' | awk '$$8 ~ /U/' | grep -v grep; \
		echo ""; \
		echo "These processes cannot be killed. Reboot required."; \
		echo "After reboot, run Disk Utility First Aid."; \
		exit 1; \
	else \
		echo "No zombie processes found."; \
	fi
```

**Change 4.3: Add `cleanup-processes` as prerequisite to `ci` and
`ci-fast`.**

Currently only `test-unit`, `test-unit-no-ml`, `test-unit-dev-venv`,
`test-integration`, and `test-e2e` depend on it. Add it to `ci` and
`ci-fast` as well.

### Group 5: Offline mode for tests (not global Makefile export)

**Change 5.1: Do not export HF offline for every recipe.**

`tests/conftest.py` sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` for pytest. A **global** Makefile `export` of
those variables broke `make preload-ml-models` / `make
preload-ml-models-production` and `make hf-hub-smoke-test`, because
those targets must reach Hugging Face.

Current approach:

- **`ci:` cache probe:** prefix the probe `$(PYTHON) -c ...` with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` so the probe does not hit
  the Hub accidentally.
- **Preload / HF smoke:** run with `env -u HF_HUB_OFFLINE -u
  TRANSFORMERS_OFFLINE` so downloads work even if the developer
  exported offline mode in their shell.

### Group 6: System setup documentation and verification

**Change 6.1: Add `check-spotlight` target.**

```make
check-spotlight:
	@echo "Checking Spotlight indexing status..."
	@if mdutil -s / 2>/dev/null | grep -q "Indexing enabled"; then \
		echo "WARNING: Spotlight indexing is enabled."; \
		echo "Heavy ML I/O + Spotlight = APFS lock contention risk."; \
		echo "Disable with: sudo mdutil -a -i off"; \
		echo "Or exclude cache dirs in System Settings > Spotlight > Privacy"; \
	else \
		echo "Spotlight indexing is disabled. Good."; \
	fi
```

**Change 6.2: Document system prerequisites in RFC.**

The following macOS settings are prerequisites for safe ML development:

- Spotlight indexing disabled (`sudo mdutil -a -i off`) or cache
  directories excluded (`~/.cache/huggingface`, `~/.cache/whisper`,
  project `.venv`)
- If Spotlight is ever re-enabled, exclude these paths in
  System Settings > Siri and Spotlight > Spotlight Privacy

### Group 7: Timeout for `preload_ml_models.py`

**Change 7.1: Add wall-clock timeout.**

The project already has `src/podcast_scraper/utils/timeout.py` using
`threading.Timer`. `preload_ml_models.py` uses `SIGALRM` (Unix) instead.

Defaults (after parsing CLI args): non-production preloads use **600s**;
``--production`` uses **7200s** because cold Whisper + Hugging Face + GIL
evidence downloads exceed 10 minutes on CI and locally. Environment variable
``PRELOAD_TIMEOUT`` overrides both; ``0`` disables the alarm.

Arm the alarm after ``argparse`` (so ``--production`` is known), before
any model loading. Reset with ``signal.alarm(0)`` after all loading completes.

Note: `signal.alarm` is Unix-only, which is fine -- this script only
runs locally on macOS or in Linux CI.

## Key Decisions

1. **Filesystem checks instead of model loading for cache probes**
   - **Decision**: Replace `spacy.load()` and
     `AutoTokenizer.from_pretrained()` in cache probes with file/directory
     existence checks
   - **Rationale**: The probe only needs to know if models are cached,
     not load them. Loading triggers the exact heavy I/O that causes
     the problem.

2. **Agent rules over technical enforcement**
   - **Decision**: Add `.cursorrules` rules to prevent rapid-retry
     pileup rather than building a process supervisor
   - **Rationale**: The agent is the primary driver of rapid `make`
     invocations. Changing its behavior is simpler and more effective
     than building infrastructure to contain it.

3. **No Docker requirement**
   - **Decision**: All mitigations work on bare macOS
   - **Rationale**: Docker adds friction to the development loop.
     The mitigations here are sufficient to prevent the problem.
     Docker can be a future enhancement.

## Alternatives Considered

1. **Docker-based local CI**
   - **Description**: Run all CI inside a Docker container
   - **Pros**: Complete isolation from host OS
   - **Cons**: Significant workflow change, slower iteration, GPU/MPS
     not available in Docker on macOS
   - **Why Rejected**: Too much friction for daily development. Can be
     added later as an optional path.

2. **Process supervisor daemon**
   - **Description**: A background daemon that monitors and kills
     runaway processes
   - **Pros**: Automatic, no agent cooperation needed
   - **Cons**: Complex, another process to manage, could itself get
     stuck
   - **Why Rejected**: Over-engineered for the problem. Preventing
     the pileup is better than cleaning it up.

## Testing Strategy

**Verification:**

- `make help` completes in under 2 seconds (no ML imports)
- `make format-check` completes without spawning ML-loading Python
- `make ci` still conditionally runs `preload-ml-models` when models
  are not cached
- Pre-commit hook times out after 120s instead of hanging indefinitely
- `make check-zombie` correctly reports UE-state processes (manual test
  after reproducing the condition, or verified by reading `ps` output)
- `make check-spotlight` reports correct Spotlight status

**No new test files needed.** These are build tooling and operational
changes, not application code.

## Rollout Plan

All changes ship in a single branch. Order of implementation:

1. Makefile changes (Groups 1, 4, 5, 6) -- highest impact, lowest risk
2. Pre-commit hook changes (Group 2) -- requires `make install-hooks`
3. `.cursorrules` changes (Group 3) -- immediate effect on agent behavior
4. `preload_ml_models.py` timeout (Group 7)
5. `ml_model_cache_helpers.py` fix (Change 1.3)

## Success Criteria

1. Zero system crashes from ML process pileup over the next 3 months
2. `make help` returns in under 2 seconds
3. No Python process spawned by Makefile parse on any target except `ci`
4. Agent never spawns overlapping `make ci` runs

## Recovery Runbook

If the problem occurs again despite these mitigations:

1. **Check for UE processes**: `ps aux | grep python | awk '$8 ~ /U/'`
2. **If UE processes exist**: reboot is the only option
3. **After reboot**: run Disk Utility First Aid on the boot volume
4. **If login fails**: hold power button for 10s to force shutdown,
   wait 30s, boot to Recovery Mode (hold Command+R), run First Aid
5. **After recovery**: run `make check-zombie` to verify clean state
6. **Report**: note what `make` target was running, how many processes
   were stuck, and whether Disk Utility found errors

## References

- Gregory Szorc, "Global Kernel Locks in APFS" (2018):
  `https://gregoryszorc.com/blog/2018/10/29/global-kernel-locks-in-apfs`
- Apple Developer Forums, "Lock Contention in APFS/Kernel?" (2025):
  `https://developer.apple.com/forums/thread/800906`
- rdar://45648013: APFS readdir() global lock
- mlx-lm issue #883: macOS kernel panic from ML workload (2026):
  `https://github.com/ml-explore/mlx-lm/issues/883`
- Eclectic Light Company, "What to do with APFS warnings and errors"
  (2026): `https://eclecticlight.co/2026/03/20/what-to-do-with-apfs-warnings-and-errors/`
- Incident session transcript: `1bbd5b8a-6b1f-4bc7-a365-9dea5c38e49e`
