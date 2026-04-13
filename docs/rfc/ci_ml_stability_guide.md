# 🧠 CI Instability & Process Leakage in Local ML Workflows (macOS)

## 1. Problem Summary

In a local development setup using agentic AI (e.g. Cursor) to run `make` scripts for CI-like workflows involving ML libraries (Whisper, Transformers, spaCy):

### Observed Issues

- Multiple Python/ML processes remain after runs
- Some processes cannot be killed (even with `kill -9`)
- Disk Utility reports:
  - `inode warnings`
  - `Resource Fork xattr missing`
- Occasional filesystem inconsistencies (repaired by First Aid)
- System instability:
  - Cannot log in after reboot
  - Requires long shutdown (overnight) to recover
- Repeated pattern across runs

---

## 2. Root Cause Hypothesis

This is **not caused directly by ML libraries**, but by:

### Primary Causes

- Poor process lifecycle management
- Orphaned subprocesses (child/grandchild processes)
- Unbounded disk I/O from ML workloads
- Shared cache mutation across runs
- Lack of cleanup after failures/timeouts

### Secondary Effects

- Processes stuck in **uninterruptible I/O wait**
- APFS inconsistencies under heavy disk load
- Snapshot / metadata inconsistencies
- System-level instability (login failures)

---

## 3. Key Insight

> Killing a single PID is insufficient.  
> Entire **process trees must be managed and terminated reliably**.

---

## 4. Recommended Improvements

### 4.1 Process Group Isolation (CRITICAL)

Ensure each CI run executes in its own process group/session.

#### Bash
```bash
setsid make test
```

#### Python
```python
import os, signal, subprocess, time

proc = subprocess.Popen(
    ["make", "test"],
    preexec_fn=os.setsid,
)

try:
    proc.wait(timeout=1800)
except subprocess.TimeoutExpired:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    time.sleep(5)
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
```

---

### 4.2 Cleanup Traps in All Scripts

Every script must guarantee cleanup on exit.

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
  sleep 2
  jobs -pr | xargs -r kill -9 2>/dev/null || true
  rm -rf "${TMPDIR:-/tmp}/ci-*"
}

trap cleanup EXIT INT TERM
```

---

### 4.3 Single Supervising Runner

Avoid chaining multiple `make` commands loosely.

Instead, use one entrypoint script that:
- Initializes environment
- Starts dependencies
- Runs tests
- Cleans up everything

---

### 4.4 Enforce Timeouts

Prevent hanging processes.

```bash
timeout 600 python bootstrap_models.py
timeout 900 pytest tests/
```

(macOS: use `gtimeout` or implement in Python)

---

### 4.5 Per-Run Isolation of Temp & Cache

Avoid shared state across runs.

```bash
RUN_ID="$(date +%s)-$$"
export RUN_ROOT="/tmp/ci-$RUN_ID"

export TMPDIR="$RUN_ROOT/tmp"
export HF_HOME="$RUN_ROOT/hf"
export TRANSFORMERS_CACHE="$RUN_ROOT/hf/transformers"
export XDG_CACHE_HOME="$RUN_ROOT/cache"

mkdir -p "$TMPDIR" "$HF_HOME" "$TRANSFORMERS_CACHE"
```

Cleanup:
```bash
rm -rf "$RUN_ROOT"
```

---

### 4.6 Separate Model Bootstrapping from Tests

Avoid repeated heavy I/O during tests.

Bad:
```bash
make integration
```

Better:
```bash
make prepare-model-cache
make integration
```

---

### 4.7 Avoid Background Daemons

Do NOT allow processes to detach or daemonize.

- No `&` without tracking
- No silent background services
- Prefer foreground execution

---

### 4.8 Kill Process Groups (Not Individual PIDs)

```bash
kill -- -$PGID
sleep 3
kill -9 -- -$PGID
```

---

### 4.9 Fix Python Multiprocessing Behavior (macOS)

Use `spawn` instead of `fork`:

```python
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
```

Always clean up pools:

```python
pool.close()
pool.join()
```

---

### 4.10 Post-Run Leak Detection

Detect leftover processes:

```bash
pgrep -fl "python|pytest|torch|whisper" || true
lsof +D "$RUN_ROOT" 2>/dev/null || true
```

Fail CI if leaks are detected.

---

### 4.11 Serialize Runs

Prevent overlapping executions:

```bash
exec 9>/tmp/ci.lock
flock -n 9 || exit 1
```

---

### 4.12 Reduce System Risk

Avoid running agent-driven CI directly on main system:

Options:
- Docker (preferred)
- VM
- Dedicated user account
- Strict temp/cache isolation

---

## 5. Recommended CI Runner Template

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="$(date +%s)-$$"
RUN_ROOT="/tmp/ci-$RUN_ID"

mkdir -p "$RUN_ROOT"/{tmp,cache,hf,logs}

export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export HF_HOME="$RUN_ROOT/hf"
export TRANSFORMERS_CACHE="$RUN_ROOT/hf/transformers"

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
  sleep 2
  jobs -pr | xargs -r kill -9 2>/dev/null || true
  rm -rf "$RUN_ROOT"
}

trap cleanup EXIT INT TERM

exec setsid bash -c '
  set -Eeuo pipefail
  make prepare-model-cache
  make integration-test
'
```

---

## 6. Makefile Best Practices

Avoid:
- Hidden background jobs (`&`)
- Silent failures (`|| true`)
- Complex logic inside recipes

Prefer:

```make
.PHONY: ci prepare-model-cache integration-test clean-runtime

ci: clean-runtime prepare-model-cache integration-test clean-runtime

prepare-model-cache:
	python scripts/prepare_model_cache.py

integration-test:
	python scripts/run_integration.py

clean-runtime:
	python scripts/cleanup_runtime.py
```

---

## 7. Operational Guidelines

- Reboot after any unkillable process incident
- Avoid running multiple heavy ML jobs concurrently
- Monitor disk I/O and cache growth
- Keep CI runs isolated and deterministic

---

## 8. Conclusion

The issue is **not ML libraries themselves**, but:

> ❗ Lack of process supervision + shared state + heavy I/O

Fixing:
- process group management
- cleanup guarantees
- isolation

will significantly improve system stability and prevent filesystem issues.
