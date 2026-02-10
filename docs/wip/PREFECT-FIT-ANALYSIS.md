# Prefect Fit Analysis for podcast_scraper Pipeline

**Status:** Analysis  
**Date:** 2026-02-10  
**Context:** Evaluate [Prefect](https://docs.prefect.io/) as a next step to make the pipeline better, more robust, and easier to manage.

## Primary usage: CLI and service (unchanged)

The app will continue to be run **as CLI or service** (daemon). That is the main and supported way to run it. Prefect, if adopted, would be an **optional** third way to invoke the same pipeline (e.g. for a run UI or future scheduling). There are no plans for scheduling runs yet; scheduling is not required to use the app.

## Executive Summary

Prefect can add real value in **observability**, **scheduling/deployment**, and **resumability**, especially as you add GIL and multi-feed workflows. **Without scheduling plans**, the main extra benefit would be the run UI (if you run with a Prefect server); you can safely **skip Prefect for now** and revisit when you want scheduling or richer visibility. It is **not** a drop-in replacement: your current orchestration is a single Python pipeline with strong existing patterns (run manifests, retries, degradation). The best path is **incremental**: either wrap the existing pipeline as one Prefect flow for quick wins (UI, scheduling) or, if you invest in refactoring, break stages into Prefect tasks for full benefits (resume from failure, per-task retries, clearer DAG). Recommendation: **try a thin wrapper first** (one flow calling `run_pipeline`), then consider task-level adoption if you need per-stage resume, multi-feed orchestration, or event-driven triggers.

---

## Pros and cons (CLI/service primary, no scheduling yet)

Adopting Prefect in your situation — CLI and service stay the main way to run, no scheduling plans — looks like this:

### Pros of adding Prefect (even without scheduling)

- **Run UI:** One place to see run history, logs, and status (when using a Prefect server). Helps with debugging long runs.
- **Optional path:** You can add a thin wrapper and still run 100% via CLI/service; Prefect is an extra way to invoke the same pipeline.
- **Future-ready:** If you later add scheduling or multi-feed/GIL workflows, the wrapper is already there; you can grow into task-level Prefect then.
- **Same code:** The pipeline stays `run_pipeline(cfg)`; no need to refactor orchestration for a minimal wrapper.

### Cons of adding Prefect now

- **Extra dependency and concepts:** New runtime (flow/task, server or Cloud), another way to run the app, and docs to maintain.
- **Limited benefit without scheduling:** The main gain without scheduling is the UI; if you rarely need run history or a DAG view, the payoff is small.
- **Ops if you want the UI:** To get the run UI you run a Prefect server (self-hosted or Cloud); that’s another thing to install, configure, and keep updated.
- **Two ways to run:** Some users might trigger via Prefect, others via CLI/service; you have to document when to use which (or keep Prefect internal-only).

### Summary

| If you… | Then… |
| ------- | ----- |
| Want a run dashboard / history soon | Adding a thin Prefect wrapper can be worth it. |
| Are fine with logs + run manifests for now | Skipping Prefect is reasonable; revisit when you want scheduling or a UI. |

### If you decide not to run the Prefect server

Then you don’t get the run UI (no dashboard, no run history in the browser). Your **remaining pros** are:

- **Optional invocation:** You can still run the pipeline via Prefect (e.g. `prefect flow run …`) as a third way alongside CLI/service — but without the server you don’t see runs in a UI.
- **Future-ready:** If you later turn on the server or add scheduling, the wrapper is already there; you don’t have to refactor then.
- **Same code:** The pipeline stays `run_pipeline(cfg)`; no refactor for a minimal wrapper.

**Does it still make sense?** With **no server and no scheduling**, those pros are slim: you’d add a dependency and an extra way to invoke the app without getting visibility or scheduling. So **it usually doesn’t make sense** to add Prefect if you’ve decided not to run the server — stick with CLI and service, and revisit Prefect when you want the run UI or scheduling.

**Conclusion (no UI needed):** Prefect is **not** the right tool for making pipelines better when you don’t need a run UI. For robustness and maintainability without a UI, focus on what you already have (run manifests, retries, degradation, timeouts, CLI/service) and on incremental improvements (e.g. better logging, metrics, or refactors inside `orchestration.py`) rather than adding an orchestrator.

---

## What is a Prefect server, and how do I run it for the run UI?

The **Prefect server** is the backend that stores flow runs, logs, and state and serves the **Prefect UI** (run history, DAG view, logs). Without it, flows still run locally but you don’t get the dashboard.

**To get the run UI:**

1. **Install Prefect** (Python 3.10+):

   ```bash
   pip install -U prefect
   ```

2. **Start the server** (in a separate terminal):

   ```bash
   prefect server start
   ```

   By default it uses SQLite (`~/.prefect/prefect.db`) and no extra setup is needed.

3. **Open the UI** in a browser: [http://127.0.0.1:4200](http://127.0.0.1:4200)

4. **Point your app at the server** (in the terminal where you run your pipeline or Prefect flows):

   ```bash
   prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
   ```

5. **Run your flow** (e.g. your thin wrapper that calls `run_pipeline`). The run will show up in the UI.

**Optional:** Run the server in Docker instead of locally:

```bash
docker run -d -p 4200:4200 prefecthq/prefect:3-latest -- prefect server start --host 0.0.0.0
```

**Stop the server:** `Ctrl+C` in the terminal where `prefect server start` is running.

References: [Install Prefect](https://docs.prefect.io/v3/get-started/install), [Run a local Prefect server](https://docs.prefect.io/v3/how-to-guides/self-hosted/server-cli).

---

## 1. Current Pipeline at a Glance

- **Entry:** `workflow.orchestration.run_pipeline(cfg)` — single entry point; called from CLI and Service API (`service.run`, `service.run_from_config_file`).
- **Execution:** Pure Python; no workflow engine. Stages: setup → RSS fetch → episode materialization → concurrent transcript download (ThreadPoolExecutor) → sequential Whisper → metadata → summarization → cleanup.
- **Failure handling:** `DegradationPolicy` (continue vs fail-fast), `--fail-fast` / `--max-failures`, per-provider retries (`retry_with_metrics`), timeouts (transcription/summarization), run manifests, metrics, optional JSON logging.
- **Scheduling:** External only (cron, systemd timers, supervisor). No built-in scheduler.
- **Deployment:** CLI, service (daemon), Docker + supervisor; docs mention systemd/supervisor and “local execution on Mac laptops” (RFC-042).

So today you already have: deterministic layout, reproducibility (run manifests, seeds), retries at provider level, timeouts, degradation policy, and good testability. What you don’t have out of the box: a **run UI**, **first-class scheduling**, **resume from last successful step**, or **per-task caching**.

---

## 2. What Prefect Offers (Relevant to This Project)

From [Prefect docs](https://docs.prefect.io/v3/get-started):

| Feature | Description |
| ------- | ------------ |
| **Pythonic** | Flows and tasks as Python; no DSL. Fits your existing codebase. |
| **State & recovery** | Tracks success/failure/retry; resume from last successful point; cache expensive steps. |
| **Flexible execution** | Run locally, then same code in containers/K8s/cloud. |
| **Event-driven** | Schedules, webhooks, triggers; chain flows. |
| **Dynamic runtime** | Create tasks at runtime (e.g. one task per episode or per feed). |
| **Modern UI** | Flow run monitoring, logs, DAG view. |
| **CI/CD** | Test flows as normal Python. |

Prefect 3.x is open-source; you can self-host the server or use Prefect Cloud. Local runs work without a server (ephemeral API); the UI adds observability when a server is available.

---

## 3. Fit Analysis

### 3.1 Where Prefect Clearly Helps

1. **Observability**  
   Run manifests and logs are good; a Prefect UI gives run history, per-run logs, and a DAG of steps. Helpful for debugging long runs and for non-developers.

2. **Scheduling and deployment**  
   Prefect deployments + schedules (cron or interval) can replace cron/systemd for “run pipeline every night” with a single definition and visibility in the UI. Fits “run on schedule” use cases (e.g. nightly scrape).

3. **Resumability**  
   Today: fail-fast or continue with degradation; no “resume from episode N” or “resume from last successful stage.” Prefect’s state and checkpointing could support “resume from last successful task” if you expose stages as tasks.

4. **Future workflows (GIL, multi-feed)**  
   With GIL (RFC-049) and possibly multiple feeds or conditional branches, Prefect’s dynamic tasks and flow composition fit well (e.g. one flow per feed, or mapped tasks per episode).

5. **Per-task retries and caching**  
   You have provider-level retries; Prefect can add task-level retries and caching (e.g. “transcribe episode” cached by episode id), reducing rework on partial failures.

### 3.2 Overlap and Migration Cost

- **Overlap:** Run manifests (reproducibility), metrics, retries, timeouts, degradation. You’d keep run manifests for reproducibility; Prefect would add run-state and UI, not replace your manifest.
- **Migration options:**
  - **Thin wrapper (low effort):** One Prefect flow that calls `run_pipeline(cfg)`. You get scheduling + UI + one “block” per run. No resumability inside the pipeline; minimal code change.
  - **Task-level (high effort):** Refactor so each logical stage (setup, fetch_rss, process_episodes, transcribe_batch, metadata, summarize, cleanup) is a Prefect task. Enables resume, per-task retries, and caching. Requires splitting `orchestration.py` and passing state between tasks.

### 3.3 Tensions and Risks

1. **Execution model**  
   You use a single process with a thread pool for episode downloads and in-process queues for Whisper. Prefect’s parallelism is task-based (separate task runs). To map “one task per episode” you’d need to decide: one flow per run with many “process_episode” tasks (more Prefect-native) vs. one flow that calls your existing “process all episodes” (keeps current concurrency, less Prefect granularity).

2. **Dependency and ops**  
   Prefect adds a dependency and, for UI/scheduling, a server (self-hosted or Cloud). For “local Mac” and simple daemon use, running Prefect without a server is fine; you only get limited observability.

3. **Complexity**  
   Your pipeline is already robust. Prefect is most justified if you want: better visibility, first-class scheduling, or resume/cache behavior you don’t have today. If you only need “run on cron,” current cron + service API may be enough.

---

## 4. Recommendation

**CLI and service remain the primary interfaces.** Prefect is optional.

- **Short term (low risk):**  
  - Add an optional Prefect **wrapper**: a single flow that receives `Config` (or config path), calls `run_pipeline(cfg)`, and optionally writes run id to Prefect metadata.  
  - Use Prefect **deployments + schedule** for “nightly run” instead of (or in addition to) cron.  
  - Run with a Prefect server (or Cloud) when you want the UI; otherwise run the same flow with ephemeral API.  
  - This gives: scheduling as code, one place to see runs, and no refactor of existing orchestration.

- **Medium term (if you want more):**  
  - If you need “resume from last successful stage” or per-episode retries/caching, refactor stages into Prefect tasks and pass minimal state (e.g. output dir, episode list, job queue) between tasks.  
  - Consider this when you introduce GIL or multi-feed flows, so the new design is Prefect-native from the start.

- **What to keep regardless:**  
  - Run manifests and reproducibility (seeds, config hash, env).  
  - Degradation policy and `--fail-fast` / `--max-failures` (Prefect retries are complementary).  
  - Provider-level retries and timeouts; add Prefect task retries only where they add value (e.g. “fetch RSS” or “transcribe episode”).

---

## 5. Next Steps (If You Proceed)

1. **Spike (1–2 days):**  
   - Install Prefect in a dev environment.  
   - Implement one flow that loads config and calls `run_pipeline(cfg)`.  
   - Run it locally with ephemeral API; then with a local Prefect server and open the UI.  
   - Add a Prefect deployment with a simple schedule (e.g. daily); run it once.

2. **Document:**  
   - Add a short “Orchestrating with Prefect” section to the docs (optional path; CLI and service API remain the main interfaces).  
   - Note in ARCHITECTURE.md that Prefect is an optional orchestration layer on top of `run_pipeline`.

3. **Decide on server:**  
   - Self-hosted Prefect server vs. Prefect Cloud (auth, backups, upgrades). For a single team/solo, Cloud can reduce ops.

4. **Revisit task-level adoption** when:  
   - You implement GIL or multi-feed flows, or  
   - You need resume-from-failure or per-task caching and are willing to refactor orchestration.

---

## 6. References

- [Prefect Introduction](https://docs.prefect.io/v3/get-started)  
- Project: `docs/ARCHITECTURE.md`, `src/podcast_scraper/workflow/orchestration.py`, `src/podcast_scraper/service.py`, `src/podcast_scraper/workflow/degradation.py`  
- RFCs: RFC-001 (workflow orchestration), RFC-049 (GIL), RFC-042 (hybrid summarization)
