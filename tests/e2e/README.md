# End-to-end tests — expected duration and the "is it hung?" question

These run the real pipeline against a mock feed server (no cloud, no network — `--disable-socket`).
They are **slow by design**, and the slowness is easy to mistake for a hang. This note sets
expectations so the next person (or agent) doesn't kill a healthy run.

## Expected duration

| Suite | Command | Tests | Wall time (measured) |
| --- | --- | --- | --- |
| **e2e** | `make test-e2e` | ~350 | **~12 min** local (`-n 2`); **~25 min** on GitHub CI |
| integration | `make test-integration` | ~2950 | **~8 min** local |

Timings measured on 2026-07-30 (14-core laptop). CI runners are slower, so budget ~25 min for e2e
there. Do not treat a run under these budgets as stuck.

## Why e2e looks like it stalls around the halfway mark

The mock feed server **simulates network reality**: injected delays, retry/backoff, and error
status codes (429/5xx). The tests that exercise the retry / backoff / failover paths **sleep
through those simulated delays**. While a worker sleeps:

- its CPU is **0%** and its process state is **`SN` (interruptible sleep)**,
- the pytest progress dots **freeze** (no test completes during the sleep),
- there is **no** `podcast_scraper.cli` subprocess running under that worker.

That is a worker legitimately waiting out a simulated backoff — **not** a deadlock. The band around
~50% happens to concentrate these tests, so the whole suite appears to stop there for a minute or
more at a time, then resumes.

## Slow vs. actually hung — how to tell

Sample the workers over ~60s:

```bash
ps -eo pid,%cpu,stat,command | grep '[p]ytest'
```

- **Healthy (slow):** workers cycle between `R`/`RN` (running a pipeline, non-zero CPU) and `SN`
  (sleeping through a simulated delay). The `.`/`s` marker count keeps climbing over minutes, even
  if it pauses for a stretch.
- **Actually stuck:** the marker count does not move over several minutes **and** no worker ever
  leaves `SN` **and** no `podcast_scraper.cli` subprocess ever appears. Only then is it worth
  investigating (grab a `py-spy dump` / faulthandler stack of the worker PID).

## Reference

The ~50% slow band and whether the simulated delays can be shortened under test (without losing
coverage of the backoff logic) is tracked in **#1354**.

## What this suite runs against

No cloud, no network, no real podcast audio. The pipeline fetches feeds, audio and transcripts from
a **local mock host** that simulates a podcast server — `make serve-e2e-mock` (port `18765`), backed
by [`fixtures/e2e_http_server.py`](fixtures/e2e_http_server.py), which also stubs the LLM provider
APIs. `--disable-socket` enforces that nothing escapes to the internet.

The fixtures it serves are committed and **versioned** — a fixture path without a version is wrong.
See [`../fixtures/README.md`](../fixtures/README.md), and
[`../fixtures/audio/README.md`](../fixtures/audio/README.md) for the audio specifically.

## Related

- [`../README.md`](../README.md) — the whole test tree: tiers, commands, where data lives
- [`../../docs/guides/E2E_TESTING_GUIDE.md`](../../docs/guides/E2E_TESTING_GUIDE.md) — how the tiers fit together, and the browser suites
- [`../stack-test/README.md`](../stack-test/README.md) — the containers-together tier above this one
- [`../../docker/mock-feeds/README.md`](../../docker/mock-feeds/README.md) — the same fixtures over nginx, for stack-test
