# #1019 — Clean-separate dependency extras: scope

**Status:** scoping (no code changes yet)
**Date:** 2026-06-16
**Branch:** `chore/corpus-scrub-tooling` (the multi-issue working branch)

Grounded in three read-only audits (pyproject inventory, install-site sweep, GH-workflow
sweep) + a precise import-ownership check of every `[ml]∩[search]` dep.

---

## 1. The actual problem (narrowed by evidence)

The current design makes `[ml]` a **superset** of `[search]` — pyproject literally comments
that "[search] is a minimal subset of [ml] for cloud deployments." Both extras list the same
six heavy libs: `torch`, `transformers`, `sentencepiece`, `sentence-transformers`, `lancedb`,
`protobuf`.

Import-ownership check (`grep` over `src/podcast_scraper/`) shows the truth is **not** a clean
subset — most of those are genuinely needed by *both* capabilities, and exactly one is not:

| dep | used by `search/` code | used by ML-pipeline code (whisper / summarization / `gi/` grounding+NLI) | verdict |
|---|---|---|---|
| `lancedb` | yes (8 files: backends, pool, two_tier_indexer, topic_clusters, hybrid, stats, route, migration) | **no** | **search-only — wrongly in `[ml]`** |
| `sentence-transformers` | yes (cli_handlers, insight_clusters, protocol) | **yes** (`gi/grounding`, `gi/about_edges`, `providers/ml/embedding_loader`, `nli_loader`) | genuinely shared |
| `transformers` | yes | yes (summarization, NLI) | genuinely shared |
| `torch` | yes (via sentence-transformers) | yes (whisper, pyannote, torch directly) | genuinely shared |
| `sentencepiece` | yes (tokenizers) | yes | genuinely shared |
| `protobuf` | yes (transitive) | yes | genuinely shared |

**Conclusion:** the only dep that violates "each extra lists *its own* deps" is **`lancedb`**
— search-only but currently duplicated into `[ml]`. The other five are honest shared deps:
both `[ml]` and `[search]` list them because both capabilities' code imports them. That is
"listed by who needs them," not a superset hack — and it requires **no** inter-extra
reference (none exist today; confirmed 0).

This matches the operator's framing exactly: *"lancedb being in both [ml] and [search] is NOT
correct … ml is ml and search is search. cloud needs search and not ml."*

---

## 2. Core change (the #1019 commit)

1. **`pyproject.toml`**: remove `lancedb>=0.33.0,<1.0.0` from `[ml]` (line ~86). Keep it in
   `[search]` (line ~129). `[ml]` no longer carries a search-only lib.
2. **Rewrite the misleading comment** that calls `[search]` "a minimal subset of [ml]." New
   framing: each extra lists the libs its own code imports; the `torch`/`transformers`/
   `sentence-transformers`/`sentencepiece`/`protobuf` overlap is the shared model-runtime base
   that **both** transcription/GI **and** embedding/search genuinely import — not a subset.

No inter-extra includes. No recursive `podcast-scraper[search]`. `[ml]` and `[search]` are
each independently installable; the ML pipeline that *also* indexes composes `[ml,search]`.

---

## 3. Cascade effects — "consumers compose" (same commit or follow-up)

Removing `lancedb` from `[ml]` changes what three `[ml]`-without-`[search]` sites resolve.
None of these break search at runtime today, but they must be made explicit:

| site | today | change to | why |
|---|---|---|---|
| `docker/pipeline/Dockerfile` ml-mode | `pip install .[ml]` | `pip install .[ml,search]` | keep LanceDB available for corpus `upgrade`/migration m0002 inside the pipeline image; matches llm-mode which is already `.[llm,search]`. Net packages unchanged, now correctly attributed. |
| `.github/workflows/snyk.yml` (snyk-dependencies, snyk-monitor) | `.[dev,ml,llm]` | `.[dev,ml,llm,search]` | a security scanner audits the **installed** tree; without `[search]`, `lancedb` would silently drop out of CVE coverage once it leaves `[ml]`. |
| `Makefile` security-audit target (~line 415) | `.[ml]` | `.[ml,search]` | same — keep `lancedb` in the pip-audit environment. |

**No change needed** for the integration/e2e/acceptance jobs in `python-app.yml` / `nightly.yml`:
they already install `.[dev,ml,llm,search]` explicitly, so `lancedb` still arrives via `[search]`.

The `docker/api/Dockerfile` already installs `.[search]` (not `[ml]`) — unaffected, correct.

---

## 4. Separate real bug surfaced by the audit (own commit / issue)

**`prometheus-fastapi-instrumentator` major-version drift.** Production API image pins
`>=7.0.0,<8.0.0` (`docker/api/Dockerfile:51`) while pyproject `[dev]` — what tests run against —
pins `>=8.0.0,<9.0.0` (`pyproject.toml:149`). The served API runs a **different major version**
than CI tests. Minimum fix: bump the Dockerfile to `>=8.0.0,<9.0.0`. (Also minor: api uvicorn
`>=0.32` vs dev `>=0.48`; api apscheduler `>=3.10` vs dev `>=3.11.2` — low risk, align while here.)

---

## 5. [api] extract — considered and REJECTED

An earlier pass extracted a dedicated `[api]` server-runtime extra (single source of truth, api
image `.[api,search]`, every `.[dev…]` site → `.[dev,api]`). **The operator rejected it** as
unnecessary restructuring / churn that was never agreed: the server deps lived in `[dev]` and
worked; ripping them out across 23 install lines was not warranted to fix a 3-line pin drift.

**Decision: keep the server deps in `[dev]`.** The production api image still can't install full
`.[dev]` (test tooling in prod), so it continues to hand-pin the four server libs in
`docker/api/Dockerfile` — but those pins are now **bumped to match `[dev]`** (the actual fix for
§4). A pyproject comment + the Dockerfile comment both flag "keep these two in sync." Drift can
still recur in principle, but it's caught in review and costs nothing structurally.

---

## 6. Explicitly NOT changing (audit false-positives — documented so we don't "fix" them)

- **Snyk installing `[ml,llm]`** is *intentional*, not over-install: a dependency scanner must
  resolve the full production tree. (The §3 change only *adds* `[search]` for coverage.)
- **`[dev]` does not pull `[ml]`** transitively — confirmed. lint / unit / security-quality jobs
  correctly run on `.[dev]` only; leave them.
- **`pyannote.audio` + `torchaudio` in both `[ml]` and `[dev]`** — intentional CI/dev parity,
  documented in pyproject. Leave.
- **`rouge-score` (compare+dev), `httpx` (llm+dev)** — honest shared deps, version-consistent.
- **DGX microservice Dockerfiles** (whisper/pyannote/speaches) don't install the package at all —
  out of scope.

---

## 7. Delivery — ALL folded into #1019 (no new issues, no [api] extra)

Per operator: scoped into the **existing** #1019; no new tickets; `[api]` extract rejected (§5).

Delivered in one #1019 change:
- §2 — `lancedb` out of `[ml]`; comment corrected. It lives in `[search]`.
- §3 — compose `[ml,search]` at the three `[ml]`-only sites (pipeline ml-mode, snyk ×2, `make
  security-audit`) so lancedb stays present/CVE-scanned.
- §4 — `docker/api/Dockerfile` server pins bumped to match `[dev]` (prometheus `>=8`, uvicorn
  `>=0.48`, apscheduler `>=3.11.2`); server deps stay in `[dev]`; both sides comment "keep in sync."
- docs — `DEPENDENCIES_GUIDE.md` extras table corrected (adds `search`/`monitor` rows; fixes the
  `lancedb`/`[ml]` attribution). `[dev]` row keeps FastAPI.
