# CodeQL Alert Dismissal Registry

## Purpose

This document is the single source of truth for **all** CodeQL alerts that have
been dismissed as false positives in this repository. It covers every alert
type we encounter, not just one query.

**Policy:** dismiss only after verifying that the code is genuinely safe, log
the dismissal here, and note which alert type it falls under.

---

## Alert types

### 1. `py/path-injection` -- Uncontrolled data used in path expression

**Why it fires:** CodeQL flags every `os.walk`, `os.path.isdir`,
`os.path.isfile`, and `open` call that receives a value derived from a FastAPI
query parameter -- even when the value has been sanitised via
`os.path.normpath` + `str.startswith` in a separate function.

**Why it is a false positive here:** CodeQL's taint-tracking state machine
requires the `normpath` + `startswith` guard to appear **inline in the same
function** as the filesystem call. Our architecture performs sanitisation in
shared helpers (`resolve_corpus_path_param`, `resolved_corpus_root_str`,
`safe_relpath_under_corpus_root`, `normpath_if_under_root`). CodeQL cannot
model cross-function sanitisation for this pattern, so every new file that
touches the filesystem with a request-derived path triggers the same false
positive.

**When alerts persist after ``safe_relpath_under_corpus_root``:** re-verify the
path string with ``normpath_if_under_root(path, root_s)`` immediately before
each ``open`` / ``os.path.isfile`` / ``FileResponse`` sink, or build the target
from ``safe_resolve_directory(corpus_root)`` plus **constant** path segments
with an inline ``os.path.normpath`` + ``str.startswith(safe_prefix)`` guard in
the same function. CodeQL does not always propagate sanitiser state out of
helpers.

**Inline pragma (same Type 1):** if sinks still alert after the above, add the
same ``# codeql[py/path-injection] -- …`` line used elsewhere under
``src/podcast_scraper/server/routes/`` (see ``corpus_binary.py``), documenting
the sanitizer chain. Prefer fixing taint flow first; use the pragma when CodeQL
cannot close the query.

**Code-side patterns added for viewer routes (same Type 1):** ``GET/PUT /feeds``
re-resolves the corpus root with ``safe_resolve_directory`` and checks
``feeds.spec`` with ``normpath_if_under_root`` before any filesystem access.
``GET …/jobs/…/log`` uses ``jobs_log_path.resolve_pipeline_job_log_path`` (same
function as ``isfile``): ``safe_resolve_directory``, then ``normpath_if_under_root``
on the ``safe_relpath_under_corpus_root`` output; ``routes/jobs`` maps
``JobLogPathError`` to ``HTTPException``.
``viewer_operator_extras_source`` (Docker mode) uses ``safe_fixed_file_under_root``
for ``viewer_operator.yaml`` before ``isfile``.
``publish_calendar_date_for_artifact_listing`` uses ``normpath_if_under_root``
before metadata ``isfile``. ``operator_config`` routes call
``_verified_operator_config_path`` so paths are either under the resolved corpus
root or exactly the server ``operator_config_fixed_path``. Generic helpers
``atomic_write_text`` and ``load_feeds_spec_file`` use pragmas documenting that
callers only pass corpus-anchored or packaged paths.

**CI unit tests (``.[dev]`` without ``.[server]``):** modules imported by unit tests
must not import FastAPI at import time. ``pipeline_jobs`` and ``operator_paths`` use
``typing.Any`` for the app handle; ``operator_config_security`` raises
``OperatorYamlUnsafeError`` (stdlib) and routes translate to ``HTTPException``.

**Sanitiser chain (reference):**

All user-supplied corpus paths flow through one of:

| Entry point | Module | Guards |
| --- | --- | --- |
| `resolve_corpus_path_param` | pathutil.py | `normpath` + `startswith(anchor)` -- raises `CorpusPathRequestError` on escape |
| `resolved_corpus_root_str` | pathutil.py | `normpath` + `startswith(anchor)` -- falls back to anchor on escape |
| `normpath_if_under_root` | path_validation.py | `normpath` + `startswith(root)` -- returns `None` on escape |
| `safe_relpath_under_corpus_root` | path_validation.py | `normpath` + `startswith` + no `..` -- returns `None` on escape |
| `safe_resolve_directory` | path_validation.py | ``realpath`` + rejects ``..`` -- use before joins from a ``Path`` root |

---

## Dismissed alerts

Alerts are dismissed via GitHub API as `false positive`.
Each row records the alert type (number from the list above), the alert
number, file, line, date, and a short comment.

<!-- Re-generate: gh api to fetch dismissed alerts, format as table. -->

| Type | Alert | File | Line | Dismissed | Comment |
| --- | --- | --- | --- | --- | --- |
| 1 | #44 | server/routes/artifacts.py | 106 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #45 | server/routes/artifacts.py | 112 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #50 | server/routes/artifacts.py | 37 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #51 | server/routes/artifacts.py | 67 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #52 | search/cli_handlers.py | 58 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #53 | search/cli_handlers.py | 59 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #54 | search/cli_handlers.py | 66 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #55 | search/cli_handlers.py | 68 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #56 | search/cli_handlers.py | 81 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #57 | search/cli_handlers.py | 82 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #58 | search/corpus_search.py | 120 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #59 | gi/explore.py | 68 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #60 | gi/explore.py | 72 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #61 | gi/explore.py | 73 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #62 | gi/explore.py | 74 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #63 | gi/explore.py | 130 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #64 | gi/explore.py | 148 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #65 | gi/explore.py | 151 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #66 | gi/explore.py | 162 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #67 | gi/explore.py | 163 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #68 | gi/explore.py | 304 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #69 | search/faiss_store.py | 164 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #70 | search/faiss_store.py | 166 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #71 | search/faiss_store.py | 169 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #72 | search/faiss_store.py | 188 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #73 | search/faiss_store.py | 188 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #74 | search/faiss_store.py | 190 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #75 | search/faiss_store.py | 192 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #76 | server/routes/index_stats.py | 64 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #77 | server/routes/index_stats.py | 49 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #78 | server/pathutil.py | 53 | 2026-04-06 | normpath+startswith via resolve_corpus_path_param |
| 1 | #129 | server/routes/corpus_binary.py | 60 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #130 | server/routes/corpus_binary.py | 66 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #131 | server/corpus_catalog.py | 33 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #132 | server/corpus_catalog.py | 160 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #133 | server/corpus_catalog.py | 293 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #134 | server/corpus_catalog.py | 294 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #136 | server/corpus_catalog.py | 331 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #139 | server/routes/corpus_metrics.py | 83 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #140 | server/routes/corpus_metrics.py | 160 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #141 | server/routes/corpus_metrics.py | 164 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #142 | server/routes/corpus_metrics.py | 191 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #143 | server/routes/corpus_metrics.py | 195 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #144 | server/routes/corpus_metrics.py | 222 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #146 | server/routes/corpus_metrics.py | 228 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #147 | server/routes/corpus_library.py | 220 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #149 | server/routes/corpus_library.py | 276 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #150 | server/routes/index_rebuild.py | 117 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #151 | search/index_source_mtime.py | 34 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #152 | search/index_source_mtime.py | 81 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #154 | server/index_staleness.py | 59 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #155 | utils/path_validation.py | 180 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #156 | utils/path_validation.py | 196 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #158 | server/pathutil.py | 74 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #159 | server/corpus_catalog.py | 328 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #160 | server/corpus_catalog.py | 358 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #161 | server/corpus_catalog.py | 359 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #162 | server/routes/corpus_library.py | 215 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #163 | server/routes/corpus_library.py | 271 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #164 | server/index_staleness.py | 55 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #165 | server/pathutil.py | 63 | 2026-04-10 | normpath+startswith via resolve_corpus_path_param |
| 1 | #206 | server/cil_queries.py | 59 | 2026-04-14 | anchor_s from server output_dir; root_path only in startswith filters |
| 1 | #207 | server/cil_queries.py | 63 | 2026-04-14 | anchor_s from server output_dir; root_path only in startswith filters |
| 1 | #166 | server/routes/index_stats.py | 89 | 2026-04-17 | normpath+startswith via resolve_corpus_path_param |
| 1 | #208 | server/corpus_catalog.py | 268 | 2026-04-17 | normpath+startswith via safe_resolve_directory |
| 1 | #209 | server/corpus_catalog.py | 345 | 2026-04-17 | normpath+startswith via safe_resolve_directory |
| 1 | #224 | server/routes/corpus_text_file.py | 68 | 2026-04-17 | normpath_if_under_root inline before isfile |
| 1 | #225 | server/routes/corpus_text_file.py | 82 | 2026-04-17 | normpath_if_under_root inline before isfile |
| 1 | #226 | server/routes/corpus_topic_clusters.py | 65 | 2026-04-17 | safe_resolve_directory + normpath+startswith inline |
| 1 | #227 | server/routes/corpus_topic_clusters.py | 76 | 2026-04-17 | safe_resolve_directory + normpath+startswith inline |
| 1 | #228 | server/routes/corpus_text_file.py | 144 | 2026-04-17 | normpath_if_under_root inline before FileResponse |
| 1 | #230 | search/topic_clusters.py | 108 | 2026-04-17 | safe_resolve_directory + normpath+startswith inline |
| 1 | #231 | search/topic_clusters.py | 104 | 2026-04-17 | safe_resolve_directory + normpath+startswith inline |
| 1 | #233 | server/cil_queries.py | 155 | 2026-04-18 | ``os.path.isdir(anchor_s)`` in ``iter_cil_bridge_bundles``; anchor/root normpath + prefix under server anchor (PR #588; same chain as #206/#207) |
| 1 | #234 | server/cil_queries.py | 159 | 2026-04-18 | ``os.walk(anchor_s)`` in ``iter_cil_bridge_bundles``; same guards as #233 |
| 1 | #235 | server/cil_queries.py | 187 | 2026-04-18 | ``_posix_relpath_under_corpus`` ``Path.resolve``; inputs from bridge paths already under ``root_prefix`` (node-episodes / bridge scan) |
| 1 | #236 | server/routes/corpus_library.py | 171 | 2026-04-18 | ``corpus_node_episodes`` ``root.resolve()`` after ``_resolve_corpus_root`` → ``resolve_corpus_path_param`` |
| 1 | #237 | server/cil_digest_topics.py | 82 | 2026-04-18 | ``_read_json_object`` ``open``; callers pass ``joined`` after normpath+startswith under corpus root or ``safe_relpath_under_corpus_root`` bridge path (PR #602) |
| 1 | #238 | server/cil_digest_topics.py | 102 | 2026-04-18 | ``corpus_root.resolve()`` + normpath join + ``startswith(safe_prefix)`` before cluster JSON access (PR #602) |
| 1 | #239 | server/cil_digest_topics.py | 108 | 2026-04-18 | ``os.path.isfile(joined)`` same chain as #238 (PR #602) |
| 1 | #240 | server/cil_digest_topics.py | 175 | 2026-04-18 | ``safe_relpath_under_corpus_root`` before ``isfile`` / ``_read_json_object`` on bridge (PR #602) |
| 1 | #241 | server/cil_digest_topics.py | 219 | 2026-04-18 | same as #240 in ``row_matches_library_topic_cluster_filter`` (PR #602) |
| 1 | #244–#297 | ``atomic_write.py``, ``feeds_spec.py``, ``corpus_catalog.py``, ``corpus_text_file.py``, ``routes/feeds.py``, ``routes/jobs.py``, ``routes/operator_config.py`` | various | 2026-04-21 | PR #649 ``py/path-injection`` batch on ``refs/pull/649/merge``; Type 1 false positives (resolve_corpus_path_param / ``normpath_if_under_root`` / ``safe_relpath_under_corpus_root`` / ``_verified_operator_config_path`` / trusted callers); dismissed via ``gh api …/code-scanning/alerts/{n}`` |

## Still open (not yet dismissed)

None.

---

## How to dismiss new alerts

### Step 1 -- Classify

Determine whether the alert matches a **known type** above.

- **Known type, same sanitiser chain:** proceed to dismiss (agent: no user
  approval needed for type 1 if the sanitiser chain is verified).
- **New type not listed here:** stop. Explain the taint flow to the user,
  propose a code fix if possible, and get explicit approval before dismissing.
  Then add the new type to the "Alert types" section above.

### Step 2 -- Dismiss via API

```bash
gh api repos/chipi/podcast_scraper/code-scanning/alerts/ALERT_NUMBER \
  -X PATCH \
  -f state=dismissed \
  -f dismissed_reason="false positive" \
  -f dismissed_comment="Type N: <short reason>"
```

### Step 3 -- Log

Add a row to the "Dismissed alerts" table and, if applicable, remove the
matching row from "Still open."
