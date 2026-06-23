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
``JobLogPathError`` to ``HTTPException``. If CodeQL still flags ``isfile`` on the
resolved path, ``jobs_log_path`` carries ``# codeql[py/path-injection]`` on the line
immediately above that sink (single-line comment; Type 1).
``viewer_operator_extras_source`` (Docker mode) uses ``safe_fixed_file_under_root``
for ``viewer_operator.yaml`` before ``isfile``. If CodeQL still flags ``isfile``,
use a **single-line** ``# codeql[py/path-injection] -- …`` immediately above the
sink (splitting the pragma across two ``#`` lines can fail to suppress; Type 1). ``routes/jobs`` uses the same pragma for ``FileResponse`` /
``to_thread(assert_operator_pipeline_extras, …)`` / log-tail ``verified_under`` where
taint does not cross from ``jobs_log_path`` / ``operator_paths`` helpers.
``publish_calendar_date_for_artifact_listing`` uses ``normpath_if_under_root``
before metadata ``isfile``. ``operator_config`` routes call
``_verified_operator_config_path`` so paths are either under the resolved corpus
root or exactly the server ``operator_config_fixed_path``. Generic helpers
``atomic_write_text`` and ``load_feeds_spec_file`` use pragmas documenting that
callers only pass corpus-anchored or packaged paths.

**CI unit tests:** ``check_test_policy`` keeps FastAPI out of ``tests/unit/`` even though ``.[dev]`` includes it. Modules imported by unit tests
must not import FastAPI at import time. ``pipeline_jobs`` and ``operator_paths`` use
``typing.Any`` for the app handle; ``operator_config_security`` raises
``OperatorYamlUnsafeError`` (stdlib) and routes translate to ``HTTPException``.

### 2. `actions/artifact-poisoning/critical` -- artifact download in same workflow_call chain

**Why it fires:** CodeQL flags `actions/download-artifact` steps in
workflows triggered by `workflow_dispatch` because, in theory, an
attacker could replace the artifact between upload and download if
the upload happened in a less-privileged workflow.

**Why it is a false positive here:** the artifact is uploaded and
downloaded within the **same** `workflow_call` chain (`github.run_id`
is identical). The triggering workflow (`drill-exercise` /
`drill-infra-destroy`) requires `workflow_dispatch` with a confirmation
input and is restricted to repo admins. No external user can inject
content into the artifact between upload (infra-apply/plan) and
download (infra-destroy).

### 3. Snyk Container -- base-image transitive CVE not reachable (formerly type 2)

**Why it fires:** the Snyk container scan inspects the final
``podcast-scraper:snyk-scan`` image's Debian package list and uploads each
``high``/``critical`` package CVE as a Code Scanning alert. These are
**not** vulnerabilities in our Python code or pip dependencies -- they are
in OS packages from the ``python:3.12-slim`` (Debian 13) /
``python:3.12-slim-bookworm`` (Debian 12) base images.

**Why it is a false positive here:** apply both gates before dismissing:

1. **Upstream Debian fix status.** Either Debian explicitly has no fix
   (``apt-get upgrade`` cannot help) or the version flagged is already
   the latest in Debian's apt index. The Dockerfiles run
   ``apt-get update && apt-get upgrade -y`` early so any backported fix
   lands automatically on the next rebuild.
2. **Reachability from our deployment topology.** The vulnerable code
   path must not be invoked by anything the pipeline runs: Whisper,
   spaCy, transformers, sentence-transformers, FAISS, ffmpeg, supervisor,
   FastAPI / uvicorn, httpx / requests, etc. We don't expose DTLS
   handshake parsing to untrusted peers; we don't process ICC color
   profiles or other image-rendering flows; and so on.

When **both** gates pass, dismiss as ``won't fix`` with a comment that
names (a) the Debian fix status and (b) the reachability reasoning.
Re-evaluate when Snyk re-scans on each PR -- if the same CVE re-surfaces
after a Debian backport ships, ``apt-get upgrade`` should pick it up
automatically; otherwise re-dismiss and append a new row below.

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
| 1 | #304 | server/operator_paths.py | 50 | 2026-04-24 | Type 1: ``candidate_s`` from ``safe_fixed_file_under_root`` before ``isfile``; CodeQL cross-function taint gap; dismissed ``gh api`` (PR #666) |
| 1 | #305 | server/jobs_log_path.py | 74 | 2026-04-24 | Type 1: ``log_path`` from ``normpath_if_under_root`` after ``safe_relpath_under_corpus_root`` before ``isfile``; dismissed ``gh api`` (PR #666) |
| 1 | #306 | server/routes/corpus_library.py | 395 | 2026-04-25 | Type 1: ``root`` sanitized via ``_resolve_corpus_root`` → ``resolve_corpus_path_param`` (normpath+startswith anchor); ``.resolve()`` on the already-anchored path. Dismissed ``gh api`` (PR #675) |
| 1 | #307 | server/routes/corpus_library.py | 401 | 2026-04-25 | Type 1: ``target = os.path.normpath(os.path.join(root_s, bridge_relative_path))`` followed by inline ``target.startswith(root_s + os.sep)`` prefix-guard before ``open()``. Dismissed ``gh api`` (PR #675) |
| 1 | #308 | server/routes/operator_config.py | 136 | 2026-04-28 | Type 1: ``corpus_root`` from ``resolve_corpus_path_param`` (normpath + startswith anchor) immediately before ``corpus_root.mkdir(parents=True, exist_ok=True)`` on GET handler — auto-create restricted to subdirs under the configured corpus root (#693 first-run UX). Dismissed ``gh api`` (PR #702) |
| 1 | #309 | server/routes/operator_config.py | 195 | 2026-04-28 | Type 1: same sanitizer chain as #308, mirror on PUT handler. Dismissed ``gh api`` (PR #702) |
| 1 | #311 | server/routes/scheduled_jobs.py | 48 | 2026-05-02 | Type 1: ``corpus`` from ``_resolve_corpus_root`` → ``resolve_corpus_path_param`` (normpath+startswith anchor); ``.resolve()`` on already-anchored ``Path`` before ``os.path.normpath``. Same shape as ``routes/jobs.py`` and ``routes/corpus_library.py`` #306. Dismissed ``gh api`` (PR #707, #708) |
| 2 | #319 | .github/workflows/drill-infra-destroy.yml | 85 | 2026-05-12 | Type 2: tfstate artifact download in same workflow_call chain (same run_id); only repo admins can trigger; no external input controls artifact content |
| 2 | #320 | .github/workflows/drill-infra-destroy.yml | 103 | 2026-05-12 | Type 2: tfstate artifact download in same workflow_call chain (same run_id); only repo admins can trigger; no external input controls artifact content |
| 3 | #298 | docker/pipeline (lcms2/liblcms2-2@2.16-2) | — | 2026-05-02 | Type 3: SNYK-DEBIAN13-LCMS2-16104015 (CVE-2026-41254 incorrect-behavior-order). Transitive system dep via ffmpeg / image libs. Pipeline processes audio + text only; no PIL/Pillow image color-management invocation in src/ (``grep -r "from PIL"`` empty). Latest in Debian 13 trixie apt index; ``apt-get upgrade`` would auto-pull a backport once published. Dismissed ``gh api`` (won't fix; not reachable). |
| 3 | #312 | docker/pipeline (gnutls28/libgnutls30t64@3.8.9-3+deb13u2) | — | 2026-05-02 | Type 3: SNYK-DEBIAN13-GNUTLS28-16344314 (CVE-2026-33845 DTLS handshake integer underflow). Snyk explicitly: "no fixed version for Debian:13 gnutls28". Pipeline uses HTTP/HTTPS via httpx + requests (OpenSSL TLS), not gnutls's DTLS path; we do not accept inbound DTLS handshakes. Dismissed ``gh api`` (won't fix; not reachable). |
| 1 | #327 | server/pathutil.py | 116 | 2026-05-24 | Type 1: ``corpus_s`` inline ``normpath`` + ``startswith(safe_prefix)`` under ``anchor_str`` before manifest join (``read_manifest_produced_by_under_anchor``; PR #815) |
| 1 | #328 | server/pathutil.py | 125 | 2026-05-24 | Type 1: ``manifest_s`` inline ``normpath`` + ``startswith(safe_prefix)`` before ``os.path.isfile`` (same function; PR #815) |
| 1 | #329 | server/pathutil.py | 129 | 2026-05-24 | Type 1: ``manifest_s`` same inline sanitizer chain before ``read_text`` (PR #815) |
| 1 | #330 | server/pathutil.py | 115 | 2026-05-24 | Type 1: post-refactor ``corpus_s`` inline ``normpath`` + ``startswith`` (PR #815) |
| 1 | #331 | server/pathutil.py | 123 | 2026-05-24 | Type 1: ``manifest_s`` inline sanitizer before ``os.path.isfile`` (PR #815) |
| 1 | #332 | server/pathutil.py | 126 | 2026-05-24 | Type 1: ``manifest_s`` inline sanitizer before ``read_text`` (PR #815) |
| 1 | #342 | search/backends/lancedb_backend.py | 131 | 2026-06-01 | Type 1: ``meta_path`` via ``normpath_if_under_root`` after ``safe_resolve_directory`` + ``safe_relpath_under_corpus_root`` (constant ``index_meta.json``) before ``open`` in ``read_index_meta``; corpus root confined at the route by ``resolve_corpus_path_param`` (raises on escape). Same shape as ``jobs_log_path:74``. Dismissed ``gh api`` (PR #865) |
| 1 | #338 | search/backends/lancedb_backend.py | 135 | 2026-06-01 | Type 1: same sanitizer chain, ``os.path.isfile(meta_path)`` sink in ``read_index_meta`` (PR #865) |
| 1 | #341 | search/hybrid_search.py | 172 | 2026-06-01 | Type 1: ``index_dir_str`` via ``normpath_if_under_root`` after ``safe_resolve_directory`` + ``safe_relpath_under_corpus_root`` (constant ``search/lance_index``) before ``os.path.isdir`` in ``hybrid_candidates``; corpus root confined at the route by ``resolve_corpus_path_param``. Dismissed ``gh api`` (PR #865) |
| 1 | #343 | kg/corpus.py | 40 | 2026-06-05 | Type 1: new ``/api/relational/*`` routes confine the corpus ``path`` via ``resolve_corpus_path_param`` (raises on anchor escape) before ``get_corpus_graph`` → ``CorpusGraph.build`` → ``scan_kg_artifact_paths``; reads only files under ``<corpus>``. Same class as #865. Dismissed ``gh api`` (PR #890) |
| 1 | #344 | kg/corpus.py | 44 | 2026-06-05 | Type 1: same sanitizer chain, ``load_kg_artifacts`` read under ``<corpus>`` (PR #890) |
| 1 | #345 | kg/corpus.py | 45 | 2026-06-05 | Type 1: same sanitizer chain, ``load_kg_artifacts`` read under ``<corpus>`` (PR #890) |
| 1 | #346 | kg/corpus.py | 46 | 2026-06-05 | Type 1: same sanitizer chain, ``load_kg_artifacts`` read under ``<corpus>`` (PR #890) |
| 1 | #347 | search/corpus_graph.py | 277 | 2026-06-05 | Type 1: ``get_corpus_graph`` cache/build on the route-confined corpus root (``resolve_corpus_path_param`` raises on escape); reads only ``<corpus>`` artifacts. Same class as #865. Dismissed ``gh api`` (PR #890) |
| 1 | #348 | search/query_log.py | 40 | 2026-06-05 | Type 1: ``search``/``query-activity`` routes confine the corpus root via ``resolve_corpus_path_param`` before ``append_query_event`` → ``mkdir`` under ``<corpus>/search/``. Same class as #343. Dismissed ``gh api`` (PR #896) |
| 1 | #349 | search/query_log.py | 41 | 2026-06-05 | Type 1: same sanitizer chain, ``append_query_event`` opens ``<corpus>/search/query_log.jsonl`` for append (PR #896) |
| 1 | #350 | search/query_log.py | 81 | 2026-06-05 | Type 1: same sanitizer chain, ``read_query_activity`` ``path.exists()`` on route-confined corpus root (PR #896) |
| 1 | #351 | search/query_log.py | 83 | 2026-06-05 | Type 1: same sanitizer chain, ``read_query_activity`` reads ``<corpus>/search/query_log.jsonl`` (PR #896) |
| 1 | #352 | server/routes/corpus_media.py | 67 | 2026-06-06 | Type 1: ``target`` from ``safe_relpath_under_corpus_root`` after ``media/`` prefix guard + ``resolve_corpus_path_param``; CodeQL cross-function taint gap before inline ``normpath_if_under_root``. Dismissed ``gh api`` (PR #898) |
| 1 | #353 | server/routes/corpus_media.py | 73 | 2026-06-06 | Type 1: ``root_s`` from ``safe_resolve_directory(root)`` after ``resolve_corpus_path_param`` anchor guard. Dismissed ``gh api`` (PR #898) |
| 1 | #354 | server/routes/corpus_media.py | 79 | 2026-06-06 | Type 1: ``verified`` from ``normpath_if_under_root(target, root_s)`` immediately before ``os.path.isfile``. Same shape as corpus_text_file #224. Dismissed ``gh api`` (PR #898) |
| 1 | #355 | server/routes/corpus_media.py | 85 | 2026-06-06 | Type 1: ``verified`` from ``normpath_if_under_root(target, root_s)`` immediately before ``FileResponse``. Same shape as corpus_text_file #228. Dismissed ``gh api`` (PR #898) |
| 1 | #357 | server/routes/corpus_media.py | 53 | 2026-06-06 | Type 1: ``_safe_media_target_str`` sink — ``safe_relpath_under_corpus_root(base, norm)`` after the ``media/`` prefix + suffix-allowlist guards; route confines ``root`` via ``resolve_corpus_path_param``. Traversal tests pass. Dismissed ``gh api`` (PR #901) |
| 1 | #358 | server/routes/corpus_media.py | 115 | 2026-06-06 | Type 1: stem-extension resolve (``_resolve_existing_media``) — each candidate re-verified by ``normpath_if_under_root`` + ``realpath`` containment before ``isfile``/``FileResponse``. Same class as #355. Dismissed ``gh api`` (PR #901) |
| 1 | #360 | search/backends/lancedb_backend.py | 332 | 2026-06-11 | Type 1: ``meta_path`` via ``normpath_if_under_root`` after ``safe_resolve_directory`` + ``safe_relpath_under_corpus_root`` (constant ``index_meta.json``) before ``os.path.isfile`` in ``stored_schema_version`` — identical shape to ``read_index_meta`` #338/#342, corpus root route-confined by ``resolve_corpus_path_param``. Dismissed ``gh api`` (PR #969) |
| 1 | #361 | search/backends/lancedb_backend.py | 335 | 2026-06-11 | Type 1: same sanitizer chain, ``open(meta_path)`` sink in ``stored_schema_version`` (PR #969) |
| 1 | #362 | search/backends/lancedb_backend.py | 359 | 2026-06-11 | Type 1: same sanitizer chain, schema-version helper filesystem sink on the route-confined corpus root (PR #969) |
| 1 | #363 | search/backends/lancedb_backend.py | 341 | 2026-06-11 | Type 1: same sanitizer chain, ``stored_schema_version`` sink re-numbered after the fix line-shift; ``meta_path`` via ``normpath_if_under_root``. Same shape as #338/#342 (PR #969) |
| 1 | #369 | server/routes/corpus_digest.py | 278 | 2026-06-15 | Type 1: corpus path sanitised upstream (resolve_corpus_path_param/safe_resolve_directory); index sub-path from constant suffix. FAISS→LanceDB refactor (#995, PR #1010) |
| 1 | #370 | gi/explore.py | 131 | 2026-06-15 | Type 1: ``default_vector_index_dir`` builds ``output_dir/search`` (constant suffix) on a validated root. #995, PR #1010 |
| 1 | #371 | search/index_pool.py | 45 | 2026-06-15 | Type 1: ``getmtime`` on a validated index_dir (resolve_corpus_path_param/safe_resolve_directory upstream). #995, PR #1010 |
| 1 | #372 | search/index_pool.py | 69 | 2026-06-15 | Type 1: ``get_lance_backend`` resolves a validated index_dir; cross-fn sanitiser. #995, PR #1010 |
| 1 | #374 | search/lance_index_stats.py | 38 | 2026-06-15 | Type 1: ``_dir_size`` os.walk on a validated lance_dir under the sanitised corpus root. #995, PR #1010 |
| 1 | #375 | search/lance_index_stats.py | 41 | 2026-06-15 | Type 1: os.path.getsize under a validated lance_dir. #995, PR #1010 |
| 1 | #376 | search/lance_index_stats.py | 50 | 2026-06-15 | Type 1: ``read_lance_index_stats`` is_dir on a validated lance_dir. #995, PR #1010 |
| 1 | #377 | search/lance_index_stats.py | 82 | 2026-06-15 | Type 1: LanceDB open on a validated lance_dir; sanitised upstream. #995, PR #1010 |
| 1 | #382 | server/routes/index_stats.py | 145 | 2026-06-20 | Type 1: ``GET /index/timeseries`` builds ``search/lance_index`` (constant suffix) on a corpus path sanitised by ``resolve_corpus_path_param`` (normpath + startswith-anchor). Mirrors #166. PR #1038 |
| 1 | #383 | search/lance_index_stats.py | 102 | 2026-06-20 | Type 1: ``read_lance_doc_type_by_month`` is_dir/LanceDB open on a validated lance_dir; sanitised upstream. Mirrors #376/#377. PR #1038 |
| 1 | #384 | search/corpus_graph.py | 414 | 2026-06-23 | Type 1: re-fire of #347 — ``get_corpus_graph`` cache/build on the route-confined corpus root (``resolve_corpus_path_param`` raises on anchor escape); reads only ``<corpus>`` artifacts. The ``reconcile_hosts`` (#1056) cache-key addition shifted the sink line 277→414, so CodeQL re-attributed the already-dismissed false positive to the PR. Sanitiser chain unchanged. Dismissed ``gh api`` (PR #1059) |

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

**List open alerts for a PR (required when Security tab shows PR-only CodeQL):**
the default ``GET …/code-scanning/alerts?state=open`` is for the default branch;
PR findings often appear only on ``refs/pull/<N>/merge``.

```bash
gh api 'repos/chipi/podcast_scraper/code-scanning/alerts?state=open&ref=refs/pull/<N>/merge&per_page=100' \
  -q '.[] | "\(.number) \(.rule.id) \(.most_recent_instance.location.path):\(.most_recent_instance.location.start_line)"'
```

**Dismiss one alert by number** (same alert id repo-wide; comment should cite Type and doc):

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
