# Fresh pilot run + observe it (agent runbook)

**Goal:** an agent (or a person) can run a small, fresh, end-to-end pipeline pilot —
fetch + transcribe + diarize + name + GI + KG for **one latest episode per feed** — into a
new corpus folder, and observe it live, with **zero bespoke setup**. Use this to validate
a code change on real data before a full corpus run.

Related: [Corpus reprocessing runbook](CORPUS_REPROCESSING.md) (rebuilding an *existing* corpus),
[Observability control plane](OBSERVABILITY_CONTROL_PLANE.md) (`podcast_obs`),
[DGX Spark runbook](DGX_RUNBOOK.md).

## Prerequisites (check these first)

1. **DGX reachable** (turbo ASR + community-1 diarization run on it):

   ```bash
   curl -sS --max-time 8 -o /dev/null -w "dgx whisper -> HTTP %{http_code}\n" http://dgx-llm-1:8000/health
   ```

   Expect `HTTP 200`. If not, the DGX box / tailnet is down — the pilot cannot transcribe.

2. **Pipeline secrets** in `.env` (auto-loaded by `podcast_scraper.config`): the summary/GI/KG
   provider key (`GEMINI_API_KEY`) plus any provider the profile uses. `langfuse_enabled()` also
   reads `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` from here — set both to trace the run.

3. **Observability query keys** in `.env.obs.dev` (auto-loaded by `podcast_obs`): `LANGFUSE_*`
   for `podcast_obs traces`, `SENTRY_AUTH_TOKEN` for `errors`. The homelab Victoria* endpoints are
   tailnet-open (no token).

## Run the pilot

Always run through the **profile-driven CLI** — never a bespoke script (same rule as reprocessing).
`--max-episodes` is **per feed**, so `1` = one latest episode per feed.

```bash
# 1. new corpus folder + the 9-feed spec (copy an existing one)
mkdir -p .test_outputs/manual/prod-v2.3.1/corpus
cp .test_outputs/manual/prod-v2.1.1/corpus/feeds.spec.yaml \
   .test_outputs/manual/prod-v2.3.1/corpus/feeds.spec.yaml

# 2. dry-run first — validates feeds/paths/cost, transcribes nothing
.venv/bin/python -m podcast_scraper.cli \
  --config config/profiles/reprocess_v23_turbo.yaml \
  --feeds-spec .test_outputs/manual/prod-v2.3.1/corpus/feeds.spec.yaml \
  --output-dir .test_outputs/manual/prod-v2.3.1/corpus \
  --max-episodes 1 --dry-run

# 3. real run (drop --dry-run). ~2-5 min/episode on DGX turbo.
.venv/bin/python -m podcast_scraper.cli \
  --config config/profiles/reprocess_v23_turbo.yaml \
  --feeds-spec .test_outputs/manual/prod-v2.3.1/corpus/feeds.spec.yaml \
  --output-dir .test_outputs/manual/prod-v2.3.1/corpus \
  --max-episodes 1
```

- `reprocess_v23_turbo.yaml` = DGX turbo ASR + community-1 diarization + gemini summary/GI/KG, with
  `transcript_cache_enabled: false` (forces a fresh transcription). It writes every per-episode
  sidecar (`.metadata.json`, `.manifest.json`, `.speakers.diagnostics.json`, `.gi.json`, `.kg.json`,
  `.context.json`, `.bridge.json`).
- Output lands under `.test_outputs/` — **gitignored**; real-episode artifacts are never committed.

## Observe it (live)

The pipeline ships cost/logs/metrics to homelab VictoriaLogs/Metrics and LLM traces to Langfuse as
it runs. Query with `podcast_obs` (auto-loads `.env.obs.dev`, so it is turnkey from the worktree):

```bash
OBS="config/observability.homelab.yaml"
.venv/bin/python -m podcast_obs --config $OBS summary       # which sources are live
.venv/bin/python -m podcast_obs --config $OBS cost-today    # LLM cost events (VictoriaLogs)
.venv/bin/python -m podcast_obs --config $OBS traces        # Langfuse LLM traces
.venv/bin/python -m podcast_obs --config $OBS correlate --run-id <run_id>   # join per run
```

`health`/`runs`/`version` probe a **deployed** prod API and will fail for a local pilot — that is
expected, not an o11y fault.

## Validate the result

Per episode, `content.speakers` (name + role) plus the diagnostics sidecar (name + role +
**source**) give the named hosts/guests and how each was identified; the manifest `naming` block
carries the `exposed` metric (clean speaker output after cameo/commercial cleanup):

```bash
# named hosts/guests + how identified, per episode
for md in .test_outputs/manual/prod-v2.3.1/corpus/feeds/*/run_*/metadata/*.metadata.json; do
  .venv/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); \
    s=(d.get('content') or {}).get('speakers') or []; \
    print(d.get('feed',{}).get('title'), '->', \
      'hosts', [x['name'] for x in s if x['role']=='host'], \
      'guests', [x['name'] for x in s if x['role']=='guest'])" "$md"; done
```

## Clean up

The run spawns a background enrichment pass; if you kill the pilot mid-run, reap orphans per the
repo's process-hygiene rule (`make cleanup-processes`, CWD-scoped). Delete the corpus folder to
retry from scratch.
