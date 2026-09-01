# Ingestion Runbook — nightly top-up vs. deliberate backfill

How episodes get selected, the two strategies that exist, and how to control each. Also: how
to get a working `viewer_operator.yaml` onto a freshly installed box.

Written after a 2026-09-01 near-miss where a corpus-wide setting would have turned the
nightly into a back-catalogue crawler. The two strategies below want **opposite** values for
the same knob, which is the whole reason this document exists.

---

## 1. The two strategies

| aspect | **nightly top-up** | **deliberate backfill** |
| --- | --- | --- |
| goal | stay current | catch a feed up |
| selection | `position` (default) | `unprocessed` |
| what the cap means | newest-N **window** | N episodes of **work** |
| back catalogue | unreachable by construction | that's the point |
| trigger | scheduler (`scheduled_jobs`) | you, per feed, on purpose |

### Why the nightly must stay positional

`max_episodes: 10` in the operator YAML is a **window on the top of the feed**: take the
newest 10 by position, then `skip_existing` drops the ones already on disk. Usually 0–2
episodes are actually new. The window is what makes the back catalogue unreachable —
however long the gap, a nightly can never reach past the newest 10.

> Comment on that key in the live operator YAML:
> *"Newest-N WINDOW per feed (items[:N] before skip logic) — the ultimate control against
> back-catalog ingestion: a nightly can never reach past the newest 10, however large the gap."*

### Why a backfill must be `unprocessed`

`unprocessed` drops already-ingested episodes **by guid first**, then applies the cap
(`workflow/stages/scraping.py:507` before `:513`). So `max_episodes=10` means ten episodes of
*actual work*, immune to feed movement. Without it, a run asking for 10 delivers fewer
whenever the feed grew — measured 8,8,8,8,7,7,7 on 2026-08-31.

### ⚠ The trap: never set `unprocessed` corpus-wide

Setting `episode_selection: unprocessed` in `viewer_operator.yaml` applies to **every feed
and every run, including the nightly**. Once the newest N are all on disk, the newest
*un-ingested* item is deep in the archive — so the nightly ingests 10 **old** episodes per
feed per night, forever, until each feed is exhausted. On a 1000-episode feed that is ~100
unattended nights of download + GI spend, with no error at any point.

**Set it per request instead** (§3). If you must set it globally for a batch, the nightly has
to be `enabled: false` for the whole window, and you revert immediately afterwards.

### ⚠ Never combine an offset with `unprocessed`

An `episode_offset` under `unprocessed` skips episodes you have **not** ingested. It is not
rejected — "drop what I have, then skip the newest N of what is left" is coherent — so
`config.py` warns and proceeds:

> `episode_offset=10 is set together with episode_selection=unprocessed. The offset still
> applies, but it is POSITIONAL … you will skip 10 episodes you have NOT ingested.`

Migrating a positional recipe? **Drop the offset.** Watch for that warning in the log.

---

## 2. Where selection can be set (cascade)

Lowest to highest precedence:

1. **profile** (`config/profiles/*.yaml`) — baked into the image; changing it needs a rebuild (#1885)
2. **corpus operator YAML** (`/app/output/viewer_operator.yaml`) — runtime, **corpus-global**, always passed as `--config`
3. **per-request** `POST /api/jobs?episode_selection=` — scoped to one run.
   Added on main in `998d5312f`; **verify it is deployed before relying on it** — FastAPI
   silently ignores unknown query params, so against an older build the parameter is
   accepted, dropped, and the run quietly uses positional selection. Check with:
   `curl -s https://<fqdn>/openapi.json | grep -c episode_selection`
4. **CLI** `--episode-selection` — direct invocation

Rule of thumb: **`profile` is a property of the feed** (durable — "this feed runs on DGX").
**`episode_selection` is a property of the run's intent** (not durable — "this run is a
backfill"). Pinning intent as durable config is what creates the trap above.

---

## 3. Running a backfill

Ten feeds, ten new episodes each. The API **is** reachable over HTTPS via the Caddy edge
(`https://<prod-magicdns-fqdn>/api/...`) — `docker ps` showing port 8000 unbound on the host
means only that it is not *directly* bound; the reverse proxy still fronts it. What gates you
is the credential, not the network. The operator key lives on the box at
`/run/secrets/app_operator_api_key` (64 bytes, staged by `deploy-prod`), so either read it
there over SSH, or hold your own copy and call the HTTPS endpoint directly.

**Where the key is.** On the operator's laptop at `~/podcast_operator_api_key.txt`
(0600, 64-char hex from `openssl rand -hex 32`). The prod host carries its own copy at
`/run/secrets/app_operator_api_key`, staged by `deploy-prod`. Either reaches the API; the
laptop route needs no SSH. Strip the trailing newline or the header is 65 chars and 403s:

```bash
KEY=$(tr -d ' \n\r' < ~/podcast_operator_api_key.txt)
```

Over HTTPS, using that key:

```bash
curl -s -X POST -G "https://<fqdn>/api/jobs" \
  -H "X-Operator-Key: $APP_OPERATOR_API_KEY" \
  --data-urlencode "feed=<RSS>" \
  --data-urlencode "max_episodes=10" \
  --data-urlencode "skip_existing=true" \
  --data-urlencode "episode_selection=unprocessed"
```

Or read the key on the box and call the loopback API:

```bash
ssh -i ~/.ssh/podcast_prod_operator deploy@<prod-ip> 'docker exec -i compose-api-1 python3 - <<PY
import pathlib, urllib.request, urllib.parse
key = pathlib.Path("/run/secrets/app_operator_api_key").read_text().strip()
qs = urllib.parse.urlencode({
    "feed": "<RSS>",
    "max_episodes": 10,
    "skip_existing": "true",
    "episode_selection": "unprocessed",   # per-request; nightly unaffected
})
req = urllib.request.Request("http://127.0.0.1:8000/api/jobs?" + qs,
    method="POST", headers={"X-Operator-Key": key})
print(urllib.request.urlopen(req).read().decode())
PY'
```

Omit `profile=` to ride the feed's own pin. `job_id == run_id` — the observability join key.

**Confirm the mode engaged** — the first selection line says so:

```text
episode_selection=unprocessed: 20 feed item(s) already ingested and dropped BEFORE the limit
Episodes to process: 10 of 95 (after order/date filter/unprocessed-filter/offset/limit)
```

`unprocessed-filter/` present = active (absent under positional). If instead you see the
offset warning, kill the run and drop the offset.

**Jobs run sequentially** (single-writer queue). A ten-feed batch submits immediately but
executes one at a time, and each job reads `--config` when *it* starts — so if you set the
selection mode in the corpus YAML as a stopgap, **do not revert it until the last job has
started**, or the queued tail silently falls back to positional.

**Brake:** `POST /api/jobs/stop` — pauses the queue, SIGTERMs, verifies.

**Editing the operator YAML without SSH:** `GET`/`PUT /api/operator-config?path=/app/output`
(body `{"content": "<full yaml>"}`). Save the original first — an empty/misnamed body field
once wiped prod's config twice, which is why empty content over a non-empty file is now
refused.

---

## 4. Seeding `viewer_operator.yaml` on a fresh box

**There is no production default in git.** Only examples:

| file | purpose |
| --- | --- |
| `config/examples/viewer_operator.example.yaml` | packaged example |
| `config/examples/viewer_operator.docker.example.yaml` | docker variant |
| `config/ci/stack-test-seed/viewer_operator.yaml` | stack-test seed only |

**The API seeds the real file from the packaged example on first GET.** So:

1. bring the stack up
2. **open the operator viewer once** — this creates `<corpus>/viewer_operator.yaml`
3. re-run `deploy-prod` so it re-pins `litellm_api_base`
4. re-apply your tuned values (§5)

`deploy-prod` deliberately refuses to create it:

> `# Never CREATE this file. The api seeds it from the packaged example on first GET; a file
> containing only our one key would strip pipeline_install_extras and every other operator
> default, which is a worse failure than the one being fixed.`

and warns if absent:

> `::warning:: … does not exist — not creating it. Open the viewer once to seed it, then re-run this deploy.`

### ⚠ The live file is corpus state, not version control

Everything you tune there — `max_episodes`, `dgx_diarize_request_timeout_sec`, `profile`,
`scheduled_jobs`, the enrichment toggles — lives **only on the corpus volume**. A
restore-from-scratch gets the packaged example's defaults, not your tuned file. Only
`litellm_api_base` is re-pinned automatically by `deploy-prod`.

**So: after any rebuild or DR restore, diff the seeded file against your intended values
before enabling the nightly.** Keep a copy of the tuned file somewhere durable; the corpus
backup covers it, but only as part of a full corpus restore.

---

## 5. Values prod runs today

Recorded so a rebuild can be checked against them (2026-09-01):

- `max_episodes: 10` — the newest-N window (do not raise casually; it is a safety bound)
- `skip_existing: true`
- `profile:` — corpus default for feeds with no per-feed pin
- `dgx_diarize_request_timeout_sec: 900` — the duration probe is dead in prod (no `soundfile`)
- `vllm_api_base` / `litellm_api_base` — pinned to prod's own gateway, NOT the profile's homelab pin
- `scheduled_jobs: nightly-ingest`, `cron: "0 3 * * *"` — **`enabled: false`** as of 2026-09-01

Per-feed profile pins live in `<corpus>/feeds.spec.yaml` as `profile:` under a feed's `url:`.
Feeds listed as bare URLs inherit the corpus default.

---

## 6. Before re-enabling the nightly

1. `episode_selection` **absent** from `viewer_operator.yaml` (or explicitly `position`)
2. `max_episodes` is a sane window
3. no `episode_offset` set globally
4. one supervised fire watched end to end — confirm the selection line shows a small number,
   not a double-digit back-catalogue pull
