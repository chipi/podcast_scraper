# Handover — prod is ready to scale (2026-09-07)

For whoever runs the next batch. Prod is healthy, idle, and unblocked. Read §2 before
launching anything: there is one trap that silently processes zero episodes and reports
success, and one that turns the nightly into a back-catalogue crawler.

## 1. State

| | |
|---|---|
| prod image | `sha-a02b03c` |
| API | `https://prod-podcast.tail6d0ed4.ts.net` — healthy, queue idle |
| corpus | 1,348 episodes |
| disk | **115 GB free of 150 GiB (21% used)** |
| audio archive | cold storage (Hetzner), **1,347 / 1,348** present |
| local audio cache | **0 files** — verified not regrowing (§3) |
| nightly ingest | **enabled**, `cron: 0 3 * * *` |

Storage cost is **~0.28 GB per 100 episodes** for the durable layer (transcripts, GI/KG,
metadata, search index). Audio no longer lands on local disk. 115 GB free is headroom for
roughly 40,000 more episodes — storage is not a constraint.

## 2. How to launch a batch — and the two traps

The operator key is on the operator's laptop at `~/podcast_operator_api_key.txt` (0600,
64-char hex). Strip the trailing newline or the header is 65 chars and 403s.

```bash
KEY=$(tr -d ' \n\r' < ~/podcast_operator_api_key.txt)
curl -fsS -X POST -G "https://prod-podcast.tail6d0ed4.ts.net/api/jobs" \
  -H "X-Operator-Key: $KEY" \
  --data-urlencode "feed=<RSS>" \
  --data-urlencode "max_episodes=10" \
  --data-urlencode "skip_existing=true"
```

Omit `profile=` — the ten DGX feeds carry `profile: prod_dgx_full` pins in
`feeds.spec.yaml` and the pin resolves on this route. `job_id == run_id`, the
observability join key. Jobs run **sequentially** (single-writer queue). Brake:
`POST /api/jobs/stop`.

### TRAP 1 — a run that processes zero and reports success

`max_episodes` counts **feed positions** by default. On a feed you have already ingested,
the newest N are all on disk, `skip_existing` drops them, and the run exits `ok` having
done nothing. Observed on 2026-09-07: an Odd Lots `max_episodes=1` job returned
`result: episodes=1 ok=1` with `ASR:0 ... cost:$0.0` — it processed nothing.

`episode_selection=unprocessed` fixes it by dropping already-ingested episodes **by guid
first**, so the cap counts episodes of real work. Two ways to set it:

* **CLI:** `--episode-selection unprocessed`
* **API:** merged to main as `998d5312f` but **NOT DEPLOYED**. FastAPI silently ignores
  unknown query params, so against `sha-a02b03c` the parameter is accepted, dropped, and
  the run quietly uses positional selection. Confirm before relying on it:
  `curl -s https://<fqdn>/openapi.json | grep -c episode_selection`

**For brand-new shows none of this matters** — nothing is on disk, so positional and
unprocessed are identical.

### TRAP 2 — do NOT set `episode_selection` in the corpus operator YAML

`/app/output/viewer_operator.yaml` is **corpus-global**. Setting `episode_selection:
unprocessed` there applies to every feed and every run **including the nightly**, which is
now enabled. Once the newest N are on disk, the newest *un-ingested* item is deep in the
archive, so the nightly ingests 10 **old** episodes per feed per night until each feed is
exhausted — ~100 unattended nights on a 1,000-episode feed, with no error at any point.

The nightly's safety property is that `max_episodes: 10` is a **newest-N window**, making
the back catalogue unreachable by construction. `unprocessed` removes exactly that.

If you must set it globally for a batch, disable the nightly first and revert immediately
after — jobs read `--config` when THEY start, so a queued tail picks up whatever the file
says at its own start time, not at submit time.

## 3. What was verified on 2026-09-07 (and what wasn't)

**Verified by observation, not reasoning:**

* A live episode download produced **0 audio-cache files**. `audio_storage_backend: remote`
  prevents local caching — this was measured after the fix, not merely traced in code.
* 47.28 GB of historical audio cache deleted after confirming 1,346/1,348 episodes were
  already in cold. Disk went 92 GB used → 30 GB.
* Cold-storage policy is pinned in the **operator YAML**, so it applies to every profile
  including a runtime `?profile=cloud_thin` switch. Deliberately not in `cloud_thin.yaml`:
  that profile also runs locally and in CI, where a remote backend has no credentials and
  `RcloneStorageBackend` fails loud by design (#1199).

**NOT verified / still open:**

* **1 episode missing from cold** — In Moscow's Shadows 261, `fetch_failed`. The backfill
  workflow has no per-episode input (only `feed` / `since` / `rate_limit`), so reaching it
  means scoping to the whole feed and running hundreds of no-op probes. The operator
  judges the episode is beyond the RSS window, in which case a publisher refetch cannot
  work at all. **Needs a system change, not a retry** — a per-episode target, or a source
  other than the live feed.
* **43 size-mismatch files** — local `media/` size differs from cold. Eviction correctly
  refuses to delete them. Local copies retained. Cause undetermined; I could not extract
  per-file sizes from available logs. Harmless where they sit.
* **`corpus-art` cache: 1,133 files, 828 MB** under `.podcast_scraper/corpus-art`. I have
  not checked whether anything prunes it. Far smaller than audio was. Flagged as
  unexamined, not as a known problem.
* **B2/B3 interaction** — a 10-insight quote bundle wants 6,400 output tokens and is
  clamped to 5,120 (~512/insight vs the 640 the analysis established). Measured 35
  truncations across the 100-episode batch, concentrated in 4 feeds (EconTalk 14, Sinica
  10, The Long Run 7). Upstream #1975 fixed the *input* side; the output cap is untouched.
  Raising 5,120 is **ruled out** — #1975 spent the headroom (26,726 + 1,024 + 6,400 =
  34,150 vs a 32,768 window). Capping bundles at 8 (8 × 640 = 5,120) fits exactly.

## 4. Undeployed work on main

`ad76608ab` is main's tip; prod runs `sha-a02b03c`. Notably undeployed:

* `998d5312f` — `episode_selection` as a per-request API param (removes TRAP 2 entirely)
* `15fe9673b` — `dev_dgx_full` / `dev_cloud_balanced` twins for local prod-parity runs

Deploy is `deploy-all-prod.yml` (`confirm=DEPLOY_ALL`, explicit `image_sha`,
`skip_control_plane=false`). Not required for anything in §2.

## 5. Reference

* [`INGESTION_RUNBOOK.md`](../guides/INGESTION_RUNBOOK.md) — nightly vs backfill strategy,
  the cascade, seeding `viewer_operator.yaml` on a fresh box
* Prod SSH: key `~/.ssh/podcast_prod_operator`, users `root` / `deploy`,
  `100.124.111.115`. **Do not probe usernames** — fail2ban banned this machine's IP on
  2026-09-07 after six failed auths in ~20s. Ports 443/8443 stay open (the jail is
  port-22-only), so `prod-ops-health` still passing means sshd is fine and only you are
  locked out.
* `docker exec … env` does **not** show the app's environment — the api image entrypoint
  is `secrets-shim.sh`, which exports `/run/secrets/*` at start-up. Reading the exec env
  and concluding "secrets are missing" is wrong; read `/run/secrets/<name>` directly.
