# Storylines in the search index — post-deploy procedure (2026-09-17)

**Audience:** whoever operates prod after the image carrying this change is deployed.
**TL;DR:** run ONE additive index command. It is **not** a rebuild, it does **not** start the
app, and it does **not** need a schema migration or an api restart.

---

## What changed in the code

Storylines (theme clusters) are now **rows in the LanceDB index**, not just something the
entity resolver could name. Three pieces:

- `search/theme_clusters.py` — `storyline_index_rows(corpus_root)` emits **one row per
  cluster** (`doc_type="storyline"`, `episode_id=None`, embed text = canonical label +
  member labels).
- `search/two_tier_indexer.py` — `_append_storyline_rows()` runs once per build after the
  episode loop; `_prune_orphaned_storylines()` removes rows for clusters that were
  relabelled or re-anchored; a `::storylines` fingerprint key makes an unchanged rebuild a
  no-op.
- `search/corpus_search.py` — `_attach_storyline_metadata()` joins `storyline_label`,
  `storyline_size` and `anchor_topic_id` onto hits at query time.

Until the index is rebuilt with this code, prod has **zero** `doc_type="storyline"` rows and
the Search surface shows no Storylines section. Nothing else regresses.

---

## The command

Run it **after** the new image is live (the row-emitting code ships in the image).

```bash
cd /srv/podcast-scraper
SEC=''; [ -n "$(ls -A /dev/shm/podcast-secrets 2>/dev/null)" ] && SEC='-f compose/docker-compose.secrets.yml'
docker compose --env-file .env \
  -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.prod.yml \
  -f compose/docker-compose.vps-prod.yml \
  $SEC \
  run --rm --no-deps api \
  python -m podcast_scraper.cli index --output-dir /app/output
```

`run --rm --no-deps` creates ONE throwaway container from the api **image**, starts no
dependencies, and deletes itself on exit. The running api/player containers are never
stopped or restarted. This is the same invocation shape `.github/workflows/reindex-prod.yml`
already uses for every prod index command.

### Read the one log line it emits

```
INFO index: scanned=<N> skipped=<N> reindexed=0 vectors=<small>
```

- `reindexed=0` (or near 0) → correct: storyline rows only, nothing re-embedded.
- `reindexed` in the hundreds → **stop**. Something (embedding model or chunk config) does
  not match what built the existing index, and the run is re-embedding the corpus. Upserts
  are idempotent, so interrupting is safe.

---

## Do NOT use these verbs

| Verb | Why not |
| ---- | ------- |
| `cli index-two-tier` | `cli_handlers.py:1541` passes `drop_existing=True` → **full rebuild from scratch**. It also uses the builder's default chunk params instead of the config's, which disagrees with what the pipeline built. Measured on the fixture corpus: switching verbs re-embedded 32/36 episodes and moved segments 131→115. |
| `cli index --rebuild` | `indexer.py:558` `rmtree`s the index directory and deletes the fingerprint sidecar. Irreversible, and unnecessary — this change is purely additive. |
| `reindex-prod.yml` in `mode=rebuild` | Same `rmtree`, plus it holds the `prod-corpus` lock. Reserve it for genuine corruption repair. |
| `POST /api/index/rebuild` | Full rebuild, not additive. |

---

## Run it as the container user, never `docker exec` as root

The api process runs as `podcast` (uid 1000). `docker compose run` / `docker compose exec`
both use the image's configured user, so this is handled for you.

If anyone runs the CLI via `docker exec` **as root**, the new Lance fragment/manifest files
are written root-owned and the api can no longer read them. The symptom is not an error —
it is silent:

```
GET /api/index/stats  ->  {"available": false, "reason": "index_unreadable"}
GET /api/search?...   ->  {"error": "no_index", "results": []}
```

i.e. **search goes dark**. This was hit during the local rehearsal: 93 root-owned files under
`lance_index`. Recovery is `chown -R podcast:podcast /app/output/search`.

---

## Precondition

Storyline rows are read from `enrichments/topic_theme_clusters.json` (written by
`TopicThemeClustersEnricher`). If that artifact is missing on prod, the run completes
successfully and indexes **zero** storylines.

Check first, read-only:

```bash
curl -s https://<prod-host>/api/corpus/theme-clusters | head -c 300
```

Note `--with-clusters` does **not** help here — it re-derives `search/topic_clusters.json`,
a different artifact (semantic clusters, not theme clusters).

---

## Verification (read-only, after the run)

```bash
# 1. index readable
curl -s https://<prod-host>/api/index/stats | jq '{available, reason}'
#    expect: {"available": true, "reason": null}

# 2. a storyline actually surfaces — query a MEMBER topic of a known cluster
curl -s 'https://<prod-host>/api/search?q=<member+topic>&top_k=60' \
  | jq '[.results[] | select(.metadata.doc_type=="storyline")
         | {label: .metadata.storyline_label,
            anchor: .metadata.anchor_topic_id,
            size:   .metadata.storyline_size}]'
```

A correct hit carries **all three** of `storyline_label`, `storyline_size` and
`anchor_topic_id`. `anchor_topic_id` is the routable id the client opens
(`/storyline/<anchor_topic_id>`) — the `thc:` cluster id is label-derived and unstable, and
is deliberately NOT the route param.

---

## Safety, rollback, repeatability

- **No schema migration.** `LANCE_SCHEMA_VERSION` stays at 3; `episode_id` was already
  nullable in the aux schema. Verified `stored=3 expected=3`, `lance_index_is_stale=False`.
- **Idempotent.** Rows upsert by id. Re-running is safe; an unchanged re-run is a ~1s no-op
  (`skipped=N reindexed=0 vectors=0`), because of the `::storylines` fingerprint.
- **Nothing is deleted** except storyline rows whose cluster no longer exists
  (`_prune_orphaned_storylines`). Episode/segment/insight rows are untouched.
- **No restart needed.** The warm backend pool (`search/index_pool.py`) invalidates on the
  index directory's mtime, so a live server picks the new rows up on the next query.
- **Rollback** = redeploy the previous image. The storyline rows can stay; older code simply
  never queries for `doc_type="storyline"`. If you want them gone, the next index run on old
  code prunes them as orphans.
- **Not one-off / not dead code.** `_append_storyline_rows` runs on every build, and the
  pipeline's own `_finalize_pipeline` → `maybe_index_corpus` → `index_corpus` reaches the
  same path. The next scheduled pipeline run would pick storylines up anyway — this command
  only makes it immediate.

---

## Evidence behind the above (local rehearsal, 2026-09-17)

Rehearsed in the `lp-e2e-api` container against a real rebuilt LanceDB index, on a
prod-shaped baseline (index built by `cli index`, i.e. config chunk params), with the
storyline row pruned and the `::storylines` fingerprint key removed to reproduce prod's
pre-migration state:

```
INFO index: scanned=36 skipped=36 reindexed=0 vectors=1     ELAPSED=5s
BEFORE {'segments': 115, 'insights': 124, 'aux': 716}
AFTER  {'segments': 115, 'insights': 124, 'aux': 717}
```

Then, against the **live** server with no restart:

```
results=60
  rank 19 STORYLINE: Managing risk across domains | anchor: topic:risk-management
```

---

## NOT verified / NOT covered

- **Nothing here was run against prod.** Every number above comes from the 36-episode
  fixture corpus in a local container.
- **Prod runtime is unmeasured.** The fingerprint scan is proportional to episode count even
  though embedding is not; 5s on 36 episodes says nothing about prod's corpus.
- **Whether prod's fingerprints match is unverified.** Both prod write paths (pipeline
  auto-index and `reindex-prod.yml` rebuild mode) go through `index_corpus` with config chunk
  params, so they should match — but "should" is the reason the `reindexed=` check above
  exists. Read that line rather than trusting this paragraph.
- **How many storylines prod will index is unknown.** The fixture yields exactly 1. Prod's
  count depends on its `topic_theme_clusters.json`.
- **Ranking is not tuned.** On the fixture, storylines land mid-list for a member-topic query
  (rank 17–19 of 60) and are buried below `top_k=10` for high-frequency terms. Measured
  fixture ranks: safety practices=5, endurance sport=5, exact name=9, risk management=42,
  risk=57. If prod storylines appear too deep to be useful, that is a ranking question, not a
  failed migration.
- **`docker compose exec` form is unrehearsed.** The local equivalent used
  `docker exec -u podcast`. Prefer the `run --rm --no-deps` form above, which is the shape
  already proven on prod by the existing workflow.
