# Prod validation & observability — quick reference

**When to read this:** you (or an agent) just did something to prod — deployed, swapped
the corpus, restarted a stack — and need to answer *"is it actually good?"* in 2 minutes,
not 15. **Almost none of this needs prod SSH.** Full detail: [Prod Runbook](PROD_RUNBOOK.md),
[Observability Runbook](OBSERVABILITY_RUNBOOK.md).

## Surfaces (how to reach prod)

| Surface | Reach | Gate |
| --- | --- | --- |
| Operator/compose API | `https://prod-podcast.<TAILNET>.ts.net` (tailnet) | ungated — use this for all corpus checks |
| Public player | `https://closelistening.app` | `/preview` cookie gate |
| Player MCP | `https://mcp.closelistening.app` | OAuth (tool calls need a token) |
| Homelab (obs) | `homelab` = `<HOMELAB_IP>` (tailnet) | services below are tailnet-open |
| Prod box (only for mutations) | `ssh -i ~/.ssh/podcast_prod_operator -o IdentitiesOnly=yes deploy@prod-podcast.<TAILNET>.ts.net` | key is **transient — operator re-adds on request**. `make` is NOT on the box; use `docker exec compose-api-1 python -m podcast_scraper.cli …` |

In-container corpus path is `/app/output` (pass `path=/app/output` to API probes).

## "I changed the corpus — is prod serving the RIGHT one?" (no SSH)

```sh
B=https://prod-podcast.<TAILNET>.ts.net; Q=path=/app/output

# 1. Identity — is the INTENDED corpus served? (not just "healthy")
curl -s "$B/api/health" | jq '{code:.code_version, produced:.corpus_produced_by.produced_at, corpus_ver:.corpus_code_version, warn:.corpus_version_warning}'

# 2. Episode count
curl -s "$B/api/corpus/episodes?$Q&limit=500" | jq '.items|length'

# 3. Index freshness + completeness — AUTHORITATIVE, no reindex-guessing
curl -s "$B/api/index/stats?$Q" | jq '{reindex:.reindex_recommended, reasons:.reindex_reasons, last_updated:.stats.last_updated, artifact_newest:.artifact_newest_mtime, counts:.stats.doc_type_counts}'
#   reindex_recommended:false + counts sane => index is fine. It compares index
#   last_updated vs newest gi.json/kg/transcript mtime (index_staleness.py).
#   REINDEX (remote, no SSH) — builds the two-tier index; poll stats for rebuild_in_progress:
#     curl -X POST "$B/api/index/rebuild?$Q&rebuild=true"   # 202; 409 if already running

# 4. Topic clusters present (smoke fails without this) + search works
curl -s "$B/api/corpus/topic-clusters?$Q" | jq '.clusters|length'
curl -s "$B/api/search?q=markets&$Q&limit=3" | jq '.results|length'

# 5. Or just run the 6-surface smoke (what deploy-prod runs):
bash scripts/ops/post_deploy_smoke.sh prod-podcast.<TAILNET>.ts.net --corpus-path /app/output
```

## Observability checkup (no SSH except GlitchTip/Umami DB)

Signals all land on **homelab**; query over tailnet. Log labels: `app=podcast|player`,
`surface=api|mcp|web|scheduler`. Note: API access logs now carry `trace_id` (shipped in
PR 1527) and player-api IS traced (older docs say otherwise).

```sh
# Metrics — api up?
curl -s "http://homelab:8428/api/v1/query?query=up{instance='prod-podcast'}" | jq '.data.result[].value[1]'
# Traces — services reporting
curl -s http://homelab:10428/select/jaeger/api/services | jq .data
# Logs — errors across a surface, deploy window
curl -sG http://homelab:9428/select/logsql/query \
  --data-urlencode 'query={app="player"} _time:3h (error OR Exception OR Traceback OR " 500 ")' --data-urlencode limit=20
```

GlitchTip (errors) + Umami (UX) need their DB (docker is at `/usr/local/bin/docker`, NOT on
the non-interactive PATH; GlitchTip pg user = `glitchtip`, projects **1=podcast/operator,
5=player**; use `make_interval()` to dodge SSH quote-mangling):

```sh
D=/usr/local/bin/docker
# New errors since deploy in podcast+player (empty / 0 rows = clean)
ssh -i ~/.ssh/homelab_mini homelab "$D exec glitchtip-postgres-1 psql -U glitchtip -d glitchtip -tAc \
  'select project_id,count(*),max(first_seen) from issue_events_issue where first_seen > now() - make_interval(hours=>14) group by 1 order by 1'"
# Umami — player UX traffic last 24h
ssh -i ~/.ssh/homelab_mini homelab "$D exec umami-db psql -U umami -d umami -tAc \
  'select count(*) from website_event where created_at > now() - make_interval(hours=>24)'"
```

Grafana UI: `http://homelab:3000` (folders *Podcast Operator*, *Podcast Player*).

## Gotchas that cost time (one-liners)

- **Corpus = a bind-backed docker volume** `compose_corpus_data` → `/srv/podcast-scraper/corpus`,
  **shared by compose+operator+player**. Never `mv` the dir to swap it (the mount follows the
  inode → serves the `.bak`); replace contents in place. Re-resolving the bind needs ALL
  consumers down + `docker volume rm` → redeploy. See [Corpus snapshot & restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).
- **Secrets are VIA_FILES** in host `/dev/shm` (empty between deploys; only the deploy
  workflows restage from GH Secrets). Never hand-run `compose up` with the secrets overlay.
- **Deploy scripts use `up -d` (no `--force-recreate`)** → a same-image redeploy is a no-op.
  A corpus cutover needs an image change (or the volume-rm dance).
- **`topic_clusters.json` is a separate step** nothing runs automatically —
  `python -m podcast_scraper.cli topic-clusters --output-dir <corpus> --threshold 0.75`
  (0.75 = profile value; the viewer-validation Make target's 0.35 is a fixture override).
- **`deploy-prod` has a `prod` GitHub environment gate**; approve via
  `gh api repos/chipi/podcast_scraper/actions/runs/<id>/pending_deployments -f state=approved -F environment_ids[]=<id>` (operator-authorized only).
- **`docker manifest inspect` before deploying a sha** — `stack-test.yml` publishes images;
  `docker.yml` is `push:false`. `deploy-prod` preflight validates api+viewer+**pipeline-llm**.

## When you actually need prod SSH

Most mutations are already remote APIs — **reindex** is `POST /api/index/rebuild` (above),
no SSH. The genuine box-only exceptions today:

- **Generating `topic_clusters.json`** — CLI-only, NO API/button (gap; the fix is to have
  `/api/index/rebuild` regenerate it). Until then: `docker exec compose-api-1 python -m
  podcast_scraper.cli topic-clusters --output-dir /app/output --threshold 0.75`.
- **Invoking the gated player MCP tools** (needs the MCP OAuth token instead, ideally).
- Raw container ops / inspecting the on-disk corpus dir.

All *validation* above is doable without SSH.
