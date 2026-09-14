# Handover — enrichment o11y + the unwatched pipeline (2026-09-14)

Everything below is pushed. **Nothing is deployed.** One step needs the operator's hands
(§4) because it requires write access to a 0700 directory on the mini.

## 1. State

| | |
|---|---|
| `podcast_scraper` main | `43a7396e5` (4 commits today) |
| `agentic-ai-homelab` main | `ffb0cae` |
| prod running | **`sha-080403f`** — predates ALL four commits |
| prod corpus YAML | edited live; **experiment caps `max_persons: 3` / `max_orgs: 3`** |
| Grafana | dashboard + alerts pushed to repo, **not pulled onto the mini** |

### Today's commits (`podcast_scraper`)

* `46c64628b` — **deploy-all resolves ONE sha in the guard.** Was a read-before-write race:
  with an empty `image_sha`, `deploy-prod` resolved newest-published while
  `deploy-operator`/`deploy-player` read the sha RUNNING ON THE BOX, 63s before the control
  plane finished writing it. All 7 jobs green, prod on two shas for a day. **Verified fixed
  in production** — a deploy at `080403f` landed all 9 containers on one sha.
* `f60d5d3c8` — **WEB enrichers actually work.** Three defects: `org_web` in no profile set;
  `org_web` read `load_gi()` for `Organization` (a KG-only node type); the Wikidata lookup
  rejected real orgs (allowlist held only generic P31 shapes) and accepted wrong ones
  (`hits[0]` then veto → "Stanford" resolved the *town*). Measured 0/5 → 4/5.
* `f98e56362` — **enrichment events reach VictoriaLogs** (#2071, logs).
* `43a7396e5` — **enrichment metrics + traces** (#2071, closes it).

### `agentic-ai-homelab`

* `ffb0cae` — pipeline + enrichment panels on *Podcast Operator / Overview*, and a new
  `podcast-pipeline` alert group.

## 2. Why #2071 existed

Two parallel event paths. `grep -rn emit_event src/podcast_scraper/enrichment/` returned
**nothing**:

```
pipeline    -> emit_event(sink="log") -> stdout -> Alloy -> VictoriaLogs
enrichment  -> _safe_append_event()   -> enrichments/run.jsonl   (stops on the box)
```

Measured on prod over 24h: **0** enrichment lines in VictoriaLogs, **0**
`enrichment.*`/`enricher.*` series in VictoriaMetrics. Enrichment was observable *on demand*
via `/api/enrichment/*` but invisible to *alerting*. `org_web`'s `NameError` existed only in
`run.jsonl` while the central stack showed a `202` and nothing else.

Fixed at `EnrichmentExecutor._safe_append_event` — the single chokepoint all nine emission
sites funnel through, so coverage is uniform across every enricher by construction.

## 3. The bigger finding

Surveying **every** dashboard in the homelab Grafana, occurrences of:

```
enrichment 0   enricher 0   podcast_pipeline 0   pipeline_stage 0
ingest 0       run_id 0     asr 0    transcri 0    corpus 0
```

The product's core work was on **no dashboard at all** — and the data was already there,
unread: `podcast_pipeline_*` metrics have always been in VictoriaMetrics, and VictoriaLogs
holds ~2,600 `pipeline_stage` + ~14,000 `llm_cost` events per week. So the enrichment gap was
the smaller half.

## 4. THE ONE BLOCKED STEP — deploy Grafana

`claude` on the mini cannot do this. Its sudo was deliberately revoked, and
`~markodragoljevic/agentic-ai-homelab` is mode **0700** — `claude` IS in `staff`, but the group
bits are `---`, so shared group membership grants nothing. (The only ACL is `_dockerhost`
list,search.)

Operator runs:

```bash
ssh -t markodragoljevic@homelab 'cd ~/agentic-ai-homelab && git pull --ff-only && docker restart grafana'
```

`git pull` updates the bind-mounted files; the restart is required because Grafana reads
`provisioning/alerting` **only at startup** (dashboards auto-reload every 30s, alert rules do
not).

### Verify afterwards (no special access needed)

```bash
# panels present
curl -s https://grafana.tail6d0ed4.ts.net/api/health          # expect database: ok
# pipeline panels should populate immediately (that data predates today)
# enrichment panels stay EMPTY until §5 — they query podcast_enrichment_*, which
# only exists after f98e56362 + 43a7396e5 are deployed to prod.
```

Alert rules loaded: Grafana UI → Alerting → Alert rules → group **podcast-pipeline**
(`podcast-ingest-stalled`, `podcast-pipeline-jobs-failing`, `podcast-enricher-failing`).

## 5. Then: deploy prod

`PODCAST_METRICS_ENABLED` needs **no change** — `compose/docker-compose.prod.yml:122` defaults
it to `1`, and `/metrics` already serves 200 on prod with `podcast_pipeline_*` present. The new
`podcast_enrichment_*` families appear automatically once the image ships.

Deploy via `deploy-all-prod.yml` with an explicit `image_sha`. **Do not leave it empty** — the
guard now resolves it once, but an explicit sha from a green Stack test is still clearest.

After deploying, re-run the scoped enrichment experiment to prove `org_web` now derives:

```bash
KEY=$(tr -d ' \n\r' < ~/podcast_operator_api_key.txt)
curl -s -X POST -H "X-Operator-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"only":["person_web","org_web"],"corpus_only":true,"force":true}' \
  https://prod-podcast.tail6d0ed4.ts.net/api/jobs/enrichment
```

Expect `org_web` > 0 rows (it derived 0/5 on the old code, 4/5 locally on the new).

## 6. MUST DO — revert the experiment caps

Prod's `/srv/podcast-scraper/corpus/viewer_operator.yaml` currently has **`max_persons: 3` /
`max_orgs: 3`**, set for a scoped test. Left as-is, the nightly enriches only 3 of each.

```bash
ssh -i ~/.ssh/podcast_prod_operator deploy@100.124.111.115 \
  'cp /srv/podcast-scraper/corpus/viewer_operator.yaml.bak-preexperiment \
      /srv/podcast-scraper/corpus/viewer_operator.yaml && echo RESTORED'
```

That backup has `person_web: {}` / `org_web: {}` (defaults 200 each). Decide the real caps
first — **5 orgs took 45s** against rate-limited Wikidata, so 200 could be a long stage. Corpus
is 1,348 episodes; the distinct person/org count is **not measured**.

## 7. Verified today

* prod egress to Wikipedia/Wikidata **works** (raw payloads cached both sides)
* `person_web` end-to-end on prod: bio + hosted image + licence/artist attribution, surfacing
  in `build_person_card()` — the people panel's own view builder
* enrichment artifact envelopes are **structurally identical** to peers (13 keys, schema 1.0)
* health registry 9 → **11**; `/api/enrichment/{health,metrics,events,run-summary}` all carry
  the web enrichers
* full unit suite **11,790 passed, 44 skipped**; governance ratchet 0 undeclared, 0 stale

## 8. NOT verified / open

* **Nothing above is live.** Prod runs `sha-080403f`.
* `org_web` on prod still derives 0 (old code). Locally 4/5.
* The two orgs prod tried were `14th-amendment` and `15th-five-year-plan` — **KG extraction
  noise**, not organizations. Expect real-world derive rates well below 100%; the P31 guard
  refusing them is correct behaviour, not failure.
* Rendered UI not checked in a browser — the data path was validated, not pixels.
* `org_web` absent from `/api/enrichment/stats` — deploy lag only (the deployed profile set has
  10 entries, no `org_web`).
* **#2071 wording is wrong in one place**: it says "`pipeline_stage` 138 series". Those are
  `podcast_pipeline_run_*` histograms; `pipeline_stage` is a VictoriaLogs *event type*.
  Substance holds, label does not — worth correcting.
* The corpus-YAML-overrides-profile trap (`enrichment/cli.py:295`) is **unfixed**: a non-empty
  YAML enrichers list REPLACES the profile set, while `per_enricher_config` and `opt_in_flags`
  MERGE. That inconsistency is what silently kept the people enricher dark since 2026-09-11.
* Prod's corpus YAML carries `audio_cache_in_corpus: true`, a dev-twin setting. **Currently
  inert** (verified: 0 files, dir absent) because `audio_storage_backend: remote` wins — but it
  re-arms if anything runs without the remote backend.

## 9. Gotchas

* Prod SSH: `-i ~/.ssh/podcast_prod_operator deploy@100.124.111.115`. **Do not probe
  usernames** — fail2ban banned this laptop's IP on 2026-09-07.
* Operator key: `~/podcast_operator_api_key.txt`; strip the newline (`tr -d ' \n\r'`) or the
  header is 65 chars and 403s.
* `/api/app/*` routes need an **app-user session**, not the operator key — validate that path
  via `build_person_card()` in-container instead.
* The mini's remote shell is **zsh, which has a `log` builtin** — use `/usr/bin/log`.
* Local full-enricher runs segfault at interpreter teardown (`exit=139`, leaked semaphores
  after torch loads). A local ML-env artifact, **not** an enricher bug — it truncates the run,
  so late enrichers show `started` with no `completed`.
