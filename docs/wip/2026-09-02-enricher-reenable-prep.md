# Re-enabling the three deterministic corpus enrichers — prepared runbook

**Date:** 2026-09-02 · **Related:** `#1921` · **Corpus:** prod `/app/output`

Prepared while the Batch A ingestion runs, so the sequence is executable the moment it lands.
Every command below has been dry-run locally against the live config; the payload is validated
against the same two schema checks `PUT /api/enrichment/config` runs server-side.

---

## What is on and off today

Nine enrichers exist. **Four enabled, five disabled** — every disable lives in the *operator*
block of the corpus's `viewer_operator.yaml`, not in the profile (`cloud_balanced` lists all nine).

| Enricher | Scope | ML provider | Operator block | Effective | Last run |
| --- | --- | --- | --- | --- | --- |
| `insight_density` | episode | — | `{}` | enabled | today |
| `insight_sentiment` | episode | — | `{}` | enabled | today |
| `guest_coappearance` | corpus | — | `expected_duration_s: 300` | enabled | today |
| `temporal_velocity` | corpus | — | `window_months: 24, weekly_window: 104, expected_duration_s: 300` | enabled | today |
| `grounding_rate` | corpus | — | `enabled: false` | **off** | 9d |
| `topic_cooccurrence_corpus` | corpus | — | `enabled: false` | **off** | 9d |
| `topic_theme_clusters` | corpus | — | `enabled: false` | **off** | 9d |
| `topic_similarity` | corpus | **`sentence_transformer_local`** | `enabled: false` | **off** | 8d |
| `topic_consensus` | corpus | **`consensus_local`** | `enabled: false` | **off** | 8d, `timeout` |

**Operator decision:** the two ML ones stay off deliberately. The other three should be on — this
runbook turns those three on and nothing else.

---

## A hypothesis for *why* they were disabled, worth testing on the rerun

The three carry a **30-second** hard cap from their manifests:

```
grounding_rate             expected_duration_s=30
topic_cooccurrence_corpus  expected_duration_s=30
topic_theme_clusters       expected_duration_s=30
```

`expected_duration_s` is both the heartbeat-stall threshold and the hard `wait_for` cap
(`enrichment/executor.py`), and the corpus-scope default when unset is 600s — so these three opt
*down* to 30s.

The two corpus-scope enrichers that **are** enabled both carry an operator override to **300s**.
That pattern is the tell: someone hit the 30s cap on corpus-scope work and raised it for those
two. The three under discussion were switched off instead.

Their health says `last_status: ok` at ~765 episodes nine days ago, so 30s sufficed then. The
corpus is now **1,006+ and climbing** (+31%). Enabling them at the bare 30s cap would likely
reproduce whatever was originally hit.

**So this runbook enables them WITH `expected_duration_s: 300`**, matching the two already-enabled
corpus-scope enrichers. If they still time out, that is a real finding about cost growth rather
than a repeat of a known-bad configuration.

---

## Sequence

### 0. Preconditions

- All ten Batch A pipeline jobs `succeeded`.
- Enrichment job `5cc321e5` (queued behind them) has **run to completion** on the current
  four-enricher config. That run is the **baseline** — do not cancel it.

```bash
B=https://prod-podcast.tail6d0ed4.ts.net
KEY=$(tr -d ' \n\r' < ~/podcast_operator_api_key.txt)
curl -fsS -H "X-Operator-Key: $KEY" "$B/api/jobs?path=/app/output" \
  | jq -r '.jobs[] | select(.created_at >= "2026-09-01T22:41") | "\(.job_id[0:8]) \(.status)"'
# expect: 10 feeds succeeded + 5cc321e5 succeeded, nothing running or queued
```

### 1. Snapshot the current config — this is the rollback

```bash
curl -fsS -H "X-Operator-Key: $KEY" "$B/api/enrichment/config?path=/app/output" \
  > /tmp/enrich-config-before.json
jq -r '.operator_block.enrichers | to_entries[] | "\(.key)\t\(.value|tostring)"' \
  /tmp/enrich-config-before.json
```

Keep that file until the analysis is done. Rollback is `PUT` with
`{"enrichment_block": <.operator_block from this file>}`.

### 2. Build the payload

**`PUT /api/enrichment/config` REPLACES the whole operator `enrichment:` block** — same
whole-object-replace hazard as `PUT /api/feeds`. Derive it from the live config, never hand-write
it, or the tuning knobs on `guest_coappearance` / `temporal_velocity` are silently dropped.

```bash
jq '{enrichment_block: (.operator_block
      | .enrichers.grounding_rate            = {expected_duration_s: 300}
      | .enrichers.topic_cooccurrence_corpus = {expected_duration_s: 300}
      | .enrichers.topic_theme_clusters      = {expected_duration_s: 300})}' \
  /tmp/enrich-config-before.json > /tmp/enrich-enable.json
```

Check before sending — all five must hold:

```bash
jq -e '.enrichment_block.enrichers
   | (.grounding_rate.enabled            == null)          # on
   and (.topic_cooccurrence_corpus.enabled == null)         # on
   and (.topic_theme_clusters.enabled    == null)           # on
   and (.topic_similarity.enabled        == false)          # ML, stays off
   and (.topic_consensus.enabled         == false)          # ML, stays off
   and (.guest_coappearance.expected_duration_s == 300)     # knob preserved
   and (.temporal_velocity.window_months == 24)             # knob preserved
   and (length == 9)' /tmp/enrich-enable.json && echo "payload OK"
```

### 3. Apply

```bash
curl -fsS -X PUT -H "X-Operator-Key: $KEY" -H 'Content-Type: application/json' \
  --data @/tmp/enrich-enable.json \
  "$B/api/enrichment/config?path=/app/output" \
  | jq -r '.resolved_block.enrichers | to_entries[]
           | "\(.key)\t\(if .value.enabled == false then "DISABLED" else "enabled" end)"'
# expect exactly two DISABLED: topic_similarity, topic_consensus
```

The route validates against base + composed schema and 400s on a bad block, so a rejected PUT
leaves the config untouched.

### 4. Run full enrichment

```bash
curl -fsS -X POST -H "X-Operator-Key: $KEY" --get \
  --data-urlencode "path=/app/output" "$B/api/jobs/enrichment" | jq
```

---

## What to measure, and what would answer the question

The point is not "did it run" — it is **whether there was a real reason these were off.**

| Check | Source | Answers |
| --- | --- | --- |
| Per-enricher wall time vs the 300s cap | `GET /api/enrichment/events`, `/run-summary` | Was **cost** the reason? A 30s manifest cap comfortably exceeded at 1,006 episodes explains the original disable and means the manifests need raising, not the enrichers avoiding. |
| Completion status | `GET /api/enrichment/health` → `last_status` | Did any hit the raised cap anyway (i.e. superlinear, not just slow)? |
| `grounding_rate` distribution across people | `enrichments/grounding_rate.json` | Does it discriminate, or is it flat and therefore useless as a signal? |
| `topic_cooccurrence_corpus` pair count; pairs clearing a `lift`/`pmi` bar | `enrichments/topic_cooccurrence_corpus.json` | Is the corpus finally big enough for association strength to beat raw frequency? |
| `topic_theme_clusters` cluster count + size distribution | `enrichments/topic_theme_clusters.json` | Are themes coherent at this scale, or one giant blob? |
| **KG topology after co-occurrence exists** | compare against `#1918` | `#1918` says the per-episode KG is a star with zero entity-entity edges; PRD-043 derives entity relations from co-occurrence **at view time**. With `topic_cooccurrence_corpus` populated, does the graph surface differently? This may reframe `#1918`. |

Then **re-measure §5i / §5j** (`docs/wip/ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md`): the ten DEEPEN
grades were taken at 20 episodes/feed with these three off. Both inputs have changed.

---

## Rollback

```bash
jq '{enrichment_block: .operator_block}' /tmp/enrich-config-before.json \
  | curl -fsS -X PUT -H "X-Operator-Key: $KEY" -H 'Content-Type: application/json' \
      --data @- "$B/api/enrichment/config?path=/app/output" | jq -r '.resolved_block.enrichers | keys'
```

---

## Verified while preparing this

- The payload transformation was run against the **live** operator block; all nine enrichers
  survive, both ML enrichers stay `enabled: false`, and the `expected_duration_s` /
  `window_months` / `weekly_window` knobs on the enabled four are preserved.
- The resulting block passes **`validate_enrichment_block`** and
  **`_validate_against_composed_schema`** locally — the same two checks the PUT route runs.

## NOT verified

- **That the three actually complete at 1,006+ episodes**, at 300s or at all. The 30s-cap
  hypothesis above is a hypothesis; the rerun is what tests it.
- **Why they were originally disabled.** No commit or note records it — the operator block is
  corpus state, not version control. The 30s pattern is circumstantial.
- **The PUT itself has not been executed.** Payload built and schema-validated locally; the route
  contract is read from `server/routes/enrichment_config.py`.
- **Cost/load of the rerun** on a box that has just finished a 100-episode ingestion. All three
  are deterministic (no ML provider, no GPU), but none has run at this corpus size.
