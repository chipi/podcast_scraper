# prod-v2 local validation plan — ADR-108 enrichers + read-time surfaces

**Scope:** verify the ADR-108 work (`topic_consensus` activation, `insight_sentiment`, read-time
`conversation-arc` / `position-arc` with sentiment tinting, `stance_timeline` retirement) end-to-end
on the **real prod-v2 corpus** locally — data layer → API → operator viewer → consumer app.

**Corpus:** `.test_outputs/manual/prod-v2/corpus` (209 raw bridges; ~99 after latest-run-per-feed
dedup; 2 505 insights, 147 persons, 196 topics with ≥2 speakers).

---

## Readiness verdict (as of this pass)

| Layer | State | Note |
|---|---|---|
| **Code** | ✅ ready | #1159 merged; push CI + comprehensive nightly green |
| **prod-v2 data** | ❌ **NOT re-enriched** | corpus still carries the **old** enricher set |

**The blocker:** the prod-v2 corpus was generated *before* ADR-108. Confirmed by inspection:

- `enrichments/` has the **old** `nli_contradiction.json` — **no `topic_consensus.json`**.
- **0** `insight_sentiment` sidecars under `feeds/**` (per-episode sidecars are `insight_density`,
  `topic_cooccurrence` only).

**Consequence:** on the current corpus the new data points **do not surface** — the conversation/
position arcs would render all-neutral (no sentiment join), and there are no consensus emissions.
**Phase 0 (re-enrich) is a prerequisite** before any surface check is meaningful.

Models are cached locally (`cross-encoder/nli-deberta-v3-small`, `all-MiniLM-L6-v2`), so
`topic_consensus` runs on CPU with no download.

---

## Phase 0 — Re-enrich prod-v2 (prerequisite)

Run only the two new enrichers (fast path), against a CPU profile that wires `consensus_local`:

```bash
make enrich \
  CORPUS=.test_outputs/manual/prod-v2/corpus \
  PROFILE=local WITH_ML=1 \
  ONLY=topic_consensus,insight_sentiment
```

- `topic_consensus` = corpus-scope (MiniLM + DeBERTa, CPU) → writes `enrichments/topic_consensus.json`.
- `insight_sentiment` = episode-scope (VADER, deterministic) → writes per-episode
  `*.insight_sentiment.json` sidecars.
- **Cost:** DeBERTa NLI over the corpus's candidate pairs is the heavy part — expect minutes to
  tens of minutes on CPU. VADER is ~instant.
- **Mutation warning:** this writes into the prod-v2 corpus tree. `enrichments/nli_contradiction.json`
  (retired enricher) is left as a stale artifact — optionally `--only` a full set later to clean up,
  or delete it by hand. It is not read by the new surfaces.

**Alternative (full clean re-enrich, heavier):** drop `ONLY=` to regenerate the whole active set
(replaces `nli_contradiction` with `topic_consensus` and refreshes everything).

---

## Phase 1 — Verify enrichment artifacts (data layer)

```bash
C=.test_outputs/manual/prod-v2/corpus
# topic_consensus emitted + shape
python -c "import json; d=json.load(open('$C/enrichments/topic_consensus.json')); \
print('emissions:', len(d.get('data',{}).get('pairs', d.get('pairs',[])))); print(list(d)[:6])"
# insight_sentiment sidecar coverage (should ≈ episode count)
find $C/feeds -name '*.insight_sentiment.json' | wc -l
# label distribution across sidecars
find $C/feeds -name '*.insight_sentiment.json' -exec cat {} \; \
  | python -c "import sys,json; from collections import Counter; c=Counter(); \
[c.update([i['label'] for i in json.loads(l).get('data',{}).get('insights',[])]) for l in sys.stdin]; print(c)"
```

**Pass criteria**
- `topic_consensus.json` exists with ≥1 emission; each carries `consensus_score`, `cosine`,
  `contradiction`.
- `insight_sentiment` sidecar count ≈ episode count; labels span `negative/neutral/positive`
  (not all-neutral).
- Gate: `data/eval/enrichment/topic_consensus/gate_metrics.json` precision 0.91 ≥ 0.5 → admitted.

---

## Phase 2 — Serve corpus + verify API (contract layer)

```bash
# operator API (path= points at the corpus)
APP_OAUTH_PROVIDER=mock APP_SESSION_SECRET=dev \
  .venv/bin/python -m podcast_scraper.cli serve --output-dir "$C" --port 8020 --host 127.0.0.1 &
```

Pick real ids first: `curl -s "http://127.0.0.1:8020/api/corpus/persons/top?path=$C&limit=5"`
and a topic from the graph. Then check each **new** surface returns non-empty, well-formed data:

| Endpoint | Check |
|---|---|
| `GET /api/topics/{id}/conversation-arc?path=$C` | `weeks[]` non-empty; each has `volume`, `negative/neutral/positive`, `avg_compound` |
| `GET /api/topics/{id}/conversation-arc?...&insight_types=claim` | filter narrows vs unfiltered |
| `GET /api/topics/{id}/timeline?path=$C` | insights carry `sentiment:{compound,label}` |
| `GET /api/persons/{id}/positions?topic={t}&path=$C` | insights carry `sentiment` |
| `GET /api/corpus/enrichments/topic_consensus?path=$C` | returns the emissions (consensus signal) |
| `GET /api/app/topics/{id}/conversation-arc` (consumer) | `{topic_id, weeks[]}` non-empty |

**Pass criteria:** every row 200 + non-empty; sentiment fields present (not absent → means the
sidecar join worked); a generic high-volume topic (e.g. an "AI"/"markets" cluster) yields many
weekly buckets (validates the aggregate-first scale design, not a 1000-row dump).

---

## Phase 3 — Verify UI surfaces (both frontends)

Serve the operator viewer (`make serve`, corpus = prod-v2) and the consumer app, then walk:

**Operator viewer (`web/gi-kg-viewer`)**
- Topic entity view → **`TopicConversationArc`**: weekly stacked bars (height = volume, colour =
  sentiment mix); click a week → drill list tinted per insight.
- Person view → **`PositionTrackerPanel`**: position-arc rows sentiment-tinted (rose/slate/emerald).
- `topic_consensus` surfaces wherever enrichment signals render (confirm the exact panel during the
  walk — enrichment health/signals section).

**Consumer app (`web/learning-player`)**
- Topic card (`EntityCardBody`) → **`TopicConversationArc`**: weekly bars + legend; empty-state when
  a topic has no dated insights.

**Capture screenshots** of each surface with real prod-v2 data for the record.

---

## Phase 4 — Data-point coverage matrix

| Data point | Generated by | Surfaces on | Verify |
|---|---|---|---|
| Consensus signal | `topic_consensus` enricher | operator enrichment panel | Phase 1 + `/enrichments/topic_consensus` |
| Per-insight sentiment | `insight_sentiment` sidecars | position-arc / timeline tint | Phase 2 timeline/positions `sentiment` |
| Weekly conversation arc | `topic_conversation_arc` (read-time) | viewer + app `TopicConversationArc` | Phase 2 conversation-arc + Phase 3 |
| Position arc (per person×topic) | `position_arc` (read-time) | viewer `PositionTrackerPanel` | Phase 2 positions + Phase 3 |
| `stance_timeline` **retired** | — | — | confirm **no** ghost artifact/surface |

---

## Phase 5 — Edge / regression checks

- Topic with **no** sentiment sidecar coverage → arc/timeline render **un-tinted, never error**.
- **Generic high-volume topic** → arc shows the *shape* (weekly bars), not a flat 1000-row list.
- **Undated insights** dropped from the arc (no crash).
- **`stance_timeline`**: `grep -r stance_timeline` over the served corpus → no artifact; no UI ghost.
- Re-run `make enrich … ONLY=insight_sentiment` twice → deterministic (identical sidecars).

---

## Sign-off = "ready"

1. Phase 0 completes without enricher errors.
2. Phase 1 artifacts present + non-degenerate (sentiment not all-neutral; consensus ≥1 emission).
3. Phase 2 every endpoint 200 + non-empty + sentiment fields present.
4. Phase 3 both frontends render the arcs/tints on real data (screenshots captured).
5. Phase 5 edges degrade gracefully; no `stance_timeline` ghost.

Until Phase 0 runs, the honest status is **code-ready, data-not-ready** on prod-v2.
