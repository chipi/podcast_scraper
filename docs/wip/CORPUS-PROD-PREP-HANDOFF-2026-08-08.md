# Handoff — prep the diarized DeepSeek corpus for prod (task #14)

**For:** the agent in the `podcast_scraper-ai-ml-improvements` worktree (the corpus
lives there — prep it **in place / in that workspace**, no cross-worktree copy).
**From:** the `-FUTURE` session (2026-08-08). **Issue context:** #1494, #1497.

## Goal

Prod's live corpus (May-05) has **zero diarization** → 6 MCP speaker tools return
empty. Fix by shipping the already-diarized DeepSeek corpus. **No Whisper / DGX /
GI-rebuild** — only the deterministic prep below, then export → gated deploy.

## ⚠️ Content-replacement caveat (operator must confirm at deploy time)

This corpus is a **newer, different 105-episode set** (08-05 run), **not** the same
episodes as prod. Deploying it **replaces** prod's current 100 episodes (the
Daily / Hard Fork / Sam-Altman content tested via MCP this session). Prior-session
memory records this as the deliberate plan — but the **deploy step is gated on the
operator explicitly re-confirming the full-replace intent.**

## Source

```
.test_outputs/manual/v2.5-deepseek-fixed-100ep     # in ai-ml-improvements worktree
```
- `produced_by`: code_version **2.7.0.dev0**, git_sha 1768f1d, 2026-08-05 →
  `make corpus-compat-check CORPUS_DIR=<dir>` returns **COMPAT OK** (verified).
- 9 feeds, **2 runs each** (18 run dirs). Newer run per feed is
  `run_20260805-*_7a69fc41`; older is `run_2026072[89]-*_ba02775e`.
- GI files are per-episode: `<run>/metadata/*.gi.json` (210 total = 2 runs × 105).
- Two-tier lance index present, `schema_version: 3` (current). `search/metadata.json`
  has **13,261 docs** — provenance (105 vs 210 eps) **unverified** → reindex below.

## Deterministic prep chain (all local, no LLM/GPU)

Work on a copy inside that workspace if you want a clean rollback point; the source
is otherwise mutated in place. `CORPUS=` is the corpus root.

1. **Compat check** — `make corpus-compat-check CORPUS_DIR=$CORPUS` → expect `COMPAT OK`.
2. **Dedup to newest run per feed** — keep each feed's `run_20260805-*_7a69fc41`,
   delete the older `run_2026072[89]-*_ba02775e`. Result: **105 eps** (12+10+12+12+12+12+12+11+12).
3. **Reindex** (safety — index doc-count provenance unverified after dedup):
   `make index-two-tier CORPUS_DIR=$CORPUS`.
3b. **Topic clusters** (fixes `topic_clusters.json` 404 — the #14-cutover red smoke; the pipeline/
   prep never generated it): `make topic-clusters CORPUS_DIR=$CORPUS` (THRESHOLD defaults to 0.75 =
   cloud_balanced's `topic_cluster_threshold`; **not** the 0.35 small-fixture override). Must run
   AFTER `index-two-tier` (reads `search/lance_index/`); query-time-read → no container recreate.
4. **Relational edges** (fixes `show_episodes` — adds `HAS_EPISODE`; idempotent):
   `make enrich-relational-edges CORPUS_DIR=$CORPUS`.
5. **Enrichment** (fixes `enrichment_signals` + `consensus_search`):
   `make enrich CORPUS=$CORPUS --with-ml ONLY="insight_density,insight_sentiment,topic_cooccurrence_corpus,topic_consensus"`
   - `topic_consensus` (#15) is a CPU-local NLI enricher (consensus_local = MiniLM +
     DeBERTa; **no LLM, no GPU/DGX**) — it needs `--with-ml` + the `[ml]` extras
     (sentence-transformers + a local DeBERTa NLI checkpoint). Without it,
     `consensus_search` returns EMPTY (the enricher works — ADR-108, precision 0.91 —
     the corpus just never had it run). Verify names via
     `python -m podcast_scraper.cli enrich --help` / the enricher registry.
   - If the `[ml]` extras / NLI checkpoint aren't available in the prep env, run the
     first three deterministic enrichers WITHOUT `--with-ml`, and run `topic_consensus`
     separately where the extras exist (still CPU-only).
6. **Regen `feeds.spec.yaml`** at the corpus root (required by `export-corpus`;
   memory flagged it missing).
7. **Export tarball** —
   `make export-corpus CORPUS_DIR=$CORPUS OUT=/tmp/corpus-deepseek-105.tgz LAYOUT=prod`.

## GATED — do NOT run without operator approval

8. Transfer tarball → prod → `make restore-corpus-prod` → **redeploy via
   `deploy-player.yml`**. **NEVER `docker restart player-api`** — it drops the
   tmpfs `/dev/shm/player-secrets` and crashes the api (happened this session);
   always redeploy so secrets are re-staged.

---

# Acceptance criteria — "what we want to see when it's done"

Three gates. **A** is the prepping agent's to prove (deterministic, pre-deploy).
**B + C** are the -FUTURE session's to prove (functional + o11y, post-deploy).
Each item is pass/fail with the check that produces the evidence.

## Gate A — corpus is well-formed (pre-deploy, deterministic)

Run against the exported corpus root; paste the command output as evidence.

| # | Criterion | Check / expected |
|---|---|---|
| A1 | Exactly **105 distinct episodes**, one run per feed | count `metadata/*.gi.json` = 105; no `ba02775e` run dirs remain |
| A2 | Code/corpus compat | `make corpus-compat-check` → `COMPAT OK` (2.7.0.dev0) |
| A3 | **Diarization intact** | ≥80% of quotes carry `speaker_id`; `SPOKEN_BY` edges > 0 (memory: 4460) |
| A4 | **`HAS_EPISODE` edges > 0** (was 0 — the `show_episodes` fix) | grep gi.json edges |
| A5 | **`enrichments/` populated** per run (the `enrichment_signals` fix) | `insight_density`, `insight_sentiment`, `topic_cooccurrence_corpus` sidecars present |
| A6 | Lance index consistent with 105 eps | doc-count sane; **no duplicate-run entries** (dedup + reindex worked) |
| A7 | `feeds.spec.yaml` present at root | required by export; export succeeded |
| A8 | Tarball restores cleanly in a **local rehearsal** | `make restore-corpus` into a throwaway `WORKSPACE_DIR`, then A1–A6 pass on the restored copy |
| **A9** | **Completeness gate PASSES (#16)** — one command covering A4/A5/A6 + index staleness + topic_clusters (A2 code/corpus compat stays `corpus-compat-check`'s job) | `make corpus-completeness-check CORPUS_DIR=$CORPUS` → `VERDICT: PASS` (exit 0). Fails on a stale/absent index or missing HAS_EPISODE / typed MENTIONS / enrichments / **`search/topic_clusters.json`** (present + non-empty); diarization is a soft warn. Run this as the final pre-export check — it would have caught the #14 topic_clusters 404 before deploy. |

## Gate B — MCP tools return real data (post-deploy, via claude.ai)

The whole point: the 6 speaker tools that were empty must now answer. Test each
against a **known diarized episode** in the new set.

| # | Tool | Was (May-05 prod) | Want |
|---|---|---|---|
| B1 | `episode_speaker_roster` | empty | named speakers per episode |
| B2 | `who_said` | empty | quote → correct speaker |
| B3 | `person_positions` | empty | a person's stances across eps |
| B4 | `person_topics` | empty | topics a person speaks on |
| B5 | `position_arc` | empty | a stance's evolution over time |
| B6 | `topic_perspective_leaders` | empty | who leads a topic's discourse |
| B7 | `show_episodes` | empty (HAS_EPISODE=0) | episodes listed per show |
| B8 | `corpus_enrichment_signals` | empty | density/sentiment/co-occurrence populated |
| B9 | `search_corpus` + `resolve_entity` | worked | still work on the new corpus |

**Metadata-first rule (memory):** WHO comes from feed/episode metadata;
diarization only says WHICH VOICE. Spot-check 2–3 speaker attributions against the
transcript to confirm they're not mislabeled.

## Gate C — observability sees it (post-deploy, the #1505 payoff)

The o11y we just shipped must light up under this real traffic.

| # | Criterion | Where |
|---|---|---|
| C1 | Per-tool **spans** `mcp.tool.<name>` with `mcp.user_id`, nested under `POST /mcp` | VictoriaTraces / Grafana |
| C2 | **Metrics** `mcp_tool_calls_total{tool,ok}` + `mcp_tool_duration_seconds` scraped | Grafana / Prometheus |
| C3 | **Structured logs** (tool/user/duration/ok) with trace_id correlation | Grafana logs (surface=mcp) |
| C4 | **GlitchTip** clean — no new MCP errors during the validation run | GlitchTip (component=mcp) |
| C5 | **Umami** records the tool-call events | analytics.<domain> |

## Known-bug regression checks (does the new corpus change them?)

Re-run the claude.ai findings against the new corpus and record whether they
persist — feeds #19/#20/#21. **Not deploy-blocking**, but we want the data:

- **#19 grounded_only** — `search_corpus(grounded_only=true)` should now keep
  grounded insights (was dropping ALL insights). If the reindex carried the
  `grounded` flag onto insight rows, this may already be fixed.
- **#20 long-episode truncation** — `episode_digest` on the longest multi-segment
  eps: is `insight_density` still early-skewed with zero late? (GI-side; DeepSeek
  GI may differ from prod's.)
- **#21 briefing_pack** — `corpus_briefing_pack` on a broad query: `show_ids`
  consistent with `episode_count`, confidence non-zero.

## Explicitly OUT of scope (not acceptance criteria here)

- Re-transcription / re-diarization / GI-rebuild (DGX down; DeepSeek GI code
  uncommitted — this corpus already IS that output).
- Fixing #19/#20/#21 in code (separate tasks; only *measured* here).
- Scalable/incremental indexing (#1494) — deferred.
- Audio hosting — audio is bridge-only, never rehosted (by design).
