# App validation corpus (`v3`) — full-fidelity synthetic corpus

A **committed, deterministically-synthesized** corpus that is **schema-current and
full-fidelity** — built to be swappable for a real corpus: fire the app (and the MCP server,
and the search/graph capabilities) against it and everything works, offline, with no pipeline
and no ML. It began as the consumer Learning Player e2e fixture and was realigned (RFC-097) so
every read surface — player, viewer, `GET /api/*`, the MCP tools, search, graph — has real
data of the current shape.

- **What it is:** 9 shows × 4 episodes (`p01`–`p09`), each `ready` with a transcript, GI
  insights, KG topics/people, **diarization diagnostics**, and per-episode + corpus-scope
  enrichments. Constructed from checked-in text fixtures with sorted keys, stable content-hash
  episode ids, and fixed dates → re-running yields a byte-identical tree.
- **Versioned:** laid out under `v3/`, matching `tests/fixtures/FIXTURES_VERSION` (`v3`).

## Schema — current, not loose (changed in the realignment)

Unlike the older loose synthetic shape, `v3` emits the **current** artifact schemas:

| Artifact | `schema_version` | Notes |
| --- | --- | --- |
| GI (`*.gi.json`) | `"3.1"` | Insight `{text, episode_id, grounded, insight_type ∈ (claim/observation/recommendation), position_hint}`, Topic `{label}`, MENTIONS + ABOUT edges, Quote provenance (`char_start/char_end/timestamps/transcript_ref`). |
| KG (`*.kg.json`) | `"2.0"` | Episode `{podcast_id, title, publish_date}`, Topic `{label, slug}`, MENTIONS edges, `extraction.model_version = "provider:synthetic-validation-corpus-v1"`. |

## Layout

```text
v3/
  feeds/<show>/run_*/metadata/<ep>.{metadata,gi,kg}.json
  feeds/<show>/run_*/metadata/<ep>.speakers.diagnostics.json   # diarization: talk-share, roster
  feeds/<show>/run_*/metadata/enrichments/<ep>.insight_{density,sentiment}.json
  feeds/<show>/run_*/transcripts/<ep>.{txt,segments.json}
  enrichments/                    # corpus-scope (RFC-088 envelopes)
    topic_similarity.json  topic_consensus.json  topic_theme_clusters.json
    topic_cooccurrence_corpus.json  temporal_velocity.json  grounding_rate.json
    guest_coappearance.json  run_summary.json
  search/
    topic_clusters.json           # committed (Profile interests picker + cluster ops)
    metadata.json                 # committed index metadata
    lance_index/                  # NOT committed — built at test setup (offline MiniLM)
  .viewer/enrichment_{health,status}.json
```

### Diarization diagnostics (`*.speakers.diagnostics.json`)

Every episode carries one — `summary.num_speakers`, `summary.exposed[]` (per-speaker
talk-share fraction + host/guest role), `voice_census`, `unattributed_talk_share`. This is
what the MCP `episode_speaker_roster` / `episode_digest` tools read (there is no HTTP route for
it), and it's the gap that drove the realignment.

## The search index is built, not committed

`search/lance_index/` is **not** in git — no binary lance blob, no lance-format-version
coupling. The test layers below build it at module setup via
`podcast_scraper.cli index-two-tier --output-dir <corpus>` (offline, cached MiniLM), and skip
cleanly if the embedding model isn't available (model-less unit CI). It runs fully in the ML
tier / locally.

## Regenerate

```sh
.venv/bin/python scripts/build_app_validation_corpus.py   # metadata + GI/KG + diarization
make enrich CORPUS=tests/fixtures/app-validation-corpus/v3 # corpus-scope + per-episode enrichers
```

The generator reuses the viewer generator's construction helpers
(`scripts/build_synthetic_validation_corpus.py`: `build_gi`, `build_kg`,
`parse_diarized_segments`, …) so GI/KG can't drift from what the readers expect. Full recipe:
`docs/wip/SYNTHETIC-CORPUS-FULL-FIDELITY-PLAN.md`.

## Used by — the tier-3 layers that rely on this corpus (for other agents)

This corpus is the shared fixture for four tier-3 layers. When you change the corpus shape,
these are what re-validate it; when you add a read surface, add a case here so it's covered.

| Layer | File | What it proves |
| --- | --- | --- |
| **Corpus invariants** | `tests/integration/test_app_validation_corpus_invariants.py` | Every enrichment artifact is present + non-degenerate (similarity has neighbours, consensus has cross-person pairs, theme clusters + super-themes form, velocity/co-occurrence/grounding-rate discriminate, sentiment has spread, a topic spans ≥3 shows). Static JSON — no index needed. |
| **Search capability** | `tests/integration/search/test_search_capability_against_fixture.py` | `structured_corpus_search` SSOT: two-tier stamping, tier filters, query classification, lift-stats shape, grounded/topic/speaker filters, + the `cluster_hits` / `consensus_pairs_for_hits` operators and two-subject `compare_subjects`. Builds the index at setup. |
| **MCP pivot chain** | `tests/integration/test_mcp_pivot_chain_e2e.py` | The RFC-095 cross-surface chain: ids flow search → insight → graph → compare; episode-scoped tools (speaker roster, per-episode insights/enrichment, digest) return data. Builds the index at setup. |
| **Player e2e** | `web/learning-player/playwright.config.ts` | Boots the real consumer API (`serve --output-dir …/v3`) over this corpus — no build step. Per-user runtime state is redirected to a gitignored `web/learning-player/e2e/.app-state/` via `APP_DATA_DIR`, so this committed tree is never mutated. |

Run the two index-building Python layers:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/integration/search/test_search_capability_against_fixture.py \
  tests/integration/test_mcp_pivot_chain_e2e.py -q
```
