# App validation corpus (`v3`) — full-fidelity synthetic corpus

A **committed** corpus that is **schema-current and full-fidelity** — built to be swappable for a
real corpus: fire the app (and the MCP server, and the search/graph capabilities) against it and
everything works, offline, with no pipeline and no ML **to read it**. It began as the consumer
Learning Player e2e fixture and was realigned (RFC-097) so every read surface — player, viewer,
`GET /api/*`, the MCP tools, search, graph — has real data of the current shape.

> **Reading needs nothing; regenerating summaries needs a model.** Until 2026-08-16 this file said
> the corpus was built "with no pipeline and no ML", full stop. That was true of the *build* — and
> is exactly why every committed `summary.raw_text` was the transcript's opening greeting
> ("Welcome back to Singletrack Sessions. Today we're talking about…") rather than a summary, so
> the fixture exercised none of the summarization the product actually runs. Summaries and episode
> durations now come from a **real pipeline run** (Deepgram nova-3 ASR + the LiteLLM gateway for
> summary/GI/KG) and are committed as data. Consumers are unaffected: still no pipeline, no ML, no
> network. See `--pipeline-run` in `scripts/build_app_validation_corpus.py`.

- **What it is:** 9 shows × 4 episodes (`p01`–`p09`), each `ready` with a transcript, GI
  insights, KG topics/people, **diarization diagnostics**, and per-episode + corpus-scope
  enrichments. Sorted keys, stable content-hash episode ids and fixed dates, so a rebuild from the
  same inputs yields the same tree.
- **Summaries:** 35 of 36 are real pipeline output. **`p01_e02` is a known exception** and keeps a
  synthesized stand-in: its transcript was authored in nearly the summarization prompt's own
  style-example wording, so a faithful summary of it is indistinguishable from a copied one and the
  #1386 poison guard correctly drops it. Accepted for v3, tracked for v4 in
  [#1671](https://github.com/chipi/podcast_scraper/issues/1671) and `FIXTURES_SPEC.md` v4 #8.
- **Versioned:** laid out under `v3/`, matching `tests/fixtures/FIXTURES_VERSION` (`v3`).

## Media / audio — NOT in this tree

This corpus carries every artifact **except the audio bytes**. That split is deliberate (audio is
large and shared across fixture consumers), and it is the single most confusing thing about this
directory — so, explicitly:

| | Where | Note |
| --- | --- | --- |
| Episode audio | `../audio/<FIXTURES_VERSION>/<episode_id>.mp3` — currently `../audio/v3/` | One file per episode id, covering **all** episodes in this corpus. Versioned: **check [`../FIXTURES_VERSION`](../FIXTURES_VERSION) first.** |
| How it reaches a client | `make serve-e2e-mock` (loopback `:18765`, serves `/audio/<episode_id>.mp3`) or the [`docker/mock-feeds/`](../../../docker/mock-feeds/README.md) nginx sidecar on the compose network | Both simulate a real podcast host: RSS + episodes + audio. |
| Everything about the fixture trees | [`../README.md`](../README.md) | Its title — *Offline Podcast Fixtures (RSS + Transcripts + Audio)* — is the map. |

**`content.media_url` in this corpus is currently a placeholder data URI that no browser can
decode.** It is not a real enclosure and it is not the audio above; the consumer Playwright suite
works around it with a route stub. Rewiring it to the mock host is
[#1618](https://github.com/chipi/podcast_scraper/issues/1618).

> Do not synthesise audio for this corpus. It already exists in `../audio/v3/`. An agent that
> searched only inside `v3/` on 2026-08-13 concluded otherwise and hand-built an MP3 encoder before
> being stopped — the reason this section exists.

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
