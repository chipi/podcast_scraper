# Synthetic corpus → full-fidelity, swappable-for-real (DONE)

**Goal:** make `tests/fixtures/app-validation-corpus/v3` a complete, real-shaped corpus you
can swap for a real one — app + MCP + search all work against it identically.

**Branch:** `fix/synthetic-corpus-full-fidelity` (off main). **Status: DONE** (pending a
final `ci-fast` + operator push approval).

## What was done (3 commits)

1. **`336f6d64`** — repaired the crashing builder (`build_kg()` signature drift).
2. **`a974b647`** — diarization: `.speakers.diagnostics.json` per episode (talk-share,
   host/guest roster, voice census, unattributed share) — what MCP `episode_speaker_roster`
   reads.
3. **`b4d124d4`** — **schema realignment to current** (the big one):
   - **GI → v3.1**: `schema_version` "3.1" (was int `2`); Insight props gain `episode_id` +
     `insight_type` (claim/observation/recommendation) + `position_hint`; Episode matches the
     v3 additionalProperties:false shape (dropped the non-schema `episode_id`/
     `metadata_relative_path` hack; added `feed_id`); insight `ABOUT` topic edges.
   - **KG → v2.0**: `schema_version` "2.0" (was 2.1); Episode `{podcast_id,title,publish_date}`;
     Topic gains `slug`; edge `RELATES_TO`→`MENTIONS`; extraction `model_version` `provider:…`.
   - Authored-claim nodes (`_inject_authored_claims`) brought to v3 (Quote required fields).
   - Super-themes: `topic_theme_clusters` v1.1.0 `super_theme_id/label` rollup + method/count.

## Verified

- **GI + KG schema: all 36 pass** (`validate-gi-schema` / `validate-kg-schema`).
- **Invariants: 14/14** (`test_app_validation_corpus_invariants.py`), incl. super-theme rollup.
- **Catalog loads 36 ready rows**; **corpus graph relational queries** (`who_said`,
  `entities_in_topic`) return rich results — the app + MCP read paths work.
- `ci-fast` = the final gate (run with `env -u NODE_OPTIONS` — the markdownlint step trips on a
  stale cmux `NODE_OPTIONS` preload, environmental).

## Rebuild recipe (reproducible)

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/build_app_validation_corpus.py   # deterministic
HF_HUB_OFFLINE=1 make enrich CORPUS=tests/fixtures/app-validation-corpus/v3 # corpus enrichers
# schema + invariants: validate-gi-schema / validate-kg-schema / the invariants test
```

The two-tier search index (`search/lance_index/`) is built at test setup, not committed.

## Not done / open

- **Push not done** (gated on operator approval). Branch is committed + green, ready.
- `ci-fast` result pending at handoff (kicked off; check the run).
- The **MCP arc** (`feat/mcp-core-refresh`, 11 commits) is separate + done; its
  `episode_speaker_roster` is the consumer of this arc's diagnostics — cross-validate once
  both land on main.
