# App validation corpus (consumer "Learning Player")

A **committed, deterministically-synthesized** validation corpus for the consumer
Learning Player app. It is the consumer-app analogue of the viewer's
`tests/fixtures/viewer-validation-corpus/` (built by
`scripts/build_synthetic_validation_corpus.py`): a small, schema-shaped corpus
constructed from the checked-in text fixtures with **no pipeline and no ML**, so
the app's Playwright e2e runs against stable, version-pinned content the real
consumer API can serve directly.

- **What it is:** `feeds/<show>/run_*/metadata/<ep>.{metadata,gi,kg}.json` +
  `feeds/<show>/run_*/transcripts/<ep>.{txt,segments.json}` for ~3 shows × 2
  episodes, plus `search/topic_clusters.json` (for the Profile interests picker +
  discovery). Every episode is `ready` (carries a transcript) and carries GI
  insights + KG topics/people, so the Player insights, entity cards, follow
  surfaces, and interests picker all have real data.
- **Committed + deterministic:** checked into git (NOT gitignored). The generator
  uses sorted JSON keys, stable content hashes for episode ids, and fixed publish
  dates, so re-running it produces a byte-identical tree (idempotent).
- **Versioned:** laid out under `v2/`, matching `tests/fixtures/FIXTURES_VERSION`.

## Shows / episodes

| Show (feed dir) | Display title | Episodes |
| --- | --- | --- |
| `p05` | Long Horizon Notes | Index Investing Without the Myths · Real Estate: Numbers Before Narratives |
| `p02` | Practical Systems | On-Call That Doesn't Break People · Staff-Engineer Communication Patterns |
| `p03` | Below the Surface | Wreck Diving Fundamentals · Marine Biology for Divers |

"What's new" (recency) order is stable: **Index Investing Without the Myths**
(Long Horizon Notes) is newest.

## Regenerate

```sh
.venv/bin/python scripts/build_app_validation_corpus.py
```

Defaults: `--rss-dir tests/fixtures/rss`,
`--transcripts-dir tests/fixtures/transcripts/<FIXTURES_VERSION>`,
`--output tests/fixtures/app-validation-corpus`, `--max-feeds 3`,
`--max-episodes-per-feed 2`. The generator reuses the viewer generator's
construction helpers (`build_gi`, `build_kg`, `parse_diarized_segments`,
`format_screenplay_with_offsets`, `parse_rss_feed_metadata`, `slug`) so the
GI/KG artifacts can't drift from what the backend readers expect.

## Schema

The GI/KG artifacts use the **same loose synthetic shape as the viewer
validation corpus** (`schema_version: "2"`), which the consumer readers
(`app_gi_view`, `app_kg_view`, `segments_view`) parse defensively. They are NOT
intended to pass the strict GIL/KG JSON-schema validators
(`make validate-gi-schema` / `validate-kg-schema`) — exactly like the viewer
corpus, which those strict targets also do not cover. The contract this corpus
validates is the **consumer API reader contract** (exercised end-to-end by the
app's Playwright e2e), not the strict pipeline-output schema.

## Used by

`web/learning-player/playwright.config.ts` boots the real consumer API
(`podcast_scraper.cli serve --output-dir tests/fixtures/app-validation-corpus/v3`)
over this corpus — no build step. Per-user runtime state (queue/profile/
interests the API writes) is redirected to a gitignored `web/learning-player/e2e/.app-state/` via
`APP_DATA_DIR`, so this committed tree is never mutated by a test run.
