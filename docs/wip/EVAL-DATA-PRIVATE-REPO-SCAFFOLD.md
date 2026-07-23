# Private eval-data repo — created 2026-07-23

**Repo:** `github.com/chipi/podcast-scraper-eval-data` (PRIVATE, standalone).
Home for eval datasets built from real / public / licensed data that must not live in the public
`podcast_scraper` code repo. Public repo stays code + tiny synthetic fixtures.

## Design decisions (operator-approved)

1. **GitHub private repo** `podcast-scraper-eval-data` — created + pushed.
2. **Standalone** (not a submodule). Main repo points at it via `EVAL_DATA_DIR` (or a gitignored
   symlink `data/eval-private` → the checkout).
3. **Pure re-fetch** for v1 — no object-store cache. Blobs are re-fetched from source into a
   gitignored `cache/` on demand.
4. **SPGISpeech** accepted (Kensho research/non-commercial) — referenced by HF download recipe.

## It MIRRORS main's `data/eval/` — no deviations

The repo is a **faithful mirror** of `podcast_scraper/data/eval/` so the existing `scripts/eval/`
tooling runs against it unchanged. Same top-level tree, same invariants, **schemas copied verbatim**
(the data contracts the harness validates against — re-sync when main's schemas change):

```text
data/eval/
├── README.md                # copied verbatim from main
├── sources/                 # immutable raw inputs (transcripts, RSS XML, metadata, index.json)
├── datasets/                # dataset definitions (episode selection / canonical lists)
├── materialized/            # derived, regenerable, byte-reproducible run inputs
├── configs/                 # experiment YAML
├── baselines/               # frozen reference runs
├── references/              # frozen silver/gold quality targets
├── runs/                    # ad-hoc experiment outputs
└── schemas/                 # JSON schemas — copied verbatim from main
```

Per-dir READMEs + templates (`MATERIALIZED_DATASET_README_TEMPLATE.md`, baseline/reference
templates) copied so the process docs match main exactly.

## Commit policy

- **Committed** (the point of a private repo): datasets, materialized inputs, references/baselines,
  results — including real transcript text (repo is private + access-controlled).
- **NEVER committed — audio.** Bridge-only: re-fetch from source into gitignored `cache/`, never
  rehost. (Main's `data/eval` commits no audio either — verified.)
- **Third-party datasets** (SPGISpeech/MediaSum): referenced by HF repo + revision, not
  redistributed. Store the recipe + our derived metrics.

`.gitignore` blocks `*.mp3/wav/m4a/...`, `cache/`, `.env`.

## Next (when operator says go — nothing processed yet)

1. Add the ASR ground-truth dataset as a `datasets/asr_human_gt_v1.json` following main's dataset
   schema, sourced per the research shortlist (80k Hours native-RSS + Dwarkesh/Lex web + SPGISpeech).
2. Materialize it (`materialized/asr_human_gt_v1/`) with the same `episode_metadata.schema.json`.
3. Run the 5-model bake-off via the existing `scripts/eval/` harness → `runs/` + a corrected report.
4. Seed `runs/` with the corrected 2026-07 bake-off, relabelled honestly (old-ASR reference, NOT
   human ground truth).

## Related

- Research shortlist of human-transcript feeds: (this session) — 80k Hours (native RSS, human),
  Dwarkesh/Lex (web, human), SPGISpeech/MediaSum (datasets). Odd Lots/Masters-in-Business/Practical
  AI/The Gradient = auto (rejected).
