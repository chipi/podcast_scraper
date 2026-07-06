# Consumer Learning Player — Tier-3 real-corpus validation

Real-backend player walk against a real corpus on disk. Mirrors
`web/gi-kg-viewer/e2e/validation/` for the operator viewer.

Runs sequentially (`workers: 1`) with screenshots on every step — the
`validation-results/` artifact IS the value: post-hoc inspection of
every critical surface across a real listening loop.

## What's covered

| Spec                                            | Surface                                                     |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `listen-through-real-corpus.spec.ts`            | Browse → episode → play → capture → verify highlight        |
| `recall-real-corpus.spec.ts`                    | Search with `scope=mine` over heard∪captured                |
| `consolidation-real-corpus.spec.ts`             | Library Revisit resurfacing ladder                          |
| `multi-perspective-topic-real-corpus.spec.ts`   | Topic card multi-speaker synthesis (#1146)                  |
| `offline-shell-real-corpus.spec.ts`             | SW-served shell survives real network drop                  |

## Run locally

```bash
# Terminal 1
make serve-for-validation

# Terminal 2
cd app
APP_CORPUS_PATH=/abs/path/to/your/corpus \
  node_modules/.bin/playwright test --config playwright.validation.config.ts
```

`APP_CORPUS_PATH` is optional. When unset, the specs use the committed
synthetic fixture at `tests/fixtures/app-validation-corpus/v2/` (same
one the fast app-e2e suite uses) — good enough for drift detection,
but the whole point of Tier-3 is to catch things the synthetic corpus
can't. Point it at a real corpus for operator drift walks.

## CI

Scheduled nightly at `nightly-tier3-app-validation` in
`.github/workflows/nightly.yml` (sibling to `nightly-tier3-validation`
for the viewer). Auto-files a deduped `tier3-app-regression` issue on
failure. Screenshots + logs upload as `tier3-app-validation-failures`
artifact (30-day retention).
