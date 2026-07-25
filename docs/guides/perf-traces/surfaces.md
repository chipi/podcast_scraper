# Surface perf traces (Library · Digest · entity-load)

Capture harness for the basic-but-important viewer surfaces **outside** search
and graph — the ones a user hits constantly and that must not regress silently.

- **UI:** `scripts/dev/capture-surface-perf.{sh,mjs}`
- **Raw artifacts:** `data/perf/traces/surfaces/`

See [index.md](index.md) for the framework, standard conditions, and split
rationale.

## Run it

```bash
scripts/dev/capture-surface-perf.sh \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --label <release>-surfaces \
  --output-dir data/perf/traces/surfaces
```

## Scenario catalog

| Scenario | Measures | Status |
| --- | --- | --- |
| `library-load` | Library tab click → `library-root` visible | Implemented |
| `digest-load` | Digest tab click → `digest-root` visible | Implemented |
| `entity-load` | Library → first episode row click → `episode-detail-rail` visible | Implemented |

**What the numbers mean:** the corpus envelope is fetched on landing, so these
measure **tab-switch / interaction → container-visible** responsiveness (not a
cold data fetch). That's the right signal for these surfaces — the regression we
care about is "a change made switching to Library / opening an episode janky."
Every scenario degrades gracefully (records an `error` field, never aborts).

These are deliberately "boring" surfaces: the point is a stable baseline so a
future change that quietly doubles Library responsiveness is caught at the next
release capture, not by a user. Latest run:
`data/perf/traces/surfaces/2.7.0.dev1-surfaces.ui.metrics.json`.

## Reports

Per-release numbers live under [reports/](reports/index.md).
