# Surface perf traces (Library · Digest · entity-load)

Capture harness for the basic-but-important viewer surfaces **outside** search
and graph — the ones a user hits constantly and that must not regress silently.

- **UI:** `scripts/dev/capture-surface-perf.{sh,mjs}` *(Chunk 3 — pending)*
- **Raw artifacts:** `data/perf/traces/surfaces/`

See [index.md](index.md) for the framework, standard conditions, and split
rationale.

## Scenario catalog *(Chunk 3 — pending)*

| Scenario | Measures | Status |
| --- | --- | --- |
| `library-load` | page load → Library grid first rows visible | Chunk 3 |
| `digest-load` | Digest tab → first digest section painted | Chunk 3 |
| `entity-load` | open a specific entity → NodeDetail rail populated | Chunk 3 |

These are deliberately "boring" surfaces: the point is a stable baseline so a
future change that quietly doubles Library load time is caught at the next
release capture, not by a user.

## Reports

Per-release numbers live under [reports/](reports/index.md).
