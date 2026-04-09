# RFC-066: Run Comparison Tool — Performance Tab

## Status

**Draft — Stub** (split from RFC-064)

## RFC Number

066

## Authors

Podcast Scraper Team

## Date

2026-04-09

## Related RFCs

- `docs/rfc/RFC-064-performance-profiling-release-freeze.md` — Frozen profiles and release freeze framework (provides the data)
- `docs/rfc/RFC-047-run-comparison-visual-tool.md` — Existing Streamlit comparison tool (extended by this RFC)
- `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` — Quality benchmarking framework (quality side of the comparison)

---

## Abstract

This RFC proposes extending the existing RFC-047 Streamlit run comparison tool (`tools/run_compare/app.py`) with a **Performance tab** that displays resource cost metrics from frozen profiles (RFC-064) alongside the existing quality comparison view.

Where RFC-047 answers "did quality change between runs?", this extension adds "did resource cost change?" — joining quality evaluation data (`data/eval/`) and performance profiles (`data/profiles/`) by release tag into a single visual tool.

Split from RFC-064 to keep the release freeze framework focused on its core deliverable (frozen profiles + diff tool) and to avoid mixing Streamlit UI design with profiling infrastructure.

---

## Scope (To Be Designed)

The following capabilities were outlined in the original RFC-064 draft and are the starting point for this RFC's design:

### Join Key

Quality runs in `data/eval/runs/` and performance profiles in `data/profiles/` are joined by release tag. Both artifact types carry the release tag in their metadata. The tool matches them automatically when both exist for the same tag. When only one side exists, the tool shows what it has and marks the other as missing.

### Performance Tab Layout

- **KPI tiles (top row)** — for each selected release: global peak RSS, total wall time, episodes processed
- **Delta table** — same colored delta pattern as RFC-047's quality delta table, applied to resource metrics (peak RSS, wall time, CPU% per stage)
- **Trend chart** — wall time and peak RSS per stage across all frozen releases, plotted as a line chart
- **Combined quality + performance view** — a two-axis scatter where x = quality metric and y = resource cost, one dot per release

### Open Design Questions

1. Which quality metric goes on the x-axis of the combined scatter? (WER? ROUGE? Semantic similarity? User-selectable?)
2. How should the tool handle profiles from different machines? (Filter by hostname? Show warning?)
3. Should the performance tab show per-episode breakdowns or only stage aggregates?
4. How should missing stages (disabled in some releases) be handled in trend charts?

---

## Dependencies

- `streamlit` — already in `[compare]` optional dependencies
- Frozen profile YAML files from RFC-064 (`data/profiles/*.yaml`)
- Quality evaluation data from RFC-041 (`data/eval/runs/`, `data/eval/baselines/`)

---

## Open Questions

1. Should this be a new tab in the existing `tools/run_compare/app.py` or a separate Streamlit app?
2. What is the minimum number of frozen profiles needed before the trend chart is useful?
3. Should the tool support comparing profiles across different datasets (e.g., `indicator_v1` vs `shortwave_v1`)?

---

## References

- **Parent RFC**: `docs/rfc/RFC-064-performance-profiling-release-freeze.md`
- **Extended Tool**: `tools/run_compare/app.py` (RFC-047)
- **Quality Data**: `data/eval/runs/`, `data/eval/baselines/`
- **Performance Data**: `data/profiles/*.yaml`
