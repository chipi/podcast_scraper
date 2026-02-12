# Architecture visualizations

This directory holds generated diagrams for documentation and architecture understanding.
See [ARCHITECTURE.md](../ARCHITECTURE.md) for where each diagram is referenced.

**Generated artifacts:**

- `dependency-graph.svg` — Module import relationships (pydeps, full package)
- `dependency-graph-simple.svg` — Simplified dependency graph (clustered, max-bacon=2)
- `workflow-call-graph.svg` — Function call graph for `workflow/orchestration.py` (pyan3)
- `orchestration-flow.svg` — Flowchart of orchestration module (code2flow)
- `service-flow.svg` — Flowchart of service API (code2flow)
- `workflow-deps.svg` — Workflow subpackage dependencies (optional)
- `providers-deps.svg` — Providers subpackage dependencies (optional)

**Automated:** The [docs workflow](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/docs.yml) installs [Graphviz](https://graphviz.org/) and runs `make visualize` on every docs build. Updated diagrams are committed back to the repo on push to `main`, so no manual steps are required.

**Local regenerate (optional):** Run `make visualize` (or `make deps-graph`, `make call-graph`, `make flowcharts`) from the project root. Requires Graphviz (`dot` on PATH) and dev dependencies: `pip install -e .[dev]` (includes pydeps, pyan3, code2flow). Graphviz: `brew install graphviz` (macOS), `apt install graphviz` (Debian/Ubuntu).
