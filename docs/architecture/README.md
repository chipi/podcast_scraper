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

**Regenerate locally:** Diagrams are not generated in CI. Run `make visualize` from the project root when you change code that affects architecture (e.g. new modules, workflow changes), then commit the updated `docs/architecture/*.svg` files. Requires [Graphviz](https://graphviz.org/) (`dot` on PATH) and dev dependencies: `pip install -e .[dev]` (pydeps, pyan3, code2flow). Graphviz: `brew install graphviz` (macOS), `apt install graphviz` (Debian/Ubuntu).

**Enforcement:** `make ci` and `make ci-fast` run `make check-visualizations`; CI (python-app and docs workflows) also runs it. If diagrams are older than the source files they depend on, the check fails. Run `make visualize` and commit the updated SVGs to fix it.
