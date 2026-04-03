# Architecture visualizations

This directory holds generated diagrams for documentation and architecture understanding.
See [Architecture](../ARCHITECTURE.md) for where each diagram is referenced.

**Generated artifacts:**

- `dependency-graph.svg` — Module import relationships (pydeps, full package)
- `dependency-graph-simple.svg` — Simplified dependency graph (clustered, max-bacon=2)
- `workflow-call-graph.svg` — Function call graph for `workflow/orchestration.py` (pyan3)
- `orchestration-flow.svg` — Flowchart of orchestration module (code2flow)
- `service-flow.svg` — Flowchart of service API (code2flow)
- `providers-deps.svg` — Providers subpackage dependency graph (pydeps)
- `gi-pipeline-flow.svg` — GIL extraction pipeline flowchart (Graphviz DOT)
- `kg-pipeline-flow.svg` — KG extraction pipeline flowchart (Graphviz DOT)
- `eval-scorer-flow.svg` — Evaluation scorer pipeline flowchart (Graphviz DOT)

**Regenerate locally:** Diagrams are not generated in CI. Run `make visualize` from the
project root when you change code that affects architecture (e.g. new modules, workflow
changes), then commit the updated `docs/architecture/diagrams/*.svg` files. Requires
[Graphviz](https://graphviz.org/) (`dot` on PATH) and dev dependencies:
`pip install -e .[dev]` (pydeps, pyan3, code2flow). Graphviz: `brew install graphviz`
(macOS), `apt install graphviz` (Debian/Ubuntu).

**Enforcement:** `make ci` and `make ci-fast` run `make visualize`; CI (python-app and docs
workflows) also runs it. If diagrams are older than the source files they depend on, the
check fails. Run `make visualize` and commit the updated SVGs to fix it.
