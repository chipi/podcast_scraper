# Work In Progress (WIP) Documentation

This folder holds early-stage notes, backlog plans, and reference material that has
not been promoted to a PRD, RFC, ADR, runbook, or release note. Content here is
**not authoritative** — when an item is ready, it moves to its proper home and the
WIP file is removed.

## Promotion targets

- Long-term feature ideas → **PRD** in `docs/prd/` and the relevant **RFC** in `docs/rfc/`
- Architectural decisions → **ADR** in `docs/adr/`
- Operator-facing procedures → **guide** in `docs/guides/`
- Post-release retrospective → **fold the substance into** the corresponding release note in
  `docs/releases/` (do NOT leave it in WIP and link to it — see the rule below)

## References are ONE-WAY: permanent docs must never point back into `docs/wip/`

A WIP doc is deleted the moment its work lands, so anything permanent that links to it becomes a
dead reference. **No ADR / RFC / PRD / release note / guide / `docs/api/` doc, and no code / test /
README / comment / docstring, may reference a `docs/wip/…` path.** (WIP↔WIP is fine — a WIP set
travels together; and a WIP doc pointing *at* a permanent artifact is just naming its promotion
target.) When a permanent doc needs WIP content, **promote it** — copy the substance inline or cite
the durable source (shipped code, commit hash, issue/PR, or the ADR/RFC/PRD that superseded it).
This is the rule the 2026-08-02 hygiene pass had to retrofit; the templates and AGENTS.md carry it.

## Rule: only open work lives here

A note here describes work that is still open. When it lands, is superseded, or nobody would
miss it, it is deleted (git keeps the history) or its substance is promoted. Dated session
handovers, run logs, one-off measurements and screenshots of shipped UI do not stay. The
2026-10-29 pass cut this folder from 249 files to the list below.

## Open notes

| Note | What is open |
|---|---|
| [HANDOVER-NAMING-DEBT-2026-10-27.md](HANDOVER-NAMING-DEBT-2026-10-27.md) | Speaker naming and migration debt after the 0034 / 0035 / 0036 release: numbered items, most open |
| [NONENGLISH-REFERENCE-SHOWS-2026-10-09.md](NONENGLISH-REFERENCE-SHOWS-2026-10-09.md) | The non-English arc: readiness per language, the gate decisions, #2352 and #2353 |
| [DIARIZATION-SPLIT-HOST-CLUSTER-MERGE.md](DIARIZATION-SPLIT-HOST-CLUSTER-MERGE.md) | The deferred root-cause fix for diarization merge/split artifacts (#1330) |
| [DEVICE-ONLY-DEFECTS-2026-09-27.md](DEVICE-ONLY-DEFECTS-2026-09-27.md) | Three defects seen only on the iPhone build, not yet solved |
| [SPEND-ESTIMATOR-OVERSTATEMENT-2026-08-28.md](SPEND-ESTIMATOR-OVERSTATEMENT-2026-08-28.md) | Run-reported LLM cost overstates gateway spend about 3.5x; the fix is not built |
| [wip-concurrent-pipeline-http-retry-metrics.md](wip-concurrent-pipeline-http-retry-metrics.md) | The process-wide HTTP retry counter clashes when pipelines run concurrently |
| [TODO-remove-pre-1.0.3-compat.md](TODO-remove-pre-1.0.3-compat.md) | Compatibility paths to remove once no client older than 1.0.3 is in use |
| [graph-tech-debt.md](graph-tech-debt.md) | Deferred graph-viewer work: layout cost, initial parse, rebuilds, preferences UI |

## Folders

| Folder | What it holds |
|---|---|
| [changes/](changes/README.md) | Corpus change records ([README](changes/README.md), [TEMPLATE](changes/TEMPLATE.md)). The records for migrations 0031 to 0037 stay: the change gate's landing check needs every registered migration's record and gate file. |
| [parked/](parked/README.md) | One parked patch (cleaned-base LLM selection), kept for later |
| `ideas/knowledge-retention/` | The knowledge-retention product package (vision, PRD and RFC drafts, July 2026), kept as an idea, not open work |

## Guidelines

- Documents here are **not** part of the official documentation site.
- Documents may be incomplete, outdated, or experimental.
- Periodically review and either:
  1. Promote to the appropriate doc category (PRD / RFC / ADR / guide / release note).
  2. Delete if obsolete or superseded.
  3. Keep as backlog / reference if it still has signal.
