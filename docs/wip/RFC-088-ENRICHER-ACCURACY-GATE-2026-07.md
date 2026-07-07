# RFC-088 amendment — enricher accuracy gate: data/eval → registry → profiles → UI

Status: **Shipped** (2026-07-06) · Amends `docs/rfc/RFC-088-enrichment-layer-architecture.md`

> The gate is live on `main`-bound code: `src/podcast_scraper/enrichment/eval/`
> (`gate.py` + `admission.py` + `runner.py` + `gold.py`) drives `profile_sets._admit()`,
> which filters the registry → profiles → UI. `nli_contradiction` (#1106) and
> `stance_disagreement` (#1144) are both gated dark by it today (0% precision); each
> auto-promotes when an eval records precision ≥ 0.5, no code edit.

## Problem

The enrichment **runtime** is already a modular framework (protocol + registry +
`config_schema` + provider injection + profile matrix — see
`src/podcast_scraper/enrichment/`). Adding an enricher is ~1 file + ~3 registration
lines; adding a knob is one `config_schema` entry.

The **accuracy / gold / gating** side is not. Today:

- Each accuracy eval is a bespoke hand-written script trio under `scripts/eval/score/`
  (`enrichment_nli_{harvest,silver,eval}_v1.py`, `enrichment_topic_similarity_*`,
  `disagreement_stance_*`). No shared scorer protocol/registry.
- Whether an enricher ships is a **hardcoded list edited by hand**. The clearest
  symptom: `nli_contradiction` was disabled by *commenting it out* of
  `profile_sets._cloud_ml_tier_set()` after its eval measured 0% precision (#1106).
  A code edit, not a data-driven decision.
- Gold on the fixture side was about to hardcode a *named field per enricher*
  (`expected_velocity`, `expected_perspectives`, …) into the v3 generator.

## Decision

Give the eval/gate side the same modularity the runtime already has, and make
**`data/eval` accuracy gate registry membership** — which cascades to profiles + UI
config — **exactly mirroring how providers are evaluated and gated**
(`evaluation.regression.RegressionRule` → `RegressionChecker` → hard-fail;
`config/acceptance/`).

```
enricher output ──┐
                  ├─► AccuracyScorer.score(output, gold) ─► metrics ─► data/eval/…
gold (generic) ───┘                                                        │
                                                                           ▼
                                          manifest.accuracy_gate ─► AccuracyGate
                                                                           │
                                              promote? ──────────┬─────────┘
                                                                 ▼
                                    admitted_enricher_ids()  (the cascade point)
                                                                 │
                        ┌────────────────────────┬───────────────┴───────────────┐
                        ▼                        ▼                                ▼
                   registry                profile_sets                    UI config route
             (register_* consult)     (enricher_set_for_profile)      (enrichment_config.py)
```

### Components (all in `src/podcast_scraper/enrichment/eval/`)

| Piece | Mirrors | Role |
|---|---|---|
| `AccuracyScorer` protocol + `ScorerManifest` + `ScoreResult` | `Enricher` / `EnricherManifest` / `EnricherResult` | grade one enricher's output vs gold → metrics. Sync (grading is pure). |
| `ScorerRegistry` | `EnricherRegistry` | register/get scorers keyed by `enricher_id`. |
| generic gold: `expected_enrichment[<enricher_id>]` | `manifest.writes` keys the *output* | one gold block per enricher, keyed by `manifest.id`. **No per-enricher field/name anywhere** (kills the "cosmetic" hardcoding). |
| `AccuracyGateRule` / `AccuracyGateSpec` (declared on `EnricherManifest`) | `RegressionRule` (declared in `RegressionChecker`) | each enricher declares its own accuracy bar beside `config_schema`. |
| `eval/gate.py` `evaluate_gate()` + `GateDecision` | `RegressionRule.check()` / `RegressionChecker` | metric ≥ threshold → promote; else reject. `on_missing_data` policy governs no-eval-yet. |
| `eval/admission.py` `admitted_enricher_ids()` | (new cascade point) | candidates + declared gates + `data/eval` metrics → admitted set + decisions. |

### Admission semantics (behaviour-preserving)

- **No `accuracy_gate` declared → always admitted.** All six deterministic enrichers
  and `topic_similarity` have no gate today → unchanged shipping set.
- **Gate declared, `on_missing_data="reject"`, no passing eval → gated OUT.**
  `nli_contradiction` declares `precision ≥ 0.5, on_missing=reject`. With no passing
  `data/eval` record it is excluded — reproducing today's manual disable, but as a
  *data-driven consequence*. When a future eval writes a passing precision, it
  **auto-promotes** with no code edit. The hand-commented list in
  `profile_sets._cloud_ml_tier_set()` is replaced by this.
- Decisions are surfaced (returned + loggable) so the UI can show *why* an enricher
  is off ("gated: precision 0.00 < 0.50"), not just silently absent.

## Scope now vs later (operator: mechanism now, content later)

- **Now (this change): mechanism only.** The scorer protocol/registry, generic gold
  schema, gate + manifest declaration, admission cascade + wiring, reference scorers
  covering the three metric shapes (scalar / ranking / set), tests. No fixture
  content and no committed gold *values* are authored.
- **Later (joint): content.** Author `expected_enrichment[...]` gold values + v3
  fixtures; add a scorer per remaining enricher (same 1-file pattern). Optional:
  a `config/acceptance`-style YAML override for gate thresholds (mirrors
  `RegressionChecker.from_config`).

## References

- Runtime framework: `src/podcast_scraper/enrichment/{protocol,registry,profile_sets}.py`
- Provider gate mirrored: `src/podcast_scraper/evaluation/regression.py`,
  `evaluation/comparator.py`, `config/acceptance/`
- Evidence for the nli gate: #1106 (0% precision), `scripts/eval/score/enrichment_nli_*`
- Fixture spec this supersedes on the gold side:
  `docs/wip/FIXTURE-CAPABILITY-GAP-ANALYSIS.md` §"v3+ generator-evolution SPEC"
