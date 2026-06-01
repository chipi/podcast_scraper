# RFC-092: ML Query Router

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-032-hybrid-corpus-search.md` — hybrid corpus search
  - `docs/prd/PRD-031-search.md` — Search product
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — ships the rules-based router this replaces
  - `docs/rfc/RFC-057-autoresearch-optimization-loop.md` — eval loop that produces labeled queries
- **Related Documents**:
  - `config/search.yaml` — router mode switch

> **Stabilization note (2026-05-30):** Split out of an earlier combined draft (RFC-079). This is the
> ML-router concern only. Of the three RFC-079 splits, **this one has its prerequisites met** — it
> needs labeled query data (from the eval loop), not new architecture — so it is the most readily
> shippable once data exists. It supersedes the rules-based router in RFC-090 §3.6 via a config
> switch; the rules router remains the default until the ML model is trained.

---

## Abstract

Upgrade the RFC-090 rules-based query router to an ML classifier. Input is the query text; output
is one of `{entity_lookup, raw_evidence, temporal_tracking, cross_show_synthesis, semantic}`, which
selects signal and tier weights for retrieval. The model is a small fine-tuned sentence-transformer
classification head exported to ONNX for local inference. Activation is config-driven; the rules
router stays the default until the model is trained on ≥500 labeled queries.

**Architecture Alignment:** Implements the existing `QueryRouter` interface
(`src/podcast_scraper/search/router.py`) so `RulesQueryRouter` and `MLQueryRouter` are
interchangeable behind a config flag. No retrieval-layer change. ONNX local inference keeps the
no-cloud-dependency constraint (runs on Apple M-series via ONNX Runtime).

## Problem Statement

The RFC-090 rules router covers obvious cases (capitalised names → entity lookup, "compare" →
synthesis) but has systematic blind spots: ambiguous queries, novel phrasings, and domain-specific
language misclassify. Misclassification degrades to sub-optimal signal/tier weights (RRF is robust,
so not wrong results — but worse ranking). An ML classifier trained on real query data is more
robust and improves continuously as query patterns emerge from the eval loop.

**Use Cases:**

1. **Ambiguous phrasing**: "what did the field think about scaling laws over the last year" should
   route to `temporal_tracking` even without an explicit date token.
2. **Domain language**: corpus-specific terminology routes correctly without hand-written rules.
3. **Continuous improvement**: misrouted queries corrected in the eval loop become training data.

## Goals

1. **Drop-in ML classifier** behind the existing `QueryRouter` interface.
2. **Local inference** via ONNX Runtime — no cloud dependency, cheap redeploy (replace file).
3. **Config-switchable** (`router.mode: rules | ml`) with the rules router as default.
4. **Trained on real data** — silver labels from the rules router + human corrections.

## Constraints & Assumptions

**Constraints:**

- Local inference only (ONNX Runtime on M-series MPS/CPU); no inference-time cloud calls.
- Label space is fixed at 5 classes; input is short text.

**Assumptions:**

- The eval loop (RFC-057) and viewer Search usage can produce ≥500 labeled queries before training
  is worthwhile.
- A ~100M-parameter encoder with a classification head is sufficient for 5-class short-text
  routing.

## Design & Implementation

### 1. Classifier Design

- **Input**: query text string.
- **Output**: one of `{entity_lookup, raw_evidence, temporal_tracking, cross_show_synthesis,
  semantic}`.
- **Model**: fine-tuned sentence-transformer classification head; ONNX export for local inference.
- **Training data**: query logs from the eval loop + viewer Search usage, labeled by (1) automatic
  silver labels from the rules router, (2) human correction of misclassified queries.

### 2. Interface

```python
# src/podcast_scraper/search/router.py

class QueryRouter:
    def classify(self, text: str) -> str: ...
    def signal_weights(self, query_type: str) -> dict: ...
    def tier_weights(self, query_type: str) -> dict: ...

class RulesQueryRouter(QueryRouter):
    """RFC-090 rules-based implementation. Default until the ML router is trained."""
    def classify(self, text: str) -> str:
        ...   # existing keyword/regex logic from RFC-090 §3.6

class MLQueryRouter(QueryRouter):
    """ONNX-exported classifier. Activated via config once trained."""
    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = ...   # matching tokenizer

    def classify(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="np")
        logits = self.session.run(None, dict(inputs))[0]
        return QUERY_TYPES[logits.argmax()]
```

### 3. Config-Driven Activation

```yaml
# config/search.yaml
router:
  mode: rules                                   # "rules" | "ml"
  ml_model_path: ./models/query_router.onnx     # only used if mode: ml
```

## Key Decisions

1. **ML router as ONNX, local inference.**
   - **Decision**: Export to ONNX; run locally.
   - **Rationale**: No cloud dependency; M-series runs ONNX Runtime efficiently; redeploy is a file
     swap + restart.
2. **Rules router stays the default.**
   - **Decision**: Ship `mode: rules` until the model is trained and A/B-validated.
   - **Rationale**: No regression risk before data exists; switch is one config line.

## Alternatives Considered

1. **LLM-based intent classification (API call per query).**
   - **Pros**: No training; flexible.
   - **Cons**: Per-query latency/cost; cloud dependency; overkill for 5 classes.
   - **Why rejected**: Violates the local-first constraint; a small local classifier suffices.
2. **Expand the rules router indefinitely.**
   - **Pros**: No model.
   - **Cons**: Rule sprawl; brittle on novel phrasings; no continuous improvement.
   - **Why rejected**: ML generalizes from real data the rules can't anticipate.

## Testing Strategy

**Test Coverage:**

- **Unit**: `MLQueryRouter.classify()` returns valid labels; config switch selects the right
  implementation; tokenizer/ONNX I/O shape.
- **Eval**: ML router vs rules router on a held-out query set with **human** labels (not
  rules-router silver labels).

**Test Organization:** `tests/unit/podcast_scraper/search/test_router.py`;
`src/podcast_scraper/eval/router_eval.py` for the A/B harness.

**Test Execution:** Unit in `ci-fast`; router eval is an operator/CI job, not per-PR.

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1 — Data collection**: instrument query logging in the eval loop and viewer Search;
  accumulate to ≥500 labeled queries.
- **Phase 2 — Train + export**: silver labels + human corrections; ONNX export; A/B eval.
- **Phase 3 — Switch**: flip `router.mode: ml` once the ML router beats rules on held-out human
  labels.

**Monitoring:** Track classification accuracy on the held-out set; track query-refinement rate
(PRD-031 success metric) as a proxy for routing quality in production.

**Success Criteria:** ML router ≥ rules router on held-out human-labeled accuracy; query-refinement
rate trends down after the switch.

## Relationship to Other RFCs

**Key Distinction:**

- **RFC-090**: ships the rules router (baseline) and the weight tables the router selects.
- **RFC-092 (this)**: replaces the router's `classify()` with an ML model behind the same interface.

| RFC | Relationship |
| --- | --- |
| RFC-090 | Defines `QueryRouter`, `SIGNAL_WEIGHTS`, `TIER_WEIGHTS_BY_QUERY`; this RFC swaps `classify()` |
| RFC-057 | Autoresearch eval loop is the training-data source |

## Benefits

1. **Robust routing** on ambiguous/novel phrasings the rules miss.
2. **Continuous improvement** as eval-loop corrections feed retraining.
3. **No cloud dependency** — local ONNX inference; cheap redeploy.

## Migration Path

1. **Phase 1**: Add query logging; keep `mode: rules`.
2. **Phase 2**: Train, export ONNX, validate A/B.
3. **Phase 3**: Flip `mode: ml`; retain `rules` as instant rollback.

## Open Questions

1. **OQ-1 Training-data quality.** Silver labels from the rules router are noisy precisely on the
   queries the ML router is meant to fix. Do a human annotation pass on a stratified sample before
   training; eval on human labels, not rules-router labels.
2. **OQ-2 Class boundary drift.** As query patterns evolve, the 5-class taxonomy may need a sixth
   class (e.g. explicit `corpus_coverage`). Version the label space.

## References

- **Related PRD**: `docs/prd/PRD-032-hybrid-corpus-search.md`
- **Related RFC**: `docs/rfc/RFC-090-hybrid-retrieval.md`,
  `docs/rfc/RFC-057-autoresearch-optimization-loop.md`
- **Source Code**: `src/podcast_scraper/search/router.py` (rules router from RFC-090)
- **Config**: `config/search.yaml`
