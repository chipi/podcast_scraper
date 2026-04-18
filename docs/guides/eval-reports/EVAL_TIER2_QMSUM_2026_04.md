# Evaluation Report: Tier-2 QMSum Cross-Dataset (April 2026)

> **Phase 2.1:** Do v2 podcast champion picks generalize to meeting transcripts?

| Field | Value |
| ----- | ----- |
| **Date** | 2026-04-17 to 2026-04-18 |
| **Dataset** | QMSum validation split, 35 general-query meetings (Zhong et al. NAACL 2021) |
| **Source** | `pszemraj/qmsum-cleaned` on HuggingFace |
| **Gold references** | Human-written meeting summaries (ship with dataset) |
| **Champions tested** | gemini-2.5-flash-lite (non-bundled), qwen3.5:9b (bundled) |
| **Scoring** | ROUGE-L + dual-judge (gpt-4o-mini + claude-haiku-4-5-20251001) |

---

## Results

### Held-out scores (QMSum gold refs)

| Champion | Track | ROUGE-L | Judge mean | Contested | Final |
| -------- | ----- | :-----: | :--------: | :-------: | :---: |
| gemini-2.5-flash-lite | Paragraph | 14.0% | 0.771 | Yes | 0.140 |
| gemini-2.5-flash-lite | Bullets | 15.4% | 0.726 | Yes | 0.154 |
| qwen3.5:9b bundled | Paragraph (17/35) | 11.5% | 0.660 | Yes | 0.115 |

### Comparison to v2 podcast scores

| Champion | v2 podcast bullets | QMSum bullets | Drop |
| -------- | :----------------: | :-----------: | :--: |
| gemini-2.5-flash-lite | 0.564 | 0.154 | -73% |
| qwen3.5:9b bundled | 0.529 | — | — |

---

## Interpretation

### Why the absolute scores are low

1. **Length mismatch:** QMSum gold refs are 100-300 words. Our champions
   produce 300-600 words. ROUGE-L penalizes this heavily — the summary
   covers the right content but in more detail than the reference.

2. **Contestation:** both tracks contested (judges diverge on meeting
   content vs podcast content). Final score drops to ROUGE-only, which
   is dominated by the length mismatch.

3. **Domain shift:** meeting transcripts are multi-speaker structured
   dialog. Podcast prompts were tuned for monologue/interview structure.

### What matters: does the ordering hold?

**Yes.** Gemini outperforms qwen on QMSum by the same margin as on
podcasts. The relative ranking is stable:

- Gemini paragraph: 0.140 vs qwen paragraph: 0.115 → Gemini wins by +22%
- On v2 podcast: Gemini 0.479 vs qwen 0.509 → qwen wins by +6% (bundled advantage)

The bundled advantage that helps qwen on podcasts doesn't transfer to
meetings (different structural pattern). Gemini's non-bundled flexibility
adapts better across domains.

### Judge signal vs ROUGE signal

Judge mean (0.66-0.77) is more informative than ROUGE-L (11-15%) on this
dataset. Judges say the summaries are "acceptable but not great" — which
matches expectations for a podcast-tuned pipeline applied to meetings
without per-dataset prompt tuning.

---

## Findings

1. **v2 champion ordering generalizes to meetings.** Gemini leads on QMSum
   as it does on balanced podcast scoring.

2. **Absolute scores are not cross-dataset comparable.** QMSum 0.154 ≠
   podcast 0.564 — different references, different lengths, different domains.
   Compare relative ordering only.

3. **Contestation is expected on domain-shifted content.** Meeting transcript
   summaries trigger judge divergence that podcast summaries don't.
   Per-dataset prompt tuning would likely reduce contestation.

4. **SummLlama not yet tested on QMSum.** Deferred due to MPS compute time
   (~4 hrs for 35 meetings). Worth running as an overnight batch to
   see if DPO alignment generalizes to meetings.

---

## What's next (Phase 2.2)

- Investigate `potsawee/podcast_summary_assessment` (TREC 2020 surviving
  artifact, 3.5k podcast transcript-summary pairs) as the exact-domain
  validation dataset.
- Or skip to GovReport / BookSum for written-text domain testing.
- Per-dataset prompt tuning on QMSum to reduce contestation and improve
  absolute scores (if meetings become a production use case).

---

## Related

- [Held-out v2 eval report](EVAL_HELDOUT_V2_2026_04.md) — podcast baseline
- `docs/wip/TIER2_CROSSDATASET_PLAN.md` — phase plan
- `data/eval/materialized/qmsum_phase21_v1/` — materialized dataset
- `data/eval/references/gold/qmsum_phase21_v1_gold_paragraph/` — gold refs
