# ADR-077: Local Ollama Model Selection — Core 5 from 11

**Status:** Accepted
**Date:** 2026-04-18
**Context:** Autoresearch v2 eval (PR #568) + pipeline validation (#591)

---

## Decision

Standardize on **5 Ollama models** (from 11 previously tested) for regular sweeps,
pipeline validation, and optimization work. One per model family plus one large-model
reference.

## Core 5

| Model | Family | Size | v2 Bullets | Role |
| ----- | ------ | :--: | :--------: | ---- |
| **qwen3.5:9b** | Qwen 3.5 | 9B | 0.580 | **Champion** — best local quality, default pick |
| **llama3.1:8b** | Llama | 8B | 0.518 | **Llama family** — full pipeline at 8B (llama3.2:3b demoted: KG unreliable at 3B) |
| **mistral:7b** | Mistral | 7B | 0.526 | **Mid-tier** — best non-Qwen, balanced |
| **gemma2:9b** | Gemma | 9B | 0.492 | **Diversity** — Google architecture |
| **qwen3.5:35b** | Qwen 3.5 | 35B | 0.576 | **Scale reference** — "does bigger help?" |

## Dropped (6 models)

| Model | Size | v2 Bullets | Why dropped |
| ----- | :--: | :--------: | ----------- |
| qwen3.5:27b | 27B | 0.543 | Worse than 9b at 15× latency. Same family as champion. |
| qwen2.5:7b | 7B | 0.477 | Previous generation; superseded by qwen3.5:9b on every metric. |
| llama3.1:8b | 8B | 0.518 | Same family as llama3.2:3b; marginal gain (+3%), paragraph contested 5/5. |
| mistral-nemo:12b | 12B | 0.497 | Worse than mistral:7b (-6%) despite 2× size. |
| mistral-small3.2 | 22B | 0.536 | Only +1% bullets over mistral:7b; paragraph contested; 79s/ep. |
| phi3:mini | 3.8B | 0.475 | 4k context variant truncates our ~9k prompts silently. Structurally unsuitable. |

## Rationale

### Family diversity over family depth

Testing qwen3.5 at 9b, 27b, AND 35b measures the same model family at different
scales — useful once to establish "bigger ≠ better" (confirmed: 9b > 27b > 35b
on paragraph), not useful to repeat on every sweep.

One model per family tests whether a finding is architecture-specific or general.
If qwen3.5:9b passes pipeline validation but gemma2:9b fails, the issue is
Gemma-specific. If we tested three Qwen variants and they all passed, we'd still
not know about Gemma.

### Size diversity matters for deployment decisions

The 5 models span 3B → 35B:

- **3B (llama3.2):** edge devices, resource-constrained, speed-first
- **7-9B (mistral, qwen, gemma):** laptop/workstation, balanced
- **35B (qwen3.5):** dedicated GPU, quality-first

This tells us whether pipeline fixes work across the full size spectrum.

### Sweep time reduction

- Before: 11 models × 3 tasks × 5 episodes = ~4 hours
- After: 5 models × 3 tasks × 5 episodes = ~1.5 hours
- 60% time reduction while retaining all family + size signal.

## Pipeline validation findings (2026-04-18, #591)

Full pipeline validation (summary → GI → KG → bridge) on 5 held-out episodes:

| Model | Summary | GI | Grounding | KG | Bridge |
| ----- | :-----: | :-: | :-------: | :-: | :----: |
| qwen3.5:9b | ✅ | ✅ | 100% | ✅ | ✅ |
| llama3.1:8b | ✅ | ✅ | ✅ | ✅ | ✅ |
| mistral:7b | ✅ | ✅ | 98% | ✅ | ✅ |
| gemma2:9b | ✅ | ⚠️ 7.8/ep | 95% | ✅ | ✅ |
| qwen3.5:35b | ✅ | ✅ | 98% | ✅ | ✅ |

**llama3.2:3b (3B) demoted:** KG entity extraction fails (0 entities on 2/5
episodes). 3B params insufficient for structured JSON extraction. Replaced
by llama3.1:8b (8B) which passes all stages.

**gemma2:9b instruction-following gap:** produces 7-8 insights when asked for
12 (avg 7.8/ep, threshold 8). Not a pipeline bug — Gemma2 is concise by
nature. Grounding (95%) and KG are fine. Keep in Core 5 for architecture
diversity but note the GI limitation.

**Minimum viable size for full pipeline: 7-8B.** All 7B+ models pass (gemma2
borderline on insight count but functional).

## Consequences

- `pipeline_validate.py` LOCAL_PROVIDERS reduced from 11 to 5
- `--local-fast` flag skips the 35b model (runs 4 models in ~1 hr)
- Dropped models are NOT deleted from Ollama or from v2 eval configs — historical
  runs remain for reference. They're just excluded from future regular sweeps.
- If a new model generation ships (e.g., Llama 4, Gemma 3), add the best variant
  to Core 5 and drop the old family member.

## Related

- EVAL_HELDOUT_V2_2026_04.md §Local models (full 11-model matrix)
- AI_PROVIDER_COMPARISON_GUIDE.md §Local picks
- #591 (pipeline validation — uses Core 5)
