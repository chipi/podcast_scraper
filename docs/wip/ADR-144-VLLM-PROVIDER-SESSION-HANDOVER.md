# ADR-144 — first-class vLLM provider + fully-airgapped DGX profiles: session handover

**Session:** 2026-08-02 → 08-03 (autonomous completion while operator asleep).
**Branch:** `feat/naming-arc-and-corpus-prep` — **committed locally, NOT pushed** (operator reviews
+ pushes). Governing decision: `docs/adr/ADR-144-first-class-vllm-provider-real-model-ids.md`.

## What this delivered

The `autoresearch` served-model-name alias is gone from production config; the DGX profiles now
name the **real HF model id** on the wire and are **fully airgapped** (consume nothing from the
internet). `vllm` is a first-class provider, a sibling of `openai`.

### Commits (in order)

| Commit | What |
| --- | --- |
| `56fa7bc4` | ADR-144 (Accepted, advisor-reviewed) |
| `bacfed77` | **S1** — `OpenAIProvider` config-namespace + telemetry parameterized (zero-behaviour) |
| `da0bc1ef` | **S2** — `OpenAICompatibleProvider` base extracted; `VLLMProvider` sibling; config `vllm_*`; factories dispatch `vllm`; tests |
| `b14feeec` | **B2/B1** — registry real HF ids + governed `vllm_*` wire materialization; grounding local; value gate local; naming→vllm; **fully airgapped** profiles (ollama/local fallbacks) |
| `22b27dd6` | **B4** — `vllm` pricing section (both copies); coverage guard green |
| `763ab4d0` | **B3** — fail-closed `GET /v1/models` served-model verification at `initialize()` |
| (this) | ADR as-built amendment + ollama-symmetry deferral note + this handover |

### Airgapped as-built (all 3 DGX profiles: prod_dgx_balanced, prod_dgx_full_with_fallback, eval_default)

- summary / naming / GI / KG / quote / entailment → `vllm` (real Qwen id `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`).
- value gate: **enabled, self-grades with the same local model** (no cloud judge; `_LOCAL_ONLY_LLM`).
- summary fallback → **DGX-local ollama** (`:11434`); transcription → DGX-whisper + local whisper +
  MOSS (no cloud Whisper); diarization → local pyannote (no deepgram).
- Verified: all three load with **zero cloud API keys** and hold no cloud provider anywhere;
  `profiles-check` green; pricing sync + coverage green; provider/config unit tests green.

## NOT done — needs the operator / a live DGX

1. **Phased homelab migration (ADR-144 S3) — the live-run blocker.** The repo now requests the real
   HF id on the wire, but the homelab vLLM currently serves under `--served-model-name autoresearch`.
   Until the homelab serves the real id, the **B3 fail-closed check will reject the run**. Required
   change in **`agentic-ai-homelab` `infra/vllm/autoresearch/docker-compose.yml`**:
   - Phase A: serve BOTH names — `--served-model-name NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 autoresearch`
     (real id FIRST). This is backward-compatible (frozen eval configs still resolve `autoresearch`).
   - Phase B: migrate live eval tooling (`scripts/eval/onboard_model_smoke.py` default, sweep drivers,
     judges) off the `autoresearch` alias.
   - Phase C: drop the alias. Frozen `data/eval/configs/` referencing `autoresearch` are accepted as
     retired (never-mutate); new configs use the real id.
   - **This is an infra deploy — left for the operator (not touched autonomously).**
2. **Live DGX validation.** Bring the vLLM up (`gpu-mode-swap.sh research`) serving the real id, then
   run `scripts/eval/onboard_model_smoke.py` + one pipeline episode on `prod_dgx_balanced` to confirm
   the B3 check passes and the airgapped path works end-to-end. All validation so far is offline/unit.
3. **Ollama full symmetry — DONE** (was deferred; operator asked to un-defer). Ollama's wire config
   is now registry-governed + materialized like vllm's (primary AND airgapped-fallback), via a
   `speaker_llm_model` split that also **fixed a latent bug** (the vllm naming option was leaking a
   Qwen HF id into `ner_model`, a spaCy field). See the ADR as-built amendment.
4. **The naming model is a placeholder.** `vllm_speaker_detector` defaults naming to the summary
   daily-driver Qwen; the winning naming model is still the bake-off's call (ADR Part-1 Step 3).

## The original arc this unblocks

This was **Step 0** of the v2.5 corpus arc (`docs/wip/1000-EPISODES-REPROCESS-PLAN.md`). With the
representation now correct + airgapped, the bake-off (Part 1, Steps 1–5) can run on a self-describing,
reproducible, fully-local pipeline.
