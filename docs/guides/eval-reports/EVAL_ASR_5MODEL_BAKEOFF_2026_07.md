# EVAL — 5-model ASR bake-off (2026-07-22, corrected 2026-07-23)

> ## ⚠️ CORRECTION (2026-07-23) — read first
>
> **The original "ground truth" below was NOT human ground truth.** The e071 "delivered Latent Space
> transcript" was actually **our own old pipeline ASR** (verified: the corpus run had
> `transcripts_downloaded=0, transcribed=10`, and the flightcast feed carries no `podcast:transcript`
> tags). So the "silver inversion" / "Table 2 vs real ground truth" narrative below is **retracted** —
> it measured agreement with our old ASR, not accuracy.
>
> **The real measurement** now exists: a human-ground-truth run against **80,000 Hours** publisher
> transcripts (native RSS, verified human), n=10, single-show, in the private eval-data repo
> (`data/eval/runs/asr_bakeoff_80k_2026-07-23`). Mean WER vs the **human** transcript:
>
> **Cloud vertical** (quality × $/min; RTF ~instant for both):
>
> | model | WER↓ | $/min | verdict |
> | --- | --- | --- | --- |
> | **openai-whisper-1** | **11.4%** | ~$0.006 | **cloud transcription winner** |
> | deepgram-nova-3 | 13.9% | ~$0.0043 | cheaper but less accurate → **diarization only, not transcription** |
>
> **DGX vertical** (quality × speed; $≈0 self-hosted):
>
> | model | WER↓ | RTF (speed) | verdict |
> | --- | --- | --- | --- |
> | **MOSS** | **12.5%** | 2.9× | slightly more accurate, slow |
> | **turbo** | 13.5% | **25×** | slightly less accurate, ~9× faster → primary on speed |
> | large-v3 | 16.3% | 7× | **worst** — beaten by its own turbo distillation; drop/investigate |
>
> **What this changes vs the (bogus) original:**
>
> - **openai = best**, validated (was "best" originally too — conclusion survives, basis was wrong).
> - **MOSS is 2nd-best accuracy**, NOT "near the bottom" as the retracted run claimed. Its demotion
>   holds **only on speed** (slowest), not accuracy. The DGX choice is genuinely **turbo (speed) vs
>   MOSS (accuracy)**.
> - **turbo-primary is a SPEED decision**, not an accuracy one (mid-pack accuracy).
> - **large-v3 last** — surprising (its own turbo variant beats it); likely a speaches/int8 serving
>   issue, and a weak coverage-failover target. Tracked in **#1273** (kept as failover for now).
> - **Deepgram = diarization only** (cheap, cloud-native speaker labels), confirmed not a
>   transcription contender.
>
> **Caveats (do not over-read):** n=10, **single-show (80k Hours)** — not yet multi-show, so no
> generalization. Human transcripts are edited → absolute WER (11–16%) is inflated; the *relative*
> ranking is the signal. Multi-show (Dwarkesh + Lex, minutes-matched) is the next step before any
> further registry change.
>
> Everything below this banner is the **retracted original** — kept for the paper trail, not for use.

---

Speed + quality comparison of five ASR models, and the transcription-tier decisions it drove in the
model registry. Supersedes the per-model transcription claims in
[EVAL_MOSS_BAKEOFF_2026_07](EVAL_MOSS_BAKEOFF_2026_07.md) and
[EVAL_DEEPGRAM_TRANSCRIPTION_2026_06_13](EVAL_DEEPGRAM_TRANSCRIPTION_2026_06_13.md) — see
"The silver inversion" below.

## TL;DR

- **Primary transcription = `large-v3-turbo`** (DGX speaches :8000). 25× realtime, accuracy tied for
  best on the only real-ground-truth episode.
- **Secondary = `large-v3`** (`Systran/faster-whisper-large-v3`), used as the ADR-123 coverage-gate
  failover (turbo silently drops speech on long episodes) and first infra fallback.
- **Cloud transcription = `openai-whisper-1`** (best on real ground truth); Deepgram demoted for
  transcription but **kept as the cloud diarizer**.
- **MOSS demoted to fallback** — its earlier "win" was an artifact of a bad reference.

## Method

- 10 episodes, one per show from the prod corpus (e001, e011, e021, e031, e041, e051, e061, e071,
  e081, e086), **all on `speech_optimal_v1`-preprocessed audio** (16 kHz mono, 32 kbps — the exact
  preprocessing the reprocess pipeline feeds every provider). All five models see identical input.
- Each model runs via the **shipped** provider/chunker path (no ad-hoc code). DGX models over the LAN
  IP; OpenAI/Deepgram cloud.
- WER: word-level Levenshtein (rapidfuzz), lowercase + punctuation-stripped normalization.
- Raw detail: `.test_outputs/moss-eval/prod/FINAL_BAKEOFF.json` (git-ignored — derived from real
  episodes).

## Table 1 — WER vs Deepgram nova-3 "silver" + speed

**WER here = agreement-with-Deepgram, NOT accuracy** (see the inversion below).

| model | mean WER | median | agg RTF |
| --- | --- | --- | --- |
| MOSS | 5.2% | 4.7% | 2.9× |
| openai-whisper-1 | 6.8% | 6.8% | 19.9× |
| turbo | 7.9% | 6.4% | 25.1× |
| large-v3 | 8.9% | 7.9% | 7.4× |

## Table 2 — e071 vs the DELIVERED transcript (real ground truth, n=1)

e071 = Latent Space "Doing Vibe Physics" — the only episode in the set with a **publisher-delivered
transcript** (flightcast feed; captured in the prod-v2 corpus as `rss.flightcast.com_c63_e01.txt`).
Delivered transcripts are edited (filler/ads stripped) so absolute WER is inflated for all, but the
**relative** ranking is valid.

| model | WER vs ground truth |
| --- | --- |
| openai-whisper-1 | 7.4% |
| turbo | 7.5% |
| large-v3 | 8.4% |
| MOSS | 8.7% |
| Deepgram nova-3 | 10.0% |

## The silver inversion (the load-bearing finding)

The two tables **rank the models in opposite order.** Against Deepgram-silver, MOSS looks best
(5.2%). Against real ground truth, **Deepgram is the *least* accurate of the five** — so Table 1 is
measuring "who agrees with the worst model," not accuracy. MOSS's Table-1 lead is
agreement-with-Deepgram; on the one episode with truth it is near the *bottom* (8.7%).

Consequence: prior transcription rankings that scored against a Deepgram (or OpenAI) silver — #1174
(MOSS), #952 (large-v3) — measured *agreement with an unvalidated reference*, not accuracy. This
bake-off is the first with a real delivered transcript in the loop.

## Decisions → registry (`model_registry.py`)

StageOption tiers (transcription):

| option | model | tier | why |
| --- | --- | --- | --- |
| `tailnet_dgx_whisper_turbo` | turbo | **primary** | 25× speed, GT-accuracy tied-best (7.5%) |
| `tailnet_dgx_speaches_thread_b` | large-v3 | **fallback** | coverage-failover + first infra fallback |
| `tailnet_dgx_whisper_openai` | large-v3 (:8002) | **deprecated** | superseded by speaches large-v3 + turbo |
| `openai_whisper_1` | whisper-1 | **primary** (cloud) | best on GT (7.4%) |
| `moss_transcribe_diarize` | MOSS | **fallback** | GT 8.7% (near bottom), slowest; silver "win" inverted |
| `deepgram_nova_3` | nova-3 | **fallback** | worst on GT (10%); stays PRIMARY for cloud *diarization* |

Profile impact (regenerated via `make profiles-materialize`):

| profile | transcription before → after | notes |
| --- | --- | --- |
| `prod_dgx_balanced` | MOSS → **turbo** + coverage gate (0.85→large-v3) | serving |
| `prod_dgx_full_with_fallback` | MOSS → **turbo** + coverage gate | serving |
| `cloud_with_dgx_primary` | MOSS → **turbo** + coverage gate | DGX-primary hybrid |
| `cloud_quality` | Deepgram → **openai-whisper-1** | diarization stays Deepgram |
| `cloud_balanced`, `cloud_thin` | openai-whisper-1 (unchanged) | already correct |
| `reprocess_dgx_{turbo,no_llm}` | turbo (unchanged) | already turbo + large-v3 gate |

Governance/plumbing change: **`dgx_whisper_model` is now a `REGISTRY_GOVERNED_FIELD`** and the
resolver emits it (turbo vs large-v3) for `tailnet_dgx_whisper` transcription — mirroring the
diarization-model routing. Without this the provider fell to the Config default (`large-v3`) and a
"turbo" profile would silently run large-v3. Now materialized + drift-checked.

Diarization is unchanged: DGX/local profiles diarize with pyannote; **cloud profiles diarize with
Deepgram** (`deepgram_diarization_nova3` — a separate StageOption from the demoted transcription
one), so demoting Deepgram-transcription does not create a cloud diarization gap.

## NOT done / NOT verified

- **Accuracy beyond e071 is unverified.** The other 9 episodes have only Deepgram-agreement (Table 1),
  which Table 2 shows is unreliable. Real accuracy needs more ground-truth episodes.
- **n=1 ground truth.** The turbo-over-MOSS and openai-over-Deepgram accuracy claims rest on a single
  delivered transcript. Directional, not final. Next step to harden: hand-transcribe 2–3 short
  segments across shows, or find more delivered-transcript feeds, and re-rank.
- **Turbo's long-episode coverage drop** is real (ep6 in the isolated bake-off: 69% coverage). It is
  *mitigated* by the ADR-123 coverage gate (re-route to large-v3), not eliminated — the gate is now
  on for every turbo profile, serving included.
