# EVAL — MOSS-Transcribe-Diarize bake-off (transcription vs whisper, diarization vs pyannote)

**Issue:** #1174 / #1177
**Date:** 2026-07-16
**Hardware:** DGX GB10 (sm_121), `nvcr.io/nvidia/vllm:26.05-py3` (torch 2.12/CUDA13, transformers 5.6)
**Runtime:** MOSS via the upstream `moss_transcribe_diarize` package (bare transformers)
**Fixture artifacts:** [`tests/fixtures/eval/moss_bakeoff_v1/`](../../../tests/fixtures/eval/moss_bakeoff_v1/)
**Harness:** [`scripts/eval/moss/`](../../../scripts/eval/moss/)

## TL;DR

MOSS-Transcribe-Diarize is a **strong transcriber, not a diarization or speed win**, and it **needs
audio chunking** to work on real-length episodes.

| Axis | MOSS | incumbent | winner |
| --- | --- | --- | --- |
| Transcription WER (prod, vs Deepgram silver) | **5.2%** | whisper large-v3 8.5% | **MOSS** (all 10 eps) |
| Diarization DER (prod, vs Deepgram silver) | 24.2% | pyannote community-1 19.0% | **pyannote** |
| Diarization DER (fixtures, vs RTTM truth) | **3.3%** | pyannote 7.9% | **MOSS** |
| Transcription WER (fixtures, vs source text) | 9.0% / 3.1% med | whisper 8.4% / 4.3% med | tie |
| Speed | 2.1–3.8× realtime | large-v3 7.8× | incumbent |

**Adoption read:** use MOSS for **transcription** (beats whisper on real audio) and keep pyannote for
diarization; or single-model MOSS where simplicity beats best-in-class diarization + throughput.
The 50–100× speed pitch does **not** hold in bare transformers (needs the untested-on-sm_121 vLLM
backend). English quality — the #1174 load-bearing unknown — is **resolved: excellent.**

## Part 1 — fixtures (45 v3 RTTM fixtures, exact ground truth)

Diarization (neither model given a count hint): **MOSS DER 3.3% / median 1.8%** vs **pyannote 7.9%
/ 6.9%**; count exact MOSS 38/45 vs pyannote 40/45, both 45/45 within-1. MOSS beats pyannote on DER.

Transcription (WER vs the source text): MOSS 9.0% mean / 3.1% median; whisper 8.4% / 4.3%; MOSS
wins 25/45. A tie — MOSS better median, whisper better mean (a few multi-accent/garble fixtures
skew MOSS's mean).

## Part 2 — real prod audio (10 episodes, one per show), reference = Deepgram nova-3 silver

Real audio has no ground-truth labels, so all systems are scored against a shared **Deepgram
nova-3** silver (precedent: #952). Raw speaker **count** is not a usable metric on real audio — both
Deepgram (up to 13) and pyannote (up to 22) over-count via clip/soundbite fragments on news shows;
the **dominant-speaker count (≥5% time)** is the sane metric (all 10 episodes: 2–4 real
participants). Judged on **DER** (time-weighted) + dominant count.

**Result:** MOSS transcription **5.2%** WER vs whisper **8.5%** — MOSS wins every episode. MOSS
diarization **24.2%** DER vs pyannote **19.0%** — pyannote wins (MOSS fragments on clip-heavy news
episodes, though dominant count stays 2–6 vs Deepgram's 2–4).

### The truncation finding (and why chunking is mandatory)

A **single MOSS pass truncates at ~30 min** — the 128k context fills with audio features on long
input, then generation stops. First-pass prod numbers were a coverage artifact (WER 29%, DER 37%,
coverage 40–98%). **Audio chunking fixes it**: e011 went 71% → 100% coverage, WER 38.7% → 12.2%;
the pooled prod WER dropped 29.4% → **5.2%**. The pipeline already has `AudioChunker` (used by the
API providers) — MOSS just needs to opt in (see the prod fix below).

## Root cause / fix

MOSS is a chat-style multimodal model whose audio + generated tokens share a 128k context, so it
cannot emit a full transcript for a >~30 min episode in one call — a *quality ceiling that behaves
like a hard duration limit*, exactly like the Voxtral chat models already handled in
`transcription_max_chunk_duration_seconds`. **Fix:** add `moss` to `_API_CHUNKING_PROVIDERS`
(`workflow/episode_processor.py`) and a `moss` branch to `transcription_max_chunk_duration_seconds`
(~1500 s), so `episode_processor` chunks MOSS through the existing `AudioChunker` split→merge path
(with the already-documented chunk-local speaker-id caveat).

## Acceptance (#1174)

- **English quality** — resolved: MOSS transcription **beats whisper large-v3** on real prod audio.
- **Diarization** — MOSS beats community-1 on clean fixtures but **loses on real audio**; does not
  clear the "beat community-1 in production" bar.
- **Speed** — 2.1–3.8× bare transformers; **no win** over large-v3's 7.8× without vLLM.
- **Comparable run over the prod-v3 pilot shows** — done (10 shows, this report).

## Follow-ups

- Ship the chunking fix (above) so MOSS is usable on long audio — part of #1177.
- vLLM backend on GB10 (sm_121) — the only path to the 50–100× speed claim; untested, filed under
  #1177's "backend" note.
- Golden hand-labeled prod references (#1189 / #1170 real-10) — to grade against gold, not silver.
