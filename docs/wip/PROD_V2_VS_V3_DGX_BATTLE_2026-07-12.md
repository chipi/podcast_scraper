# prod-v2 (cloud) vs prod-v3 (DGX-only) — running observations

**Status:** live · started 2026-07-12 · corpus build in progress
**Question:** can the fully-local DGX stack match the cloud one — and does the DGX investment pay?

Running notes, to be assessed later together with `data/eval`. Not authoritative; findings that
survive get promoted (eval report → `docs/guides/eval-reports/`, decisions → ADR).

---

## 1. The question, stated honestly

The naive framing ("DGX saves API spend") **does not survive contact with the data**. The v2 corpus
manifest records what the cloud actually charged to build all 99 episodes:

| stage | cloud cost |
| --- | ---: |
| GI insights | $0.333 |
| summarization | $0.117 |
| speaker detection | $0.064 |
| KG extraction | $0.000 |
| **total (99 episodes)** | **$0.514** |

**Fifty-one cents to build the entire corpus.** The DGX cannot pay itself back on corpus builds.
If the investment is to make sense the case must rest on:

- **quality parity** — can local match cloud at all? (if not, cost is irrelevant)
- **unmetered experimentation** — our eval sweeps and judge matrices are where cloud spend actually
  explodes, not the pipeline
- **privacy** — audio and transcripts never leave the house
- **no rate limits / no vendor lock-in**

Operator decision (2026-07-12): score **quality parity** first. The rest is moot if local loses.

## 2. What actually differs between v2 and v3

Established from v2's own `config_snapshot`, not assumed:

| stage | v2 (cloud) | v3 (DGX-only) | comparable? |
| --- | --- | --- | --- |
| transcription | mixed: openai whisper-1 (110) + DGX whisper (99) | DGX faster-whisper large-v3 | partly |
| diarization | Deepgram (cloud) | pyannote v4 community-1 | yes |
| speaker detection | gemini (LLM) | **qwen3.5:35b (LLM)** | yes |
| summary / GI / KG | gemini-2.5-flash-lite | qwen3.5:35b | yes |
| preprocessing | silence removal ON (drifted) | timeline-preserving | n/a — v3 strictly correct |

**Design trap caught before it cost us the experiment:** v3 originally used spaCy for speaker
detection, i.e. an LLM (gemini) vs *no LLM at all*. A worse v3 result would have been unreadable —
"is the local LLM weaker, or did no LLM run?". Every LLM stage in v3 now runs on the same pinned
model. Cost of catching it at 18 min instead of 15 h: one restart.

## 3. Bugs found on the way (these stand regardless of who wins)

Each was surfaced by running the **real pipeline** on the **real corpus** — none was visible in the
fixtures or unit tests.

| # | bug | evidence | fix |
| --- | --- | --- | --- |
| 1 | **Silence removal broke the transcript↔audio timeline.** `silenceremove` deleted every interior pause; we transcribed the shortened audio but stored timestamps against the original. | r01's stored transcript ends at 1438.8 s; silence-stripped audio is 1438.9 s; original is 1470.9 s. Corpus-wide: **3.05%** timeline error. A/B on the same episode: median drift **−17.6 s → +0.00 s**. | `13a4a87d` — preprocessing is timeline-preserving; silence removal opt-in, off |
| 2 | **The ad narrator was crowned host.** `resolve_speaker_roster` accepts `ad_intervals`; `pipeline.py`, its only caller, never passed any — the guard was dead code in prod. | On a Barclays pre-roll episode, host "Katie Martin" landed on a voice with **one turn** (the ad). Fixtures never caught it — they have no ads. | `ef7134a6` — char→time bridge feeds real ad intervals to the roster |
| 3 | **The transcript cache replayed the very bug we were fixing.** Key = original audio + provider + model, but the transcriber only ever sees the *preprocessed* audio. | The first reprocess "succeeded" for all 99 episodes **in 60 seconds** — every episode a cache hit, handing back the drifted transcripts. A green run that did nothing. | `4c1820e5` — key includes a preprocessing fingerprint |

Bug 3 is the most dangerous class: **a fix that appears to work and silently doesn't.** Verified
post-fix — the same audio now has two cache entries, and the June (drifted, no-fingerprint) one is
unreachable.

## 4. Baseline: v2 (cloud), deterministic metrics

From `scripts/eval/compare_corpora_v1.py`, 90–99 episodes:

| metric | v2 | reading |
| --- | ---: | --- |
| `timeline_error_pct` | **3.05** | ~110 s of accumulated error by the end of an hour-long episode |
| `voices` / `voices_named` | 6.43 / **2.37** | only 37% of diarized voices get a real name |
| `insights` / `quotes` per ep | 11.99 / 23.24 | |
| `summary_chars` | 1422 | |
| `kg_person_placeholders` | 0.0 | clean — no `SPEAKER_NN` leaking into entities |
| `quote_ts_valid_pct` | 99.63 | **misleading — see below** |

**`quote_ts_valid_pct` cannot see drift.** It only asks whether a timestamp lands inside
`[0, duration]`; a quote drifted by a minute still does. It scored 99.6% on the corpus whose
timestamps were *broken*. Only `timeline_error_pct` catches drift. Recorded here so nobody later
cites 99.6% as evidence of correctness.

## 5. Live observations — the DGX under load

| observation | value | note |
| --- | --- | --- |
| transcription | **~7.3–8× realtime** | 3801 s audio → 522 s |
| **v3 timeline error (first episode)** | **0.01%** | audio 3894.2 s, transcript 3893.9 s — vs v2's 3.05% |
| summarization probe (`qwen3.5:35b`) | **34 s** for a one-sentence prompt | slow; `bundled` mode sends the whole transcript → likely minutes/episode |
| GPU | 79–91% util, 3 models resident | whisper + pyannote + Ollama (28 GB) all on one GB10 |
| stage parallelism | transcription **serial** (single-flight, #876) | concurrent requests just queue server-side |

**Bottleneck:** one GPU hosting three models, stages serial. Est. ~14 h for 99 episodes — almost
entirely ASR. The cloud would fan all 99 out at once. This asymmetry is the real cost of going
local and must be priced in the verdict.

### 5a. Transcription speed: cloud vs DGX (measured, not assumed)

v2 recorded `transcribe_time` per episode, so this is from the corpus itself (n = 189):

| provider | n | median speed |
| --- | ---: | ---: |
| openai whisper-1 (cloud) | 99 | **25.2× realtime** |
| tailnet_dgx_whisper (v2) | 90 | **6.0× realtime** |
| DGX faster-whisper large-v3 (v3, live) | — | **~7.8× realtime** |

Transcription dominates wall-clock in **both** stacks — the LLM stages are noise beside it
(3–11 s vs ~510 s per episode). So "ASR is the bottleneck" is not a DGX-specific problem.

But the cloud was **~3–4× faster per episode**, *and* the API parallelises across episodes while
the DGX serializes. So:

- "transcription is the bottleneck stage" → true in both, nothing new
- "therefore the DGX costs nothing on throughput" → **false**: ~4× slower per episode, and serial

Caveat in the DGX's favour: it runs `large-v3`, a heavier model than the cloud's `whisper-1`. Part
of the slowdown buys a bigger model — whether it buys a *better transcript* is what the quality
comparison must answer.

### 5b. `bundled` LLM mode is unreliable on qwen3.5:35b

Repeated on real transcripts:

```text
[3] Bundled clean+summary failed, falling back to staged: Bundled JSON missing non-empty summary
[8] Bundled clean+summary failed, falling back to staged: ...
```

The model does not reliably emit the bundled clean+summary JSON, and the pipeline degrades to
`staged` — which works, so summaries are still produced. But it contradicts the comment in
`local_dgx_balanced.yaml` ("35b handles bundled mode cleanly"): on real prod transcripts it does
not. This is a genuine **structured-output reliability gap** for the local model, and precisely
the sort of thing a quality comparison exists to surface. The graceful `staged` fallback is what
keeps it from being an outage.

Open: pin `llm_pipeline_mode: staged` for the local profile rather than paying for a bundled
attempt that usually fails.

## 6. Evaluation design

Two arms, because a naive head-to-head is confounded:

| arm | compares | answers | cost |
| --- | --- | --- | ---: |
| **A — as-shipped** | v2 corpus vs v3 corpus | "Is the v3 corpus a better artifact?" | free |
| **B — controlled** | gemini vs qwen3.5:35b **on the same v3 transcripts** | "Is the local LLM as good as the cloud LLM?" | ~$0.50 |

Arm A measures `transcript × LLM` **together** — v3 reads a better transcript, so a v3 win there
does not prove the local model is good. **Arm B is the one that adjudicates the DGX question.**

**Judges:** `claude-sonnet-4-6` + `gpt-5.4` (the pair that decided #928). Scalar, not pairwise
(pairwise over-indexes on style). Candidates are Gemini (Google) and Qwen (Alibaba), so **Qwen and
Gemma judges are disqualified** — same-vendor judging hands a candidate a free style boost (#939:
a Qwen judge crowned a Qwen candidate rank 1 while cloud judges put it at 8). This rule is
**enforced in code** (`_assert_cross_vendor` refuses to run), not left to discipline.

Each summary is judged **against its own transcript** — grading v3's summary against v2's text
would punish it for correctly reporting what it saw.

## 7. Method: pilot before commitment

Operator call: **stop after feed 1 (10 episodes) completes fully**, evaluate, and only commit the
remaining ~89 episodes (15–20 h of GPU) if the pilot shows the local stack is competitive. The
run is stopped at a *feed* boundary, not mid-feed — a mid-feed kill would leave episodes without
summaries and nothing for the judges to grade.

## 8. Open questions

- [ ] Does `qwen3.5:35b` match `gemini-2.5-flash-lite` on summary faithfulness/coverage? (arm B)
- [ ] Does pyannote v4 name more voices than Deepgram did? (v2 named only 37%)
- [ ] Summarization wall-cost per episode in `bundled` mode — is it the real bottleneck, not ASR?
- [ ] Is the DGX case better made on *eval sweeps* than on pipeline runs? (the sweeps are where
      cloud spend actually lands)

## 9. Running log

- **11:01** first reprocess launched → aborted: 99/99 transcript-cache hits (bug 3).
- **11:14** relaunched after cache fix + LLM-parity fix. 0 transcript-cache hits.
- **11:28** Ollama summarization confirmed live in-pipeline; first metadata/GI artifacts written.
- **11:30** first v3 episode: timeline error **0.01%** (v2: 3.05%). #1173 confirmed end-to-end on
  real prod audio through the real pipeline.
- **11:45** measured, against the operator's prior that "ASR was the bottleneck on cloud too, so
  nothing new": half right. ASR dominates in both stacks, but the cloud did it at **25.2x**
  realtime vs the DGX's **6-8x** — ~4x slower per episode, and serial where the API fans out.
  Throughput is a real cost of going local, not a wash.
- **11:45** `bundled` clean+summary fails on qwen3.5:35b (invalid JSON) and falls back to
  `staged`. Summaries still produced; the profile comment claiming 35b handles bundled cleanly
  is wrong on real transcripts.
