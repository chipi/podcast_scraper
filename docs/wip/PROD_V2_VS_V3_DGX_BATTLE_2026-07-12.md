# prod-v2 (cloud) vs prod-v3 (DGX-only) — running observations

**Status:** live · started 2026-07-12 · corpus build in progress
**Question:** can the fully-local DGX stack match the cloud one — and does the DGX investment pay?

Running notes, to be assessed later together with `data/eval`. Not authoritative; findings that
survive get promoted (eval report → `docs/guides/eval-reports/`, decisions → ADR).

---

## 1. The question, stated honestly

> **CORRECTION (2026-07-12).** This section first claimed the cloud built the whole corpus for
> **$0.51**, and concluded the DGX could never pay itself back. **That was wrong by ~54x.** The
> $0.51 is only the *LLM* stages. It omitted **transcription** — which the v2 manifest reports as
> `llm_transcription_cost_usd: 0.0` *because v2 already transcribed on the DGX*, not because ASR is
> free. The original claim rested on the one stage that barely costs anything. Corrected below.

The v2 corpus manifest records the LLM spend:

| stage | cloud cost |
| --- | ---: |
| GI insights | $0.333 |
| summarization | $0.117 |
| speaker detection | $0.064 |
| KG extraction | $0.000 |
| LLM subtotal | $0.514 |

But a real all-cloud rebuild also pays for ASR, and that dominates:

| component | cost of one full rebuild |
| --- | ---: |
| transcription — 75.2 h of audio @ whisper-1 $0.006/min | **$27.08** |
| all LLM stages | $0.51 |
| **true cloud cost** | **~$27.59** |
| **DGX** | **$0 marginal** |

**98% of the cloud bill is the exact stage the DGX does for free.** At ~$3.5k for the box the
break-even is ~**127 rebuilds** — reachable, since every pipeline change invites another (we
rebuilt this corpus several times in a single day).

The case for the box therefore rests on:

- **quality parity** — can local match cloud at all? (if not, none of the rest matters)
- **rebuild economics** — ~$27.59 a rebuild, and reprocessing is routine
- **unmetered experimentation** — eval sweeps and judge matrices are where cloud spend really
  compounds; they run free here
- **perf-per-dollar** — ~4x slower than an API fleet on H100/H200-class silicon costing 10-100x
  more is a strong result, not a weakness
- **privacy** — audio and transcripts never leave the house

Operator decision (2026-07-12): score **quality parity** first. The rest is moot if local loses.

### 1a. Scaling to the real target: 500 → 1 000 → 10 000 episodes

Operator's aspiration is **10 000 episodes** once the pipeline is solid, via 500 → 1 000 → 10 000.
On measured numbers (45.1 min/episode from this corpus; whisper-1 at $0.006/min):

| episodes | cloud $ **per rebuild** | DGX (large-v3) | audio storage |
| ---: | ---: | ---: | ---: |
| 100 (today) | $28 | 0.4 d | 4.9 GB |
| 500 | $138 | 2.0 d | 24 GB |
| 1 000 | $276 | 4.0 d | 49 GB |
| **10 000** | **$2 759** | **40 d** | **488 GB** |

**At 10 000 episodes the box pays for itself in ~1.3 rebuilds** (one cloud build = $2 759 vs a
one-off ~$3.5k), and every rebuild after that is free. Since reprocessing follows every pipeline
change, the economics stop being arguable at this scale.

Two walls arrive before 10 000, and both are cheaper to fix now:

**Wall 1 — time, not money.** 40 days of serial ASR is not an iteration loop. The lever is the ASR
model, not the hardware: `large-v3-turbo` is ~5x faster at near-equal WER → ~8 days for 10 000.
That one swap dominates every other optimisation available to us. Testable against the oracles we
already have (45-fixture RTTM, real-10). Folds naturally into #1174, which is already an ASR
alternatives evaluation.

**Wall 2 — storage, which is self-inflicted.** 488 GB of audio, against 60 GB free on the laptop
(94% used). But **we do not need to keep the audio**: our own policy is bridge-only / never rehost.
Audio is a transient input — download → transcribe → discard. The GUID-keyed audio cache (#947) is
an optimisation, not a requirement. Transcripts for 10 000 episodes are ~1 GB. The wall dissolves
if we stop hoarding media, and the corpus should live on the DGX rather than the laptop regardless.

**Pre-10k agenda:** (1) swap the ASR model — the only thing between us and a usable loop at scale;
(2) stop persisting audio, move the corpus off the laptop; (3) solidify the pipeline (today's work)
so a 10k run is worth starting.

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

## 7. Method: pilot before commitment — and then HOLD

**Decision (2026-07-12): the full v3 build is ON HOLD.** The remaining ~89 episodes are *not*
being processed. Reason: there is no point spending ~14 h of GPU building a corpus on
`large-v3` when the audio-side stack may be replaced within days — `large-v3-turbo` (#1178) and
MOSS (#1174/#1177) are both credible replacements for exactly that stage, and the 10k-episode
target makes the ASR choice load-bearing (40 days vs ~8).

So the **10-episode pilot becomes the common benchmark set**: every contender in the deathmatch
(#1179) runs the same 10 episodes, same audio, same downstream LLM. Only the ASR/diarization
stage varies. The full corpus gets built **once**, with the winner.

Order: finish the pilot → evaluate it → settle the audio stack (#1178, #1177, #1174, #1179) →
*then* build the corpus.

### Original pilot rationale

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

## 8a. The pilot verdict — and the bug it exposed

**The deterministic half of the local stack beat the cloud outright:**

| metric | v2 cloud | v3 DGX | |
| --- | ---: | ---: | --- |
| `timeline_error_pct` | 1.92 | **0.19** | the #1173 fix, on real audio |
| `voices_named` / `voices` | 2.67 / 8.11 (33%) | **6.11 / 7.44 (82%)** | pyannote v4 + local LLM name far more speakers |
| `kg_persons` | 0.0 | **8.44** | v2's KG had **no Person nodes at all** |

**The LLM half was never actually tested — we tested our prompts.** The summaries came back
scoring 1.92/5 (Sonnet) and 2.64/5 (GPT-5.4) against the cloud's 3.86/3.97, with **coverage at the
floor (1.00) on every episode**. The cause was not model weakness:

> An episode about **Tim Cook's retirement** was summarized as *"Speed gains come from braking
> earlier and smoother rather than taking bigger risks — a counterintuitive but reliable principle
> for riders at any level."*

That sentence is a **verbatim line from our own prompt's few-shot style examples**. The prompt
says in plain English that the examples are style references, "not from the current episode".
Gemini obeys; qwen3.5:35b copies them. It failed in two ways, and the second is the dangerous one:

- invalid JSON -> falls back to staged (loud, harmless)
- **VALID JSON containing the copied examples -> "succeeds" and writes fiction into the corpus**
  (silent, poisonous). Nothing flagged it. The judges caught it after the fact, for money.

Insights were **fine** (10/episode, on-topic and specific), and direct probes showed quote
extraction and entailment both work on qwen. So the LLM is capable — our prompts were the problem.

**Fix loop (each iteration surfaced a distinct bug):**

| iteration | result |
| --- | --- |
| 1 — pilot | 10/10 summaries poisoned; quotes -90% |
| 2 — `llm_pipeline_mode: staged` | poison 10/10 -> **2/10**; a SECOND source found: the *shared* bullets prompt |
| 3 — shared prompt hardened + runtime guard | in progress |

The examples live in **eight** prompt files, so hardening the text is necessary but not sufficient.
`_warn_if_prompt_examples_leaked` now inspects every generated summary and shouts
`SUMMARY POISONED` if the model copied the prompt. A green run that writes fabricated content is
worse than a crash.

**Two accelerants found along the way:**
- The transcript cache (now correctly keyed) means a re-run **skips ASR entirely** — 10/10 cache
  hits. That turns the iteration loop from ~2 h into ~25 min, since ASR is 98% of the wall-clock.
- Probing a provider directly (one transcript, no pipeline) validates a prompt fix in **90
  seconds** instead of 13 minutes.

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

## 10. Two findings that invalidate earlier readings (evening)

### 10.1 The "insight yield gap" was ours, not qwen's

The DGX corpus showed 10 insights/episode against v2's 12, and the obvious reading was that the
local model is a weaker idea-generator. It is not. Every episode landed on **exactly** 10, on 17 of
18 episodes — a perfectly integral average is a ceiling, not a model deciding how many ideas an
episode holds.

All 26 profiles set `gi_max_insights: 12`. Config validated it, the pipeline passed it. Then every
one of the 7 LLM providers did `max_insights = min(max(1, max_insights), 10)` and rendered
"Extract 10 key takeaways" into the prompt. The knob has been dead repo-wide since 2026-03-29
(359691d3). **Qwen followed instructions perfectly; it was never asked for 12.**

Fixed in f16d1f30. The token budget was capped the same way (`min(1024, max_insights * 150)`);
lifting the clamp alone would have truncated insights 11-12 away and made the fix look like a
no-op, so both moved together.

### 10.2 "Insight density is front-loaded" was a magic number

We had noted that insights cluster in the first two-thirds of an episode and treated it as a
property of podcast speech. It was `[:50000]`.

Six providers (gemini, anthropic, deepseek, grok, mistral, openai) truncated the transcript to a
hardcoded 50,000 chars before quote extraction. Episodes here run 67k-117k chars, so 26-36% of
every long episode was invisible. 68dcd0a2 had fixed this in **ollama only**.

The v2 corpus proves it. Of 1418 grounded quotes located across 60 long episodes, **4** fall
beyond 50,000 chars. Deepest quote per episode: 48,612 / 48,665 / 48,903 / 48,934 / 49,152 —
walking up to the cut and stopping. The final 20% of every episode contains **zero** quotes.

| quote position (% of episode) | quotes |
|---|---|
| 0-10% | 502 |
| 10-20% | 398 |
| 20-30% | 267 |
| 30-40% | 152 |
| 40-50% | 48 |
| 50-60% | 32 |
| 60-70% | 10 (the 50k cut lands here) |
| 70-80% | 9 |
| 80-90% | **0** |
| 90-100% | **0** |

Fixed in 848448bd; budget is now a measured constant (150k chars: clears the longest episode seen,
116,596, and fits DeepSeek's 64k-token context).

**Consequence for the comparison:** gemini reached 91.3% grounding while blind to a third of every
episode. The cloud baseline is *stronger* than we credited, not weaker.

### 10.3 prod-v2 is not a fair baseline

Chasing 10.1 surfaced that v2 was built by materially different code:

- its 12 insights came through a path the (then-live) clamp did not gate;
- its two runs used **different evidence paths** — the June run records
  `gi_evidence_extract_quotes_calls = 0` with 346 NLI candidates, the May run 148 calls;
- its GI artifacts carry `Topic: 10` and no `Person`/`Organization` nodes; v3's carry `Topic: 3`
  plus both.

The funnel table we built therefore attributed to "gemini vs qwen" a delta that is substantially
**old code vs new code**, and it summed two incompatible gemini pipelines into one column. Before
any v2-vs-v3 number is trusted, a cloud LLM needs a run through *current* code on the same
episodes.

### 10.4 The entailment threshold is a dead lever

Calibration at 0.6 returned 95.0% recall / 80.0% rejection — **identical to 0.5**. The graded
entailment prompt only ever emits {0.0, 0.3, 0.7, 1.0}, so every threshold in (0.3, 0.7] produces
the same partition. The knob is three-valued, not continuous:

- threshold <= 0.7 → accepts {0.7, 1.0} → 95% recall / 80% rejection
- threshold > 0.7  → accepts {1.0} only → 65% recall / 95% rejection  ← current (0.75)

Our strict setting has ~65% recall against gemini's ~66%, i.e. **our gate is about as permissive as
gemini's**. That points away from "the gate is too strict" and toward candidate supply. Tuning must
happen in the prompt (more graded levels), not the threshold.

### 10.5 Pattern, not coincidence

Four hardcoded-prompt/param defects in one day (ollama entailment, ollama extract_quote, the
max_insights+token clamp in all 7 providers, the 50k cut in 6). Every one produced plausible output
and reported success. The audit is not optional hygiene — inline prompts and magic literals in
provider code are where this project's silent failures live.
