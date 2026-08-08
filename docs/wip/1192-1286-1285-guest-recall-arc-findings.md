# Guest-recall arc — measured findings (#1192 / #1286 / #1285)

**Session:** 2026-07-31 (naming arc, feat/naming-arc-and-corpus-prep)
**Method:** deterministic relabel replay of `resolve_speaker_roster` over the prod-v2 corpus
(90 diarized episodes, `.test_outputs/manual/prod-v2/corpus`, `--llm none`, no GPU). Harness:
`scripts/backfill/relabel_corpus.py` + a per-voice roster-capture wrapper (roles/names/talk-time).

This doc records a **validated negative** for #1192's text lever and scopes #1286/#1285 with the
evidence the measurement surfaced. It is the honest counterpart to the #1228 revert: another recall
lever measured on real data and declined.

---

## TL;DR

| Issue | What was tested | Result |
|---|---|---|
| **#1192** | Augment `detected_guests` with transcript-intro `is_introduced_guest` hits (revive the dormant intro-NER path) | **0 new guest voices, 3 name-flip regressions → do NOT ship.** Text recall is exhausted. |
| **#1286** | (scoping) voice-embedding attribution for un-introduced panels | The **only** lever that reaches the ~113-voice tail. Needs GPU/DGX; not this session. |
| **#1285** | (scoping) canonicalize ASR-garbled names in the body | Confirmed real by the 3 flips below; heaviest, do last. |

---

## Two structural facts that reframed #1192 (both verified)

1. **The `detect_speaker_names` transcript-intro path is DORMANT in production.** Its intro-NER
   guest mining (`_merge_intro_guests`, gated by `is_introduced_guest`) reads
   `transcript_text[:INTRO_SNIPPET_LENGTH]` (`detection.py:79`), but the only production caller
   (`ml_provider.py:1054-1062`) never passes `transcript_text` — it runs at metadata time, before a
   transcript exists. So `INTRO_SNIPPET_LENGTH = 3000` is dead in prod; extending the window (the
   originally-proposed "Lever 2") changes nothing. Only unit/integration tests exercise it.

2. **The roster already mines on-air introductions — and it is live in both prod and relabel.**
   `resolve_speaker_roster` runs `guests_introduced_by_the_host` (`hosts.py:726`: cue-first +
   name-first + greeted regex families) plus `voice_intro` self-intro naming (`roster.py:1802`).
   The dormant intro path's `is_introduced_guest` (`guests.py:71`) uses only leading-cue patterns —
   a **subset** of what the roster already does. So the baseline already contains the on-air-intro
   recall; the intro-NER lever's only structurally-additive slice is reading *non-host* voices for
   introductions, which was previously **removed** as the N2 regression (a wrong name harvested onto
   an unrelated voice — `roster.py:1937-1941`).

## The A/B measurement (#1192)

- **baseline** — guest pool = `corroborate_guests(description NER)`; roster internally runs
  `guests_introduced_by_the_host` + `voice_intro` (the real relabel behaviour, `voice_texts` passed).
- **+lever** — same, plus `is_introduced_guest` hits over `transcript[:3000]` added to
  `detected_guests`.
- **Metric that counts:** Δ named **guest** voices (`voice_type=="guest" and named`). Hosts are
  already covered.
- **Precision proxy (no human GT):** a newly-named guest whose name appears nowhere on-air (not in
  its own turns, not adjacent to a cue in a host's turns) = mention-only = likely wrong.

**Result over 90 episodes (with `voice_texts` active, matching relabel/prod):**
```
extra guest-pool names added by lever: 10
NEWLY-named guest voices:               0
name-flip regressions on named voices:  3
  Alex Karnal      -> Alex Carnell     (NVIDIA — "Alex Karnal - The Trillion…")
  Nicolas Cerisier -> Nicolas Serissier (NVIDIA — "How Dassault Systèmes…")
  Nick Allardice   -> Nick Allardyce   (NVIDIA — "Accelerating Disaster R…")
```
(An earlier run showed +1 new voice, but that run had a harness bug — `voice_texts=None` disabled the
roster's self-intro path. With it fixed to match relabel, the "+1" is already named at baseline.)

**Verdict:** 0 recall, 3 churn flips. The lever names nothing the roster didn't already name, and it
destabilises already-correct names. Per #876 ("a wrong name is worse than no name") and the advisor's
ship gate (Δ named guest > 0 **with** zero flips), the lever fails. **Not shipped.**

## NOT covered / open (equal weight)

- **The ~113 truly-unknown tail is untouched here.** They are un-introduced multi-guest panelists —
  never named on-air — so NO text lever (this one or any) can reach them. This is not a gap in the
  measurement; it is the measurement's finding. Reaching them requires **voice-embedding attribution
  (#1286)**.
- **Precision proxy is a proxy, not human GT.** Zero mention-only flags ≠ proven precision; it means
  no *automatically-detectable* wrong name. The #1189 human-GT fixtures remain the real gate.
- **Dead code not removed.** `_merge_intro_guests` / `INTRO_SNIPPET_LENGTH` are production-dead but
  unit-tested; deleting them is a separate, careful cleanup (not done — would drop tested code).
- **`--llm none`.** This measures the deterministic path only. The Gemini (ADR-110) resolution channel
  is unchanged by any of this and was not exercised.

## #1286 — WORKED on the DGX; cross-episode attribution empirically NOT viable (2026-08-01)

Ran it, did not defer. Extracted **real voice embeddings for all 90 episodes** on the DGX GPU
(`dgx-llm-1`, `podcast-pyannote:0.2.0-community1`, `pipe._embedding` = WeSpeaker ResNet34-LM 256-d),
two ways: 90s per-speaker concatenation, and the advisor's recommended windowed centroids (3s windows
inside turns, batch-embed, L2-mean, outlier-reject). Then AS-Norm score normalization + a
precision-gated attribution harness (trusted refs only, negative-cohort p99.5 threshold, margin gate).

**Result: cross-episode attribution does not reach precision on this corpus.**

| metric | 90s-concat | windowed |
|---|---|---|
| split-half self-cos (same voice, same ep) | 0.71–0.88 (good) | (method sound) |
| within-ep DIFFERENT-named-person cos | mean 0.66, p90 0.95 | mean 0.53, p90 0.90 |
| cross-ep SAME-name cos | mean 0.40, p50 0.34 | mean 0.31, p50 0.25 |
| held-out attribution precision | 13% | — |
| AS-Norm recall @ p99.5-neg THR | 0% | 0% |

The embedding *method* is sound (split-half self-cos ~0.8, batched==single call verified). Two things
kill cross-episode attribution: **(1) channel domination** — the same person across two recordings
(~0.3) scores LOWER than two different people in one recording (~0.5), and AS-Norm did not separate
them; **(2) label noise** — the corpus-transcript diarization over-segments and mislabels (advisor's
Cause C: e041 = 13 speakers in the transcript vs 22 in `segments_v4`), so the "different-named-person"
pairs at p90≈0.9 are frequently the SAME voice split into a named + `SPEAKER_NN` cluster, and some
"same-name" cross-ep pairs at p10≈0.06 are the roster painting one name on two different voices.
Both the clustering INPUT and the reference LABELS are too noisy to validate against.

**Intra-episode merge on the CLEAN `segments_v4` (community-1) diarization — also run, also 0.**
Extracted windowed embeddings on the 599 `segments_v4` clusters (90 eps), named each by max time-
overlap with a corpus-transcript named voice, then tried to name each of the 253 unnamed clusters
(≥8s) from a same-episode NAMED cluster by embedding cos. Result: **0 newly-named at cos 0.72–0.78,
and 0 different-name collisions.** The zero is informative, not a failure: on the clean diarization
there are no over-segmentation *fragments* to merge (community-1 already merged them — the 166
"matches" seen on the OLD corpus-transcript diarization were that diarization's over-segmentation
artifact). The 253 unnamed clean clusters are genuinely DISTINCT un-introduced voices with no
same-episode named reference. The 0-collision precision guard shows the embeddings ARE discriminative
within a recording — the un-introduced voices simply have nothing to match to.

**Honest conclusion (worked both ways, not a deferral):** neither cross-episode attribution (channel
domination, AS-Norm 0% recall) nor intra-episode over-seg merge (0 on the clean diarization) names the
un-introduced panel tail with community-1 embeddings. Reaching it needs a fundamentally different
signal — a fine-tuned / ECAPA cross-session speaker-verification embedding, or a non-voice lever — a
multi-session R&D effort, not a same-session fix. Scripts + all embeddings are on the DGX
(`~/embed-1286/`: `embeddings.json` 90s-concat, `embeddings_win.json` windowed, `embeddings_v4.json`
clean-diarization) for a follow-up.

## #1285 — CANDIDATE GENERATOR delivered; automatic corpus canonicalization is NOT precision-safe

Built `speaker_detectors/name_canonicalization.py` (`canonicalize_text`) + 8 unit tests (green). It
surfaces **31 candidate rewrites** across the 90-ep corpus with their source sentences — Kevin
Russo/Roos→Roose, Ryan Knudsen/Knudson→Knutson, Farnaz Fasihi→Fassihi, Sebastian Malaby→Mallaby,
Greg Rosalski→Rosalsky, etc. — most of which are genuine garbles of the SPEAKER (self-intros
"I'm Kevin Russo, tech columnist at the NYT" → Kevin Roose).

**It was briefly wired into `relabel_corpus.py` and REVERTED** after a review found the gate is not
precision-safe for unattended corpus writes (#876). Two confirmed failure modes the gate does NOT catch:
- **Distinct-person collision** — the collision guard only abstains when two *speaking* voices share a
  first name. It does nothing when the episode has one speaker + a *mention* of a different real person
  with the same first name and near surname: "Kevin Rose" (Digg) → "Kevin Roose" (NYT), "Eric Schmidt"
  (Google) → "Eric Schmitt" (reporter) — both edit-1/same-Soundex — would be silently corrupted.
- **Reverse garble** — the ROSTER name can be the garble while the body is correct: "Duncan Macmillan"
  (correct body) → "Duncan Macmillans" (roster's trailing-s ASR error). Confirmed in the candidate list.

The "32 fixes, 0 collisions" earlier framing was NOT precision validation — "0 collisions" only means
no episode had two speaking same-first-name people; none of the 31 were checked against their source
sentence. On inspection they are mostly correct, but the gate got lucky on this corpus, not safe.

**Honest status:** delivered as a review-assist candidate generator (module + tests, not wired to any
corpus-writing path). Safe automatic use needs a stronger gate — abstain when the garbled form is a
plausible distinct real name, and pick the more-canonical of roster-vs-body — not yet built.
