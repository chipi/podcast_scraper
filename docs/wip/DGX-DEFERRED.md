# Deferred to the DGX — only after every text fix is done

Prod corpus fixes are text-file fixes (see `CLAUDE.md`). This list holds what text on disk
**cannot** fix, with the evidence. It is not a queue: the operator decides if and when any of it
runs, after the manual / text-only work is finished. Add an item only with evidence that the text
lacks what is needed.

| Item | Episodes | Why text cannot fix it (evidence) | Would need | Unverified | Issue |
|---|---|---|---|---|---|
| Two-host shows where both hosts' intros merge into one diarized voice (Hard Fork) | Hard Fork: neither host placed on 52 of 69 stored episodes (both on 9); deterministic replay names neither on 68 of 69 | The cold open ("I'm Kevin Roose… I'm Casey Newton…") is diarized into ONE voice, so the merged-cluster guard correctly refuses it; the text then cannot say which remaining voice is which. Vocatives were measured at 62.5% as a binder (#2078, "do not retry without new evidence"); a turn-order prior is a coin flip when the merged voice holds both openings | cross-episode voice identity from speaker embeddings (#2078 direction 1) — audio + GPU | whether the 9 correctly-named episodes share a pattern the others lack; how many other two-host feeds have the same merged-intro shape | #2078 |
| Value gate rates its own output ("SELF-GRADING") | every GI episode (warned once per run, 26 runs on 2026-10-02) | the gate is configured to `vllm` and the DGX serves one model (`Qwen3-30B-A3B-Instruct-2507-FP4`), so the rater is the generator; no text fix changes which model answers | a second, distinct rater model served on the DGX and the gate pointed at it (#1895) | whether a distinct rater changes which insights are dropped (the #1895 rater comparison has not been run) | #1895 |

## Removed

- **2026-10-27, publisher-transcript episodes with no speakers (#2100, was 51):** no longer
  needed. On prod's served corpus (`select_served_artifacts`, 2.7.30) Explaining Brazil 47/47,
  The Flip 42/42 and Korea Deconstructed 13/13 have speakers, all transcribed by Whisper and
  diarized (12 multi-voice + 1 solo lecture, one voice correct). The 0-speaker publisher versions
  remain only in superseded runs.
