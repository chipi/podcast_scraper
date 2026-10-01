# Deferred to the DGX — only after every text fix is done

Prod corpus fixes are text-file fixes (see `CLAUDE.md`). This list holds what text on disk
**cannot** fix, with the evidence. It is not a queue: the operator decides if and when any of it
runs, after the manual / text-only work is finished. Add an item only with evidence that the text
lacks what is needed.

| Item | Episodes | Why text cannot fix it (evidence) | Would need | Unverified | Issue |
|---|---|---|---|---|---|
| Publisher-transcript episodes with no speakers | 51 (Explaining Brazil 40, The Flip 8, Korea Deconstructed 3) | `direct_download`; diarization never ran; the publisher originals (VTT/SRT) carry 0 voice tags and 0 `Name:` cues (re-fetched 2026-10-01); the deployed publisher path takes speakers only from voice tags | diarize the audio, align to the existing text (`rediarize_only`) | `rediarize_only` has never run on a real episode nor on a `direct_download` transcript; how many of the 51 have more than one voice | #2100 |
