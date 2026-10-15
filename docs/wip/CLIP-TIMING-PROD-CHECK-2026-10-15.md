# Do clips play what their card says? — prod check, 2026-10-15

Operator report: in short listens "the sound that comes out is not what's in the quote".

## How it was measured

`scripts/eval/clip_alignment_check.py` (committed 5ad0d422d, `--exact` in d77e5546d). For each
episode's Moments — the server's own `pick_moments` with the route's inputs and config; short
listens and learning days reuse the same clip windows — it:

1. cuts the clip's audio and transcribes it with Whisper `base.en` (cached locally);
2. reports **heard**: the share of the clip text's content words the audio actually says;
3. transcribes ±40 s around the clip with word timestamps and reports **offset**: where the clip
   text's opening words really are, relative to the clip's start (−2 = they start 2 s before the
   clip does, so the clip cuts in late).

Sample: 10 English episodes, 10 shows, one per feed, picked at random from prod (read-only tar of
their metadata, GI and transcript files over the deploy key; audio streamed from each episode's
own `media_url`). 73 clips. Run twice: once seeking the MP3 directly, once (`--exact`) decoding
the whole file first. Both give the same numbers, so seek error is not the cause.

## Results (`--exact` run)

| Show | Audio host | Clips | Heard (median) | Offset median | Offset range |
|---|---|---|---|---|---|
| The Pragmatic Engineer | api.substack.com | 15 | 0.93 | −2.3 s | −4.7 … −0.2 |
| Past Present Future | pscrb.fm | 4 | 1.00 | −1.6 s | −1.8 … −1.3 |
| The a16z Show | mgln.ai | 9 | 0.94 | −2.8 s | −5.6 … −1.4 |
| Round Table China | dts.podtrac.com | 6 | 0.95 | −2.1 s | −2.7 … −1.4 |
| Huberman Lab | traffic.megaphone.fm | 7 | 0.97 | −2.7 s | −4.2 … −1.9 |
| Hard Fork | dts.podtrac.com | 1 | 0.98 | −3.5 s | — |
| **The Rest Is Politics: Leading** | **pdst.fm** | **12** | **0.13** | **−32.7 s** | **−34.0 … −31.7** |
| NVIDIA AI Podcast | cohst.app | 2 | 0.94 | −4.3 s | −5.0 … −3.5 |
| The Long Run with Luke Timmerman | feeds.soundcloud.com | 12 | 0.96 | −2.7 s | −5.6 … −1.2 |
| Eco-Business Podcast | feeds.soundcloud.com | 5 | 1.00 | −2.4 s | −3.8 … −1.2 |

Overall: `73 clips · 20 play what they say (heard ≥ 0.6, |offset| ≤ 2 s) · median |offset| 2.6 s`.

## What this shows

1. **One show is 32–34 s off on every clip** (The Rest Is Politics: Leading). The listener hears
   other words than the card shows. The offset is the same from minute 8 to minute 54, so it is
   one fixed shift for the whole episode — the audio streamed today is ~33 s shorter before the
   speech than the audio the transcript was made from. A different-length pre-roll (ads inserted
   per download) fits that, but it is **not verified**: the cause needs the audio we transcribed
   compared with what streams now.
2. **The other nine shows play the right words but start ~2–3 s late**, on every host, including
   SoundCloud, which does not insert ads per download — so this one is not ad insertion. Where the
   lag comes from is **not determined**: the transcript's segment times, the clip window built from
   them (`build_clip`), or a bias in the measurement itself (the fixture's first clip measured
   +0.5 s, the opposite direction, which argues against a large measurement bias but does not
   rule it out).
3. **The fixture corpus is a separate, known fault**: its transcripts are on a made-up 6-second
   grid the TTS audio does not follow (2 of 24 clips play what they say there). That explains the
   dev phone, not prod.

## Not covered / not verified

- 10 episodes of ~80 shows; the 32 s case is one show in this sample. How many prod shows carry a
  fixed shift is unknown until the check runs over more of the corpus.
- The cause of the −33 s shift (pre-roll length) is a hypothesis; not checked against the
  transcribed audio.
- The cause of the −2.5 s lag is not found. Next evidence: for the same episode, compare a
  transcript segment's start time with Whisper's word time for that segment's first words —
  that separates "segment times are late" from "the clip window is late".
- Only Moments windows were measured. Insight "Play from" in the Brief uses the quote's start time,
  not the clip window; it is not measured here.
- Non-English episodes were left out (Whisper `base.en`).
