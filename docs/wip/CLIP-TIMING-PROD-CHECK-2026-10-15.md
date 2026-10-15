# Do clips play what their card says? — prod check, 2026-10-15

Operator report: in short listens "the sound that comes out is not what's in the quote".

> **Corrected the same day.** The first version of this note said nine shows played their clips
> "~2.5 s late". That was a bug in the checker, not in prod: it matched a quote's words with a
> sliding window that let a match start a few words early, so every offset read about −2.5 s
> (found against the fixture's RTTM ground truth, where segment starts are exact). Fixed in
> `clip_alignment_check.py` (match the quote's words in order, anchored on its first word, and
> measure the quote, not the whole segment). Everything below is from the corrected checker.

## How it was measured

`scripts/eval/clip_alignment_check.py`: for each episode's Moments (the server's own picker and
inputs; short listens and learning days reuse the same clip windows) it cuts the clip's audio from
the episode's own `media_url`, transcribes it with Whisper `base.en`, and reports

* **heard** — the share of the quote's content words the clip actually says;
* **offset** — where the quote's first words really are, relative to the clip's start. A clip starts
  at the beginning of the transcript segment holding its quote, up to 6 s early
  (`clip_lead_max_seconds`), so **0 to ~6 s is by design** — the sentence's lead-in.

Sample: 10 English episodes, 10 shows, one per feed, picked at random from prod (read-only copy of
their metadata / GI / transcript files over the deploy key). 73 clips. "Plays its quote" below =
heard ≥ 0.6 and the quote starts between 2 s before and 8 s after the clip.

## Results (afternoon run)

| Show | Audio host | Plays its quote | Heard (median) | Offset (median) |
|---|---|---|---|---|
| The Pragmatic Engineer | api.substack.com | 14 / 15 | 0.96 | +3.7 s |
| The a16z Show | mgln.ai | 8 / 9 | 0.96 | +1.2 s |
| Round Table China | dts.podtrac.com | 4 / 6 | 0.97 | +1.7 s |
| Huberman Lab | traffic.megaphone.fm | 7 / 7 | 1.00 | +2.2 s |
| NVIDIA AI Podcast | cohst.app | 2 / 2 | 0.97 | +6.2 s |
| The Long Run with Luke Timmerman | feeds.soundcloud.com | 11 / 12 | 1.00 | +0.3 s |
| Eco-Business Podcast | feeds.soundcloud.com | 5 / 5 | 1.00 | +2.1 s |
| Hard Fork | dts.podtrac.com | 0 / 1 | 0.55 | −0.2 s |
| **Past Present Future** | **pscrb.fm** | **0 / 4** | **0.07** | **+28.5 s** |
| **The Rest Is Politics: Leading** | **pdst.fm** | **0 / 12** | **0.12** | **mostly not found within ±40 s** |

**51 of 73 clips play their quote.** Seven shows are right. Two are badly off, and the cause is
in the audio, not the timings we hold:

## The cause: the audio a listener gets changes over time

* **Past Present Future measured fine this morning** (heard ~0.9, offset ~−1.6 s on the old
  checker) **and 28 s off this afternoon** — same transcript, same code; only the download
  differed. Its feed is served through `pscrb.fm` (an ad-insertion prefix).
* Three requests a few minutes apart returned the same length for every episode, so it is not
  random per request: the ad load changes over hours (campaigns rotate).
* Length served now vs where our transcript ends (0–11 s is normal trailing music or silence):

  | Episode | Host | Served now − transcript end |
  |---|---|---|
  | Hard Fork (The Ezra Klein Show: Jensen Huang) | dts.podtrac.com | **+111.4 s** |
  | Past Present Future | pscrb.fm | **+60.7 s** |
  | The Rest Is Politics: Leading | pdst.fm | **+15.3 s** |
  | the other seven | — | 0.1 – 11.0 s |

  The offset also grows through an episode (27.0 → 29.3 s on Past Present Future), so mid-roll
  ads change too, not only the pre-roll. A single fixed shift cannot correct it.

## What this means

With dynamically inserted ads, **no fixed timestamp is right for every listener at every time**.
Our times are right for the audio we transcribed; a listener streaming the same episode later can
get a different ad load, and every clip after a changed ad slot plays other words.

## Not covered / not verified

* 10 episodes of the corpus. How many prod episodes are served through ad-insertion hosts, and how
  often their ad load changes, is not measured.
* That the extra length is ads (and where in the episode it sits) is inferred from length and
  offset, not confirmed by listening to what was inserted.
* Whether our stored metadata records the length of the audio we actually transcribed (the
  feed-declared `duration_seconds` is the publisher's number, not the file's) — not checked.
* Hard Fork's single clip (heard 0.55) is ambiguous: partly heard, quote at the clip start.
* Only Moments clip windows were measured; an insight's "Play from" in the Brief uses the same
  quote times and is exposed to the same shift, but was not measured.
* Non-English episodes were left out (Whisper `base.en`).
