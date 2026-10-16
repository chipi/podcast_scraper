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

## How common it is (evening, 2026-10-15; #2359 step 1)

**Served length vs transcript end.** 402 prod episodes, up to 15 per audio host (30 hosts; 13
more unreadable), `ffprobe` on the stream as served now. 64 (16%) run more than 20 s past their
transcript's end, all on ad-inserting hosts: BBC 10/15 (median +73.5 s), podtrac 9/15 and 4/15, acast
8/15 (+20 s), cohst 8/15 (+51 s), pdst.fm 6/15. Plain-file hosts show none (substack, libsyn,
buzzsprout, simplecast, blubrry). Some hosts serve SHORTER than the transcript (swap.fm, median
−25 s), which only a changed ad load explains.

**Length alone does not prove drift**: a fixed outro or end ad the transcript never covered makes
the stream longer without moving any clip. So 10 of those episodes were listened to with the clip
checker (`--exact`, 48 clips):

| Host | Episode | Clips on target | Where the quote really is |
|---|---|---|---|
| acast | 2 episodes | 4/4, 2/2 | 0 to +7 s: a fixed tail, no drift |
| swap.fm | 2 episodes | 5/5, 2/4 | on target, one clip −16 s |
| cohst | 2 episodes | 2/5, 2/5 | the rest not found in the audio: inconclusive |
| podtrac | "Is starting a family becoming impossible" | 0/10 | +9 s, then +47 to +52 s after a mid-roll |
| BBC | "Archive: Plato's Republic" | 0/8 | a constant +8.7 s (a changed opening) |
| BBC | "Furnishing with Fungi" | 0/5 | −11 s growing to −23 s |

Drift is real and per episode: on podtrac and BBC every clip of these episodes plays the wrong
moment; acast's extra length is a harmless tail. A fixed per-host correction cannot work (BBC
alone moved +8.7 s on one episode and −23 s on another).

### Not covered

* 10 episodes listened to, on 5 hosts. The share of prod clips that play the wrong moment is not
  measured; 64/402 is an upper bound for "length changed", not a count of drifted clips.
* The stream was fetched once, from one place, with one user agent; ad loads can differ by region,
  device and time.
* cohst's clips that were not found could be drift or a checker miss.

## Re-timing drifted episodes (#2359 option A, 2026-10-16)

`scripts/ops/audio_retime.py` listens to the stream as served now and maps transcript time to it:

- It takes a 20 s sample every 2 minutes and transcribes it with Whisper `base.en` on the Mac. The DGX is not used.
- Every run of three words heard that also occurs in the transcript votes for an offset. The biggest group within 1.5 s wins, and it needs 3 votes.
- Where neighbouring samples disagree, a binary search finds where the talk resumes. A sample that matches nothing is inside the ad.
- It writes `<corpus>/audio_time_maps.json`.
- The api sends the map with the episode and sends each reel clip's `play_start_ms` / `play_end_ms`. The player converts with the map.

Run on the 2 episodes `audio_drift_check.py` flagged in the copied prod set (1 podtrac, 1 BBC). The ad loads had changed since the 4 flagged the evening before.

| Episode | Map |
|---|---|
| podtrac "Is starting a family…" | +7.9 s; −75.0 s from 1187.5 s (an old ad gone); +34.2 s from 2564.7 s (an ad added) |
| BBC "Plato's Republic" | +12.7 s; +46.9 s from 2329.9 s |

Both maps took 65 s together, downloads included.

Samples on the same side of a change agree to within 0.1 s.

`clip_alignment_check.py --exact` over every Moment of the two episodes (18 clips):

- **Without the map:** 0 play what they say; median offset 13.3 s; 6 not located.
- **With `--mapped`:** 16 play what they say; median offset 0.6 s; 1 not located.

The 2 left:
- **Podcast clip at transcript 3297 s:** +4.7 s. Its unmapped offset (+38.9 s) disagrees with the samples on both sides (34.2 s at stream 3245 s and 3367 s). Not explained.
- **Plato clip at 1717 s:** heard 0.94, but its start was not located.

### Not covered / not verified

- Only 2 episodes, both English. The tool uses Whisper `base.en`, so a non-English episode is not handled.
- Not run on prod, and no schedule. The map is only as current as its last run, and a host can change its ads again.
- **Downloads:** a downloaded episode keeps the audio, with its ads, from the day it was downloaded. The map describes the stream on the day it was measured.
- **Timecodes outside the player:** search hits, highlights and episode notes show transcript time, while the player clock shows stream time. A listener can see two different times for one moment.
- **Clips with an ad change inside them:**
  - Moments, short listens and board reels leave the clip out.
  - A frozen Learn day keeps it: it starts at the right place but plays across the change.
- **Episodes that drifted but have no map** still leave short listens and Learn. Once mapped, they come back.
- The in-player sync nudge (hidden) still adds on top of the map.
