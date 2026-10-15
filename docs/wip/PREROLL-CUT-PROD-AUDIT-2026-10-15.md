# Does the pre-roll cut take real content? — prod audit, 2026-10-15

Question: `detect_ad_cut_ranges` cuts the first ad block in the head window **from 0**
(`start = 0`), so the ad pitch before its first URL/CTA pattern goes too. Does that also cut
real episode talk on prod?

## How it was measured

Read-only (`scripts/eval/preroll_cut_audit.py`). 3508 prod `*.segments.json` (copied over the
deploy key earlier today, 79 feeds) were
run through `excise_ad_regions` on the segments' joined text, the same text `ad_time_intervals`
uses. For each episode whose first cut starts at 0:

* **lead** = talk wholly before the first ad pattern (the part cut only because of `start = 0`);
* the lead text and the matched pattern were read by hand for every episode with lead > 30 s
  (82 episodes, grouped by feed), and one sample per feed for 15–30 s.

## Results

289 episodes on 20 feeds get a pre-roll cut.

| Lead before the first ad pattern | Episodes | What it is (read) |
|---|---|---|
| 0–15 s | 169 | the pattern is at or near the start |
| 15–30 s | 38 | sponsor pitches (Ramp, Sonos, Framer, Guardian, Attio, Plaud) — correct to cut |
| > 30 s, an ad or show promo | 36 | The Drive's 46 s subscribe promo (28), FT festival promo (4), Odd Lots tour promo (2), Sonos (1), FetchPet / vote-abroad promo (1) |
| **> 30 s, real episode talk** | **46** | **114.5 min in all, median 151 s per episode** |

The 46, by feed:

| Feed | Episodes | Lead | What is cut |
|---|---|---|---|
| Colombia Calling (pdrl.fm) | 15 | 138–263 s | the host's opening monologue, before "sponsored by Latin News" |
| Lenny's Podcast (substack deda024b) | 17 | 120–188 s | the cold open: guest quotes from the episode |
| acast c9151d47 | 4 | 57–142 s | pre-episode banter |
| Machine Learning Street Talk (anchor.fm) | 2 | 65–179 s | cold open |
| Unhedged (acast f08ef8e2) | 2 | 88 s | the episode's intro |
| Odd Lots (omny) | 2 | 126, 209 s | the episode's opening; first "hit" is in conversation ("indeed dot com is", "5% off of its high") |
| The Pragmatic Engineer (substack 8c774140) | 1 | 72 s | the cold open |
| Explaining Brazil (beehiiv) | 1 | 119 s | the show's format-change announcement |
| megaphone 2abd0565 | 1 | 138 s | a reading of the Odyssey's opening |
| Why This Universe (buzzsprout) | 1 | 64 s | the guest introduction |

Two causes:

1. **`start = 0` with a long lead.** A sponsor read in the middle of an intro (Colombia Calling's
   sponsor comes after 2–4 minutes of the host talking) pulls the whole intro into the cut.
2. **A pattern matched in conversation** (Odd Lots: "indeed dot com", "5% off of its high"),
   so the "ad block" is not an ad at all and `start = 0` cuts everything before it.

## What reads the cut

* the ad-free transcript (`workflow/adfree_transcript.py`);
* GI/KG input (`metadata_generation.py`) — no insight is extracted from the cut lead;
* Moments ad spans (`app_episode_search.episode_ad_spans` via `ad_time_intervals`) — no clip is
  picked inside it;
* the diarization commercial rule (`diarization/pipeline.py` via `ad_time_intervals`).

## Not covered / not verified

* The run used the segments' joined text. The prod admaps were built from the transcript text,
  and their counts differ (416 admaps with a pre-roll cut, against 289 here). Episodes cut in
  the admaps but not here were not read.
* The 15–30 s bucket was read one sample per feed, not every episode. The 0–15 s bucket was not read.
* "Real talk" vs "promo" is my reading of the first ~300 characters of each lead. Leads with the
  same text were read once.
* Feeds with no pre-roll cut were not checked for missed ads.
* How many insights or clips the 46 episodes are missing was not measured, and neither was whether
  a cold open's quotes also appear later in the episode (in which case less is lost).
* No fix has been made and no replay has run.
