# Moments and Step — plan (2026-10-10)

A beta tester liked the player's live insight card and asked to move through an episode by its
insights: "jump to the next bit in my command, like speed checking just the highlights". Two
features answer that, sharing one idea (the insights are the episode's chapters):

- **Step** — previous / next insight during normal playback.
- **Moments** — a quick-listening mode for people on the run: the episode's strongest moments,
  back to back, each a clip of the speaker making the point. It pulls the best moments instead of
  synthesising.

Design brief with every mockup: <https://claude.ai/artifact/Ww8ogdXqoVqqkg7WoAVcbZ>.
Obi study: <https://claude.ai/artifact/FPk6sK8BSYSqDh55q5SHs2>.

## Decided with the operator (2026-10-10)

| Topic | Decision |
| --- | --- |
| Step controls | Arrows on the live insight card ("‹ 4 / 40 ›") plus swipe. No long-press on the skip buttons. Between insights the "Next · in 0:06" line carries the arrows. Desktop: `[` and `]`. |
| Step behaviour | Jumps to the insight's first quote; previous behaves like a music player (within 3 s, the one before). Works paused. |
| Name | **Moments** ("Highlights" is taken: the listener's saved highlights, 39 uses in `en.json`). |
| Player entrance | A third door on the obi. Three **equal** doors, all labels the same 11px kicker: **Moments · Brief · About** ("Description" does not fit a third of the artwork: 92px at 11px against 63–73px of room). In Moments mode the top door reads **Episode**. |
| Count | Dynamic, server config: one moment per 6 min, min 5, max 15, never more than the episode has. Clip cap 30 s, moments at least 3 min apart. |
| Ticks | Moments are marked on the density strip under the scrubber, not on the scrubber. |
| Lock screen / headphones | In Moments mode next / previous skip moments. Normal playback unchanged. |
| Listening | A reel does not count as listening: no listen logged, no milestones, not marked heard. |
| Everywhere | First item in the shared episode ⋯ menu. A visible chip ("▶ Moments · 3½ min") on its own line in search episode results, the Discover episode list, show pages, Discover 2 cards and Downloads. Not on Home tiles (108px, three controls already), not on passage results. |
| Discovery | Find → taste (moments) → "Keep listening here" into the full episode. |
| Offline | A download saves the episode's moments with it (like the Brief); the reel plays from the file. Not-downloaded episodes show the chip greyed offline. |

## Which moments — measured

Prod, 2,924 episodes (read-only, 2026-10-10): median 40 insights per episode, 25 s of quotes per
insight; playing every insight would take 17 min (37% of a 48-min episode). Salience cannot pick a
top N: a median of 2 distinct values per episode, 32 insights tied at the top.

**Check 1 — ranking** (`scripts/eval/moments_eval.py`, 100 episodes / 48 shows, blind, Claude
Sonnet judge, Gemini 2.5 Pro on 20): a score from depth, topic centrality, insight kind and speaker
**lost** to the player's existing order (extraction order, since salience ties): 2.90 vs 3.02,
baseline preferred in 54 episodes to 25. Per signal (Spearman vs judge): depth −0.05, centrality
+0.01, kind −0.01; extraction order −0.17; clip length +0.34, quote length +0.42 (model judges also
favour length — ambiguous). Cost $1.39. **Decision: rank by the player's order; work on the clip.**

**Check 2 — clips** (`--mode clips`, same 100 episodes, moments fixed): A = new clip (sentence
start, nearby quotes, ≥ 12 s), B = first quote, C = B stretched to A's length (length control).
Result: **new clips beat first-quote clips**, Claude 3.05 vs 2.91 (A preferred in 63 episodes
to 22); Gemini on 10 episodes spread across lengths agrees (3.60 vs 3.35). **The gain is length**:
the length-matched control C scored 3.04, tied with A (39–39). So longer clips help, but sentence
alignment adds nothing a text judge can measure; whether length is real or judge taste is not
separable here. A is kept: it scores the same as C and never starts or ends mid-word, which matters
to a listener and is invisible to a text judge. Every arm is still about 3/5: insight quality is
the limit. Cost $2.94 (plus $1.47 lost to a parser crash before the run saved per judgment, now
fixed: judgments are cached in `judgments_<mode>.jsonl` and a crash resumes).

## Build plan

1. **Server — `app_moments.py`** (written, unit-tested): `pick_moments(gi, duration, config,
   segments)`; `MomentsConfig` from `APP_MOMENTS_CONFIG` (JSON). Ranking `player` default, `score`
   kept as an option; clip rule `segments` default.
2. **Server — route** `GET /api/app/episodes/{slug}/moments`: reads the GI artifact, raw
   `*.segments.json` and duration; returns moments (id, text, speaker, start/end ms, clip text) plus
   total seconds. Integration test with the fixture corpus.
3. **Server — moments on lists**: episode payloads carry `moments_seconds` (or a count) so the chip
   can show "3½ min" without a second fetch.
4. **Offline**: the download bundle stores the moments response with the episode (alongside the
   Brief); the client reads it from disk offline.
5. **Player — Step**: arrows + swipe on the Zone D card and the rest line; keyboard on desktop;
   reuse `nextInsightIndex` / insight start times.
6. **Player — obi**: three equal doors (Moments / Brief / About); the Description door becomes
   About (sheet title unchanged).
7. **Player — Moments mode**: reel playback over the moment windows, segment bar on the artwork,
   skip buttons and media-session next/previous skip moments, "Keep listening here", ✕ back to
   the prior position, end card; no listen logging in the mode; density-strip ticks.
8. **Everywhere**: ⋯ menu item in `EpisodeActions` → `OverflowMenu`; chip on the listed surfaces;
   the Brief panel's "Play N moments" button.
9. **Docs**: UXS-011 / UXS-014 entries, surface map, i18n.

Tests per layer: unit (selection, clips, config, reel state machine), integration (route on the
fixture corpus), e2e (Step, Moments mode, obi doors, chip, offline reel with the network cut).

## Not covered yet

- Screen-reader design for the card arrows, the reel's segment bar and the obi's third door.
- How the move between clips sounds (cut, fade, tick).
- Episode rows on topic / person / storyline pages (no actions today) — a later step.
- Label widths on Android: measured in desktop Chromium with SF Mono only.
