# Onboarding more shows/episodes to unlock enricher value

Living notes (started 2026-07-06). **Goal:** grow the **eval** corpus (`prod-v2`, ~100 real
episodes today) with more real shows so the enrichers we just built produce *visible value*.
This is about **real content for eval**, distinct from the deterministic **test** fixtures →
see `docs/wip/CORPUS-EVOLUTION-FOR-COMPLEX-ENRICHERS.md`.

> **Why now:** this session shipped `topic_perspectives` (#1146, live) and reframed
> disagreement (#1144) as *scale-gated*. Both get richer strictly with more/better real
> content — perspectives deepen per topic; disagreement + prediction-tracking only *appear*
> at scale + time span. More shows is the lever.

---

## Value model — what each enricher wants from new content

| Enricher / feature | What richer content unlocks | Show/episode selection signal |
| --- | --- | --- |
| `topic_perspectives` (#1146) | More distinct speakers per topic → deeper perspective cards; more topics clear the dashboard's ≥2-speaker bar | **Overlap:** new shows that discuss topics *existing* shows already cover (same topic, new voices) |
| disagreement / prediction (#1144) | The signal that's ~absent today: cross-person opposition + "who called it" | **Debate/panel/dialogue** shows; **recurring contested topics** covered **over time** |
| `guest_coappearance` | Real co-appearance edges | **Multi-guest** episodes (2+ named guests) |
| `temporal_velocity` | Meaningful "heating up" trends | Episodes **spread across months**, not a single snapshot |
| `topic_similarity` (#1105) | Denser, more reliable neighbour clusters | Broad but **thematically clustered** topic coverage |

**The highest-leverage single lever = topic OVERLAP across shows.** Every cross-person enricher
(perspectives, disagreement, co-appearance) needs multiple speakers on the *same* topic. Adding
shows that re-cover existing topics compounds value across all of them at once; adding
disconnected niche shows does not.

---

## How onboarding works (mechanics — to confirm/expand)

- Corpus = pipeline output over a set of RSS feeds (`feeds.spec.yaml` in the corpus root).
- Onboard = add feed(s) to the spec → run the pipeline → enrichers run per the profile.
- prod-v2 today: 100 episodes across ~10 shows (No Priors, Odd Lots, The Journal, Hard Fork,
  NVIDIA AI, The Daily, Unhedged, Invest Like the Best, Planet Money, Latent Space).

- Reprocessing existing transcripts is cheaper than re-scraping (transcripts are ours to keep;
  audio is bridge-only) — relevant when we re-run enrichers over a grown corpus.

*(Confirm the exact onboarding commands + whether it's incremental or full-rebuild.)*

---

## Candidate directions (rank by enricher payoff, not novelty)

1. **Deepen existing topics first** — more AI / markets / policy shows (the topics prod-v2
   already clusters on) → immediate perspective-card + co-appearance payoff.

2. **Add genuinely dialogic shows** — debates, panels, "X vs Y" formats — the only reliable
   source of *opposition* (the #1144 gap) short of decade-scale.

3. **Add a time dimension** — back-catalog episodes of existing shows so recurring topics get a
   real timeline (velocity + prediction-tracking).

4. **Multi-guest formats** — for co-appearance.

## Open questions

- Target size for a "next" eval corpus (v2 → 500? 1000 episodes)? At what point does #1144's
  disagreement signal become *measurable* (the scale gate)?

- Curated vs broad ingest — do we hand-pick for topic overlap, or ingest widely and let the
  enrichers find the density?

- Licensing / bridge constraints on new feeds (audio never rehosted; transcripts + derivatives
  are ours — `[[project_transcript_vs_audio_hosting]]`).

- Cost: transcription + ML enrichment per episode — budget a growth step before committing.

*(Notes doc — extend freely.)*
