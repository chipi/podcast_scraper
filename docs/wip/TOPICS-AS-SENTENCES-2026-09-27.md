# 96 episodes render zero topic chips, because GI wrote insights into the Topic nodes

Found from one operator screenshot (2026-09-27): an episode's Insights panel showed
**"Topics & People · 1"** — a single person chip, no topics at all — on an episode carrying nine
insights.

**The filter is not the bug. The data is.** This is a GI extraction defect, and it is repairable by
re-deriving the affected episodes; nothing needs deleting.

---

## What is actually in the KG

The episode's KG has **10 Topic nodes**, and every one is a 27–35 word sentence:

```text
The November 2025 inflection point — GPT-5.1 and Claude Opus…      29 words
Agentic engineering — using coding agents professionally — r…      28 words
The lethal trifecta — an agent with access to private data,…       35 words
```

Those are the **insight texts**. They appear verbatim as insights in the same panel. GI emitted its
insight sentences as `Topic` nodes.

`is_filler_topic` then rejects all ten on `_TOPIC_MAX_RAW_WORDS`
(`app_kg_view.py:107`, the same predicate the corpus enrichers apply), which is **correct** — a
35-word sentence is not a topic, must not become a followable interest, and must not become a
discover signal. The chip row is empty because there was nothing legitimate to put in it.

So the visible symptom is a filter working as designed on top of bad input.

## Scale, measured on prod

Scanned every `*.kg.json` under `corpus/feeds/*/run_*/metadata/`:

| | |
| --- | --- |
| episodes with Topic nodes | 2,283 |
| …with ANY sentence-shaped topic (>8 words) | 97 (4%) |
| …with **EVERY** topic sentence-shaped | **96 (4%)** — these render zero topic chips |
| topic nodes total | 22,458 |
| …sentence-shaped | 830 (4%) |

**97 have any, 96 have all.** It is essentially binary: an episode either gets proper topics or it
gets none. That is the signature of a per-extraction failure, not of noise.

## It clusters in TIME, not by feed

| run date | affected |
| --- | --- |
| 2026-08-11 | 1 |
| 2026-08-12 | 13 |
| 2026-08-13 | 15 |
| **2026-08-14** | **53** |
| 2026-08-18 | 2 |
| 2026-08-19 | 3 |
| 2026-08-27 → 09-02 | 9 |

**82 of 96 land in 2026-08-11 → 08-19**, across **15 distinct feeds** and **24 distinct runs**. A
failure that spans unrelated feeds but concentrates in one window is a pipeline or model-state
problem — a prompt, a model swap, or a schema change in that period — not anything about the
shows themselves.

The top-affected feeds are simply the ones with the most episodes processed in that window
(`rss_api.substack.com_8c774140` 47, `…deda024b` 14, `rss_feeds.megaphone.fm_3581c092` 14).

## What has NOT been established

- **The cause in the pipeline.** The window is identified; the change inside it is not. Nobody has
  diffed the GI prompt, model pin or schema across 2026-08-11 → 08-19. That is the next step and it
  is where the actual fix lives.
- **Whether the tail (2026-08-27 → 09-02, 9 episodes) is the same cause** or a second one that
  merely looks alike.
- **Whether re-running GI on an affected episode produces good topics now.** If the cause was
  transient and has since been corrected, a re-derive fixes all 96. If it is still live, re-running
  reproduces it — and that is the cheapest possible test of whether the defect is still shipping.
  **Do that on ONE episode before any batch.**
- Whether the same episodes are also degraded anywhere else that consumes Topic nodes — discover
  signals, followable interests, `feed_signals.top_topics`, the corpus co-occurrence enricher. All
  of them apply the same filter, so all of them are quietly missing these episodes' topics. Nobody
  has measured that blast radius.

## Why nothing in the app should change

The rendering, the filter and the panel are all behaving correctly. Loosening
`_TOPIC_MAX_RAW_WORDS` to make the chips appear would push 830 sentence-shaped "topics" into the
interest picker, the discover signals and the entity cards — turning an invisible data defect into
a visible product one. The fix belongs in GI, and the 96 episodes are re-derivable.
