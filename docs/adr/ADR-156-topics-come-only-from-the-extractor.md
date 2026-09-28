# ADR-156: Topics come only from the extractor — no fallback

- **Status**: Proposed
- **Date**: 2026-09-28
- **Authors**: Marko Dragoljevic
- **Related issues**: #2164 (96 episodes render zero topics), #1208 (the first half of this
  fallback removed), #1936 (`_enforce_noun_phrase_label`), #587 (noun-phrase cap), #1933
  (canonical topic slug)

## Context & Problem Statement

A `Topic` node is not just a node. It becomes a theme-cluster member, a co-occurrence pair, a
trending chip, a followable interest, an entity card, and — through clustering — the substrate for
storylines. Anything that reaches the `Topic` type is offered to a listener as a subject worth
following.

Topics have therefore had **two** sources: the KG extraction provider, and a fallback that turned
the episode's `summary.bullets` into `Topic` nodes when no provider ran. The fallback predates most
of the product and was, by its own docstring, for "tests and legacy callers".

The fallback does not work, and #2164 measured the damage on prod (2026-09-27):

| | |
| --- | --- |
| episodes with `Topic` nodes | 2,283 |
| …where EVERY topic is a 27–35 word sentence | **96** — these render no topic chips at all |
| sentence-shaped topic nodes corpus-wide | 830 |

Summary bullets are prose. A 30-word bullet is not a subject, so `is_filler_topic` correctly
rejects every one of them and the chip row renders empty. The app layer is behaving correctly on
bad input.

Three findings make the fallback unsalvageable rather than merely buggy:

1. **It was never coherent.** The workflow layer shortens bullets through
   `_bullet_to_topic_phrase` (max 4 tokens) before handing them to the **GI** path
   (`metadata_generation.py:4925`) and does **not** before handing them to the **KG** path
   (`:5174`). The same bullets therefore became 4-token phrases in one artifact and 30-word
   sentences in another. A fallback with a shortening step that one of its two consumers skips is
   not a design, it is an accident.
2. **The filtering architecture has a hole.** `_loaders.topic_nodes` documents the contract —
   "filler is still WRITTEN into new KGs; it is filtered on the way out, at the two read
   chokepoints". But `kg/corpus.topic_cooccurrence` reads `art["nodes"]` directly and applies no
   filter, so fabricated sentence-topics **do** become co-occurrence pairs, feeding trending and
   theme clusters. The 830 nodes have been polluting those surfaces all along.
3. **Re-deriving repairs the episodes properly.** Verified on the #2164 repro episode against the
   live DGX route (`prod_dgx_full`, `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`):
   `model_version: topic_labels -> provider:…`, topics surviving the gate `0 -> 10`, sentence-shaped
   `10 -> 0`, and entity nodes `1 -> 15`. So the extractor handles this content fine; the affected
   artifacts are stale, not unfixable. The fallback is not protecting against anything.

## Decision

**Only the KG extraction provider may create a `Topic` node.** There is no fallback.

When no extractor runs, the episode has **no topics**, the artifact says so in its provenance, and
every downstream consumer correctly sees nothing.

Consequently:

- the bullets→`Topic` path in `kg/pipeline.build_artifact` is removed, along with
  `_append_topics_from_labels`, `_topic_labels_from_args`, and the `topic_label` / `topic_labels`
  parameters — which have no other consumer
- GI's bullet-derived topic nodes are removed; GI receives topics **only** by alignment from the
  KG's canonical set (`gi/topic_alignment.align_gi_topics_with_kg`), which becomes their sole source
- `topic_alignment`'s "no-op when the KG declares no topics, better to keep GI's own" branch is
  inverted: keeping GI's own means keeping the bullet-derived ones
- `utils/corpus_graph_bullet_sync` is removed — it exists to patch bullet-derived topics
- `_bullet_to_topic_phrase` and its call sites are removed
- provenance gains an explicit `no_extractor`, distinct from `metadata_only`: the latter is a
  deliberate reduced mode, the former is a misconfiguration, and collapsing them makes the accident
  unreadable

`kg/topic_clustering` and `gi/topic_alignment` are **kept**: they derive from real extractor topics
rather than inventing new ones.

## Rationale

An empty topic set is honest and recoverable. A fabricated one is neither: it is invisible at the
app layer (the filter hides it), corrosive underneath (co-occurrence, clustering, trending), and it
masks a real operational failure — the 96 episodes said "extraction succeeded, here are 10 topics"
while GI had not run at all and 14 of 15 entity nodes were missing.

## Alternatives Considered

1. **Loosen `_TOPIC_MAX_RAW_WORDS` so the chips appear.** Rejected. It would push 830
   sentence-shaped "topics" into the interest picker, discover signals and entity cards, converting
   an invisible data defect into a visible product one. The app layer is correct.
2. **Apply `_bullet_to_topic_phrase` on the KG path too**, making the fallback produce 4-token
   phrases. Rejected: it preserves a fallback that fabricates topics from prose, only shorter. The
   first four words of a summary bullet are not the episode's subject, and such a topic still cannot
   recur across episodes, so it can never cluster — structurally incapable of becoming a storyline.
3. **Keep the parameters but ignore them.** Rejected: a parameter that looks live and is not is how
   this accumulated in the first place.
4. **Better topic extraction (hints, candidate generation, ranking).** Deferred deliberately. That
   is an enhancement with its own evidence bar, not part of fixing this. Recorded here so it is not
   confused with remediation.

## Consequences

- **Positive**: one topic source and one propagation path (extractor → KG → GI by alignment);
  co-occurrence, clustering and trending stop ingesting propositions; a failed extraction becomes
  visible instead of being disguised as success.
- **Negative**: episodes processed without a provider now show zero topics where they previously
  showed (unusable) chips. This is the intended change and affects the 96 known episodes until they
  are re-derived.
- **Neutral**: a test asserting the old behaviour (`test_the_legacy_hint_path_is_untouched`) is
  deliberately inverted; the chaos-artifact checker's `provenance="topic_labels"` case is no longer
  producible.

## Implementation Notes

- **Modules**: `podcast_scraper/kg/pipeline.py`, `podcast_scraper/gi/pipeline.py`,
  `podcast_scraper/gi/topic_alignment.py`, `podcast_scraper/utils/corpus_graph_bullet_sync.py`,
  `podcast_scraper/workflow/metadata_generation.py`
- **Safety argument**: consumers read `Topic` nodes off disk and cannot tell whether a node came
  from the extractor or a bullet, so removing the fabricators cannot change how a real topic is
  treated — only how many nodes exist. Pinned by
  `tests/unit/podcast_scraper/kg/test_topic_consumers_characterization.py`, which must pass
  identically before and after the removal.
- **Acceptance criterion for the local re-derive POC**: extraction is sampled (`temperature`
  defaults to `0.3`, `seed` unset), so topic *labels* churn between identical runs — measured at
  roughly 40% of labels on a control pass. Byte-identical output is therefore not a valid criterion.
  The criterion is **structural**: `topic_count`, `surviving_count`, `sentence_shaped` and
  `entity_count` unchanged for healthy episodes, with only the affected episode's structure changing.

## References

- #2164 — the measurement and the repro episode
- `podcast_scraper/enrichment/enrichers/_loaders.py` — the read-chokepoint contract
- `podcast_scraper/kg/filters.py` — `is_filler_topic`, and slug-based truncation detection
