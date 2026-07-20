# Super-theme rollup — signal comparison (follow-up)

**Status:** queued — not started
**Owner:** operator (Marko)
**Related:** graph-v3 tier 7-1a (shipped in commit forthcoming);
              `src/podcast_scraper/enrichment/enrichers/topic_theme_clusters.py`

## Context

Tier 7-1a shipped a super-theme rollup on top of `topic_theme_clusters` using
**cross-cluster topic-lift mean** (shared-bridge / co-occurrence in
same-episode windows). Two alternate signals were considered and parked for
later comparison so we can pick the best one after we've lived with the first
on real content.

## Signals to compare

1. **Cross-cluster topic-lift mean** *(shipped as default in v1.1.0)*
   For each pair of clusters, mean pairwise lift between their member
   topics. Reuses the existing lift graph from the first pass. Semantic
   meaning: "these two themes co-occur in the same episodes more than
   chance."

2. **Cluster centroid cosine similarity**
   Embed each cluster (mean of member-topic embeddings or of the
   canonical topic's embedding), then merge closest centroids under
   average-linkage. Semantic meaning: "these two themes are about
   similar things" — closer to what the semantic `topic_clusters`
   enricher does at the topic level. Would require plumbing embeddings
   from `topic_similarity` into this enricher (currently a decoupled
   deterministic-tier enricher — cross-enricher deps need `_loaders`
   support).

3. **Member-topic set Jaccard overlap**
   For each pair of clusters, `|A ∩ B| / |A ∪ B|` over topic ids.
   Almost always 0 because the first-pass average-linkage produces
   disjoint clusters — so this variant needs a re-clustering step that
   allows fuzzy membership, OR needs the overlap computed over each
   cluster's **1-hop lift neighbourhood** (the topics with non-trivial
   lift to a cluster member, even if not in the cluster). Cheap and
   fully deterministic.

## Comparison harness

- Same corpus (prod-v2), same first-pass output.
- Run each signal producing a super-theme assignment.
- Metrics to record per signal:
  - Super-theme sizes (want a distribution, not one giant + many singletons)
  - Cross-signal agreement (how often do 3 signals agree on which
    super-theme two clusters land in?)
  - Human labelling: for each corpus, do the super-themes make
    editorial sense to a human? (Read the canonical labels aloud —
    do they cluster into a coherent narrative arc?)
- Winner criterion: the one that, on real editorial reading, produces
  super-themes a human would name unprompted.

## Non-goals

- Not building a scorer for this. Human read + a compact table in this
  doc is enough to pick a winner.
- No new enricher; this modifies the existing one via a `super_theme_method`
  config knob.

## Notes

- Enricher writes `super_theme_method: "cross_cluster_lift_avg_linkage"`
  on every payload so downstream consumers know which signal produced
  the rollup.
- If we ever ship user-controllable super-themes ("regroup by X"), the
  same knob turns into a viewer preference.
