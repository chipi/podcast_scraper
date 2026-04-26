import type { TopicClustersCluster } from '../api/corpusTopicClustersApi'

/**
 * #656 Stage A: canonical display label for a topic cluster.
 *
 * Pure function, extracted from ``TopicLandscape.vue`` so ``<script
 * setup>`` can stay free of exports while the fallback chain is unit-
 * testable. Prefers, in order:
 *
 *   1. ``canonical_label`` (post-#653: a short canonical noun phrase
 *      like ``"oil prices"``)
 *   2. ``cil_alias_target_topic_id`` (post-RFC-075 v2 merge target id,
 *      e.g. ``"topic:oil-prices"``)
 *   3. ``canonical_topic_id`` (legacy v1 field)
 *   4. literal ``"Cluster"`` — last-resort so the card is never empty.
 *
 * Whitespace-only ``canonical_label`` is treated as absent so cards
 * don't render as bare padding.
 */
export function clusterDisplayLabel(cluster: TopicClustersCluster): string {
  const trimmed = cluster.canonical_label?.trim()
  if (trimmed) {
    return trimmed
  }
  const alias = cluster.cil_alias_target_topic_id
  if (alias && alias.trim()) {
    return alias
  }
  const legacy = cluster.canonical_topic_id
  if (legacy && legacy.trim()) {
    return legacy
  }
  return 'Cluster'
}
