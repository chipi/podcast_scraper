<script setup lang="ts">
import { computed } from 'vue'

import type { BridgePartitionSummary } from '../../api/corpusLibraryApi'

/**
 * #656 Stage B: per-episode bridge ``{gi_only, kg_only, both}`` indicator.
 *
 * Meaningful on-screen for the first time post-#654 (the bridge
 * threshold rewrite replaced the mechanical ``both = 10 × episodes``
 * distribution with a real partition). Consumes the backend-computed
 * ``bridge_partition`` field on ``CorpusEpisodeDetailResponse``; the
 * row hides itself when the partition is ``null`` (bridge file missing
 * / unreadable) so legacy episodes don't render an empty placeholder.
 *
 * Visual: three inline counts — ``GI only``, ``Both``, ``KG only`` —
 * in a compact row with a shared border and colour-coded cells. The
 * center cell ("Both") is the one operators actually care about —
 * it's the overlap signal #654 was tuned for. Total is shown in the
 * header muted text so readers can see relative proportions.
 *
 * Accessibility:
 *   - full ``aria-label`` on each cell with the partition name +
 *     count (screen readers announce all three counts even though
 *     the visual cells are compact)
 *   - ``role="group"`` on the wrapper
 *   - safe ``{{ count }}`` interpolation, no v-html
 */

const props = defineProps<{
  partition: BridgePartitionSummary | null | undefined
  /** Optional testid for the row (Playwright). */
  dataTestid?: string
}>()

const hasData = computed(() => !!props.partition && props.partition.total > 0)

const cells = computed(() => {
  const p = props.partition
  if (!p) return []
  return [
    {
      key: 'gi_only' as const,
      label: 'GI only',
      count: p.gi_only,
      tooltip:
        'Identities (topics, persons, orgs) extracted only by the Grounded Insights pipeline.',
      accent: 'bg-gi/10 border-gi/30 text-gi',
    },
    {
      key: 'both' as const,
      label: 'Both',
      count: p.both,
      tooltip:
        'Identities confirmed by BOTH layers — the overlap signal the bridge threshold (#654) is tuned to surface.',
      accent: 'bg-primary/10 border-primary/40 text-primary font-semibold',
    },
    {
      key: 'kg_only' as const,
      label: 'KG only',
      count: p.kg_only,
      tooltip:
        'Identities extracted only by the Knowledge Graph pipeline (usually entities with no quote-backed insight).',
      accent: 'bg-kg/10 border-kg/30 text-kg',
    },
  ]
})
</script>

<template>
  <div
    v-if="hasData"
    class="mt-2 flex flex-col gap-1"
    :data-testid="dataTestid ?? 'episode-bridge-partition'"
    role="group"
    aria-label="Bridge partition summary"
  >
    <div class="flex items-baseline justify-between">
      <span class="text-[10px] font-medium text-muted">
        Bridge partition
      </span>
      <span class="text-[10px] text-muted">
        {{ partition!.total }} identities
      </span>
    </div>
    <div class="flex gap-1">
      <div
        v-for="cell in cells"
        :key="cell.key"
        :class="[
          'flex min-w-0 flex-1 items-baseline justify-between rounded border px-1.5 py-1 font-mono text-[11px] leading-tight',
          cell.accent,
        ]"
        :title="cell.tooltip"
        :aria-label="`${cell.label}: ${cell.count} identities`"
        :data-testid="`bridge-partition-${cell.key}`"
      >
        <span class="truncate font-sans text-[9px] font-medium">
          {{ cell.label }}
        </span>
        <span class="ml-1 tabular-nums">
          {{ cell.count }}
        </span>
      </div>
    </div>
  </div>
</template>
