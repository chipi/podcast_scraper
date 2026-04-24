<script setup lang="ts">
import type { CilDigestTopicPill } from '../../api/digestApi'
import { cilClusteredTopicPillChrome } from '../../utils/colors'
import { renderPillLabel, type CilPillTruncation } from '../../utils/topicPillLabel'

export type CilClusterMemberAppearance = 'quote' | 'kg'

/**
 * Pre-#656 foundation: make truncation + max-width configurable so the
 * same component works for today's long bullet-slug labels AND the
 * short #653 canonical noun-phrase labels without awkward whitespace or
 * mid-word cutoff. Legacy call sites keep ``ellipsis`` + 24-char default
 * behavior; new call sites opt into ``wrap`` or wider caps.
 */

const props = withDefaults(
  defineProps<{
    pills: CilDigestTopicPill[]
    /** Max chars before applying truncation (ellipsis strategy only). */
    maxPillChars?: number
    /** Truncation strategy: ``ellipsis`` (legacy, default), ``wrap`` (soft line break), ``none`` (full label). */
    truncation?: CilPillTruncation
    /** Tailwind max-width class applied to each pill; ``auto`` shrink-wraps. */
    maxWidthClass?: string
    dataTestid?: string
    /**
     * Cluster-member pill chrome: ``quote`` (amber, legacy) or ``kg`` (Digest Recent parity with graph TopicCluster).
     */
    clusterMemberAppearance?: CilClusterMemberAppearance
  }>(),
  {
    maxPillChars: 24,
    truncation: 'ellipsis',
    // Legacy default preserves today's 11rem cap; new sites pass ``auto`` for short labels.
    maxWidthClass: 'max-w-[11rem]',
    dataTestid: undefined,
    clusterMemberAppearance: 'quote',
  },
)

const emit = defineEmits<{
  'pill-click': [index: number]
}>()

function displayLabel(label: string): string {
  return renderPillLabel(label, props.maxPillChars ?? 24, props.truncation)
}

function buttonShapeClasses(): string {
  const base = 'shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-medium hover:opacity-95'
  // ``wrap`` / ``none`` need the button to grow; ``ellipsis`` keeps the legacy single-line cap.
  const widthClass = props.maxWidthClass === 'auto' ? '' : props.maxWidthClass
  const truncClass =
    props.truncation === 'ellipsis' ? 'truncate' : 'whitespace-normal break-words text-left'
  return [base, widthClass, truncClass].filter(Boolean).join(' ')
}
</script>

<template>
  <div
    v-if="pills.length"
    class="flex flex-wrap gap-1"
    :data-testid="dataTestid"
    role="list"
    aria-label="Topic chips"
  >
    <button
      v-for="(p, i) in pills"
      :key="`${p.topic_id}-${i}`"
      type="button"
      :class="[
        buttonShapeClasses(),
        p.in_topic_cluster
          ? clusterMemberAppearance === 'kg'
            ? 'border border-kg bg-kg/15 font-semibold text-surface-foreground'
            : 'border-2 border-transparent font-semibold text-surface-foreground shadow-sm'
          : 'border border-border bg-canvas text-surface-foreground hover:bg-overlay',
      ]"
      :style="
        p.in_topic_cluster && clusterMemberAppearance === 'quote'
          ? cilClusteredTopicPillChrome
          : undefined
      "
      :title="p.label.trim() || undefined"
      :aria-label="`Open graph for topic: ${p.label}`"
      role="listitem"
      @click.stop="emit('pill-click', i)"
    >
      {{ displayLabel(p.label) }}
    </button>
  </div>
</template>
