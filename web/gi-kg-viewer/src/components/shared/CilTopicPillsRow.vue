<script setup lang="ts">
import type { CilDigestTopicPill } from '../../api/digestApi'
import { cilClusteredTopicPillChrome } from '../../utils/colors'

export type CilClusterMemberAppearance = 'quote' | 'kg'

const props = withDefaults(
  defineProps<{
    pills: CilDigestTopicPill[]
    maxPillChars?: number
    dataTestid?: string
    /**
     * Cluster-member pill chrome: ``quote`` (amber, legacy) or ``kg`` (Digest Recent parity with graph TopicCluster).
     */
    clusterMemberAppearance?: CilClusterMemberAppearance
  }>(),
  {
    maxPillChars: 24,
    dataTestid: undefined,
    clusterMemberAppearance: 'quote',
  },
)

const emit = defineEmits<{
  'pill-click': [index: number]
}>()

function shortLabel(label: string): string {
  const cap = props.maxPillChars ?? 24
  const s = label.trim()
  if (s.length <= cap) {
    return s
  }
  return `${s.slice(0, cap - 1)}…`
}
</script>

<template>
  <div
    v-if="pills.length"
    class="flex flex-wrap gap-1"
    :data-testid="dataTestid"
  >
    <button
      v-for="(p, i) in pills"
      :key="`${p.topic_id}-${i}`"
      type="button"
      class="max-w-[11rem] shrink-0 truncate rounded-full border px-1.5 py-0.5 text-[10px] font-medium hover:opacity-95"
      :class="
        p.in_topic_cluster
          ? clusterMemberAppearance === 'kg'
            ? 'border border-kg bg-kg/15 font-semibold text-surface-foreground'
            : 'border-2 border-transparent font-semibold text-surface-foreground shadow-sm'
          : 'border border-border bg-canvas text-surface-foreground hover:bg-overlay'
      "
      :style="
        p.in_topic_cluster && clusterMemberAppearance === 'quote'
          ? cilClusteredTopicPillChrome
          : undefined
      "
      :title="p.label.trim() || undefined"
      :aria-label="`Open graph for topic: ${p.label}`"
      @click.stop="emit('pill-click', i)"
    >
      {{ shortLabel(p.label) }}
    </button>
  </div>
</template>
