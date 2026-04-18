<script setup lang="ts">
import type { CilDigestTopicPill } from '../../api/digestApi'
import { cilClusteredTopicPillChrome } from '../../utils/colors'

const props = withDefaults(
  defineProps<{
    pills: CilDigestTopicPill[]
    maxPillChars?: number
    dataTestid?: string
  }>(),
  {
    maxPillChars: 24,
    dataTestid: undefined,
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
      class="max-w-[11rem] shrink-0 truncate rounded-full border px-1.5 py-0.5 text-[10px] font-medium text-surface-foreground hover:opacity-95"
      :class="
        p.in_topic_cluster
          ? 'border-2 font-semibold shadow-sm'
          : 'border border-border bg-canvas hover:bg-overlay'
      "
      :style="p.in_topic_cluster ? cilClusteredTopicPillChrome : undefined"
      :title="p.label.trim() || undefined"
      :aria-label="`Open graph for topic: ${p.label}`"
      @click.stop="emit('pill-click', i)"
    >
      {{ shortLabel(p.label) }}
    </button>
  </div>
</template>
