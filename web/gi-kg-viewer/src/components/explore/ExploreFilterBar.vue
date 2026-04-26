<script setup lang="ts">
/**
 * #671 — Explore filter chip bar. Replaces the Topic/Speaker text-input
 * stack + Advanced filters summary block with 3 chips (Topic, Speaker,
 * More). The "More" dialog still hosts low-traffic fields.
 */
import { computed } from 'vue'
import { useExploreStore } from '../../stores/explore'
import ExploreTextChip from './chips/ExploreTextChip.vue'
import ExploreMoreChip from './chips/ExploreMoreChip.vue'

const props = defineProps<{
  enabled: boolean
  disabledTitle?: string
}>()

const emit = defineEmits<{
  'open-more': []
  'submit': []
}>()

const ex = useExploreStore()

const topicModel = computed({
  get: () => ex.filters.topic,
  set: (v: string) => {
    ex.filters.topic = v
  },
})

const speakerModel = computed({
  get: () => ex.filters.speaker,
  set: (v: string) => {
    ex.filters.speaker = v
  },
})
</script>

<template>
  <div
    class="flex flex-wrap items-center gap-1.5"
    role="region"
    aria-label="Explore filters"
    data-testid="explore-filter-bar"
  >
    <ExploreTextChip
      v-model="topicModel"
      label="Topic"
      chip-testid="explore-chip-topic"
      popover-testid="explore-popover-topic"
      :enabled="props.enabled"
      :disabled-title="props.disabledTitle"
      @submit="emit('submit')"
    />
    <ExploreTextChip
      v-model="speakerModel"
      label="Speaker"
      chip-testid="explore-chip-speaker"
      popover-testid="explore-popover-speaker"
      :enabled="props.enabled"
      :disabled-title="props.disabledTitle"
      @submit="emit('submit')"
    />
    <div :class="props.enabled ? '' : 'pointer-events-none opacity-50'">
      <ExploreMoreChip @open="emit('open-more')" />
    </div>
  </div>
</template>
