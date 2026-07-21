<script setup lang="ts">
/**
 * Search filter chip bar. Merged Search + Explore (Search v3 §S1, PRD-045 FR2,
 * RFC-107 §5): chips include the original #671 four (Since, Top‑k, Doc types,
 * More) plus the four merged from the retired Explore surface — Topic contains,
 * Speaker contains, Min confidence, Grounded. Topic + Min confidence are
 * client-side filters over top-K (accuracy caveat inside each chip); Speaker +
 * Grounded pass through to /api/search server-side.
 */
import { computed } from 'vue'
import { useSearchStore } from '../../stores/search'
import DateChip from '../shared/DateChip.vue'
import SearchTopKChip from './chips/SearchTopKChip.vue'
import SearchDocTypesChip from './chips/SearchDocTypesChip.vue'
import SearchMoreChip from './chips/SearchMoreChip.vue'
import SearchTopicChip from './chips/SearchTopicChip.vue'
import SearchSpeakerChip from './chips/SearchSpeakerChip.vue'
import SearchMinConfidenceChip from './chips/SearchMinConfidenceChip.vue'
import SearchEnrichedChip from './chips/SearchEnrichedChip.vue'
import SearchGroundedChip from './chips/SearchGroundedChip.vue'

defineProps<{
  /** When false, chip popovers/inputs are visually disabled. */
  enabled: boolean
  /** Tooltip applied to disabled chips. */
  disabledTitle?: string
}>()

const emit = defineEmits<{
  'open-more': []
}>()

const search = useSearchStore()

const sinceModel = computed({
  get: () => search.filters.since ?? '',
  set: (v: string) => {
    search.filters.since = v
  },
})
</script>

<template>
  <div
    class="flex flex-wrap items-center gap-1.5"
    role="region"
    aria-label="Search filters"
    data-testid="search-filter-bar"
    :title="enabled ? undefined : disabledTitle"
  >
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <DateChip
        v-model="sinceModel"
        label="Since"
        chip-testid="search-chip-since"
        popover-testid="search-popover-since"
      />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchTopKChip />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchDocTypesChip />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchTopicChip />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchSpeakerChip />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchMinConfidenceChip />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchGroundedChip />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchEnrichedChip />
    </div>
    <div :class="enabled ? '' : 'pointer-events-none opacity-50'">
      <SearchMoreChip @open="emit('open-more')" />
    </div>
  </div>
</template>
