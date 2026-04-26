<script setup lang="ts">
/**
 * #671 — Search filter chip bar. Replaces the Since/Top‑k inline form
 * row + Advanced filters summary block with 4 chips (Date, Top‑k, Doc
 * types, More). The "More" dialog still hosts low-traffic fields.
 */
import { computed } from 'vue'
import { useSearchStore } from '../../stores/search'
import DateChip from '../shared/DateChip.vue'
import SearchTopKChip from './chips/SearchTopKChip.vue'
import SearchDocTypesChip from './chips/SearchDocTypesChip.vue'
import SearchMoreChip from './chips/SearchMoreChip.vue'

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
      <SearchMoreChip @open="emit('open-more')" />
    </div>
  </div>
</template>
