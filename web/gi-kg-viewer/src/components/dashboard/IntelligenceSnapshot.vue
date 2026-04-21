<script setup lang="ts">
import { computed } from 'vue'
import type { CorpusDigestResponse } from '../../api/digestApi'
import { useDashboardNavStore } from '../../stores/dashboardNav'

const props = defineProps<{
  digest: CorpusDigestResponse | null
}>()

const emit = defineEmits<{
  'open-digest': []
}>()

const dashNav = useDashboardNavStore()

const windowLine = computed(() => {
  const d = props.digest
  if (!d) {
    return ''
  }
  const n = d.rows?.length ?? 0
  const feeds = new Set((d.rows ?? []).map((r) => r.feed_id).filter(Boolean)).size
  return `Last 7 days — ${n} new episode${n === 1 ? '' : 's'} across ${feeds} feed${feeds === 1 ? '' : 's'}`
})

const topBands = computed(() => (props.digest?.topics ?? []).slice(0, 3))
</script>

<template>
  <section
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="intelligence-snapshot"
  >
    <h3 class="mb-2 text-sm font-semibold">
      Corpus snapshot
    </h3>
    <p
      v-if="!digest"
      class="text-xs text-muted"
    >
      Loading digest…
    </p>
    <template v-else>
      <p class="text-sm">
        {{ windowLine }}
      </p>
      <ul class="mt-2 space-y-1 text-sm">
        <li
          v-for="b in topBands"
          :key="b.topic_id"
        >
          <button
            type="button"
            class="text-left hover:underline"
            @click="dashNav.setHandoff({ kind: 'digest' }); emit('open-digest')"
          >
            {{ b.label }} — {{ b.hits?.length ?? 0 }} episode{{ (b.hits?.length ?? 0) === 1 ? '' : 's' }}
          </button>
        </li>
      </ul>
      <button
        type="button"
        class="mt-2 text-xs font-medium text-primary hover:underline"
        @click="dashNav.setHandoff({ kind: 'digest' }); emit('open-digest')"
      >
        Open Digest →
      </button>
    </template>
  </section>
</template>
