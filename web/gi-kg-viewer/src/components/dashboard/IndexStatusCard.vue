<script setup lang="ts">
import { computed } from 'vue'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'

const emit = defineEmits<{
  'rebuild-index': []
}>()

const shell = useShellStore()
const indexStats = useIndexStatsStore()

const indexedLabel = computed(() => {
  const env = indexStats.indexEnvelope
  if (!env?.available || !env.stats) {
    return '—'
  }
  const v = env.stats.total_vectors
  const feeds = env.stats.feeds_indexed?.length ?? 0
  return `${v.toLocaleString()} vectors · ${feeds} feed${feeds === 1 ? '' : 's'} listed`
})

const lastRebuild = computed(() => {
  const iso = indexStats.indexEnvelope?.stats?.last_updated?.trim()
  if (!iso) {
    return '—'
  }
  const t = Date.parse(iso)
  if (Number.isNaN(t)) {
    return iso
  }
  const days = Math.floor((Date.now() - t) / 86_400_000)
  if (days <= 0) {
    return 'today'
  }
  if (days === 1) {
    return '1 day ago'
  }
  return `${days} days ago`
})
</script>

<template>
  <div
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="index-status-card"
  >
    <h3 class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
      Index status
    </h3>
    <p class="text-sm">
      Last rebuilt: <span class="font-medium">{{ lastRebuild }}</span>
      <span
        v-if="indexStats.indexEnvelope?.reindex_recommended"
        class="ml-2 text-warning"
      >⚠ Rebuild recommended</span>
    </p>
    <p class="mt-1 text-sm text-muted">
      {{ indexedLabel }}
    </p>
    <p
      v-if="indexStats.indexEnvelope?.rebuild_last_error"
      class="mt-2 text-xs text-danger"
    >
      Last rebuild error: {{ indexStats.indexEnvelope.rebuild_last_error }}
    </p>
    <div class="mt-2 flex flex-wrap gap-2">
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay disabled:opacity-40"
        data-testid="index-status-update"
        :disabled="
          !shell.healthStatus
            || !shell.hasCorpusPath
            || indexStats.rebuildActionsDisabled
            || indexStats.indexEnvelope?.rebuild_in_progress
        "
        @click="indexStats.requestIndexRebuild(false)"
      >
        {{ indexStats.rebuildSubmitting ? 'Queueing…' : 'Update index' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay disabled:opacity-40"
        data-testid="index-status-full-rebuild"
        :disabled="
          !shell.healthStatus
            || !shell.hasCorpusPath
            || indexStats.rebuildActionsDisabled
            || indexStats.indexEnvelope?.rebuild_in_progress
        "
        @click="emit('rebuild-index')"
      >
        {{
          indexStats.indexEnvelope?.rebuild_in_progress
            ? 'Rebuilding…'
            : 'Full rebuild'
        }}
      </button>
    </div>
  </div>
</template>
