<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { fetchTopicClustersFromApi, type TopicClustersCluster } from '../../api/corpusTopicClustersApi'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()
const clusters = ref<TopicClustersCluster[] | null>(null)
const status = ref<'idle' | 'loading' | 'missing' | 'error'>('idle')
const error = ref<string | null>(null)

async function load(): Promise<void> {
  const p = shell.corpusPath.trim()
  if (!p || !shell.healthStatus) {
    clusters.value = null
    status.value = 'idle'
    return
  }
  status.value = 'loading'
  error.value = null
  const r = await fetchTopicClustersFromApi(p)
  if (r.status === 'missing') {
    clusters.value = null
    status.value = 'missing'
    return
  }
  if (r.status === 'error') {
    clusters.value = null
    status.value = 'error'
    error.value = r.message
    return
  }
  clusters.value = r.document.clusters ?? []
  status.value = 'idle'
}

onMounted(() => {
  void load()
})
watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    void load()
  },
)

const insight = computed(() => {
  const c = clusters.value
  if (!c?.length) {
    return undefined
  }
  const topics = c.reduce((n, cl) => n + (cl.members?.length ?? 0), 0)
  return `${c.length} topic clusters covering ${topics} distinct topics.`
})
</script>

<template>
  <section
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="intelligence-topic-landscape"
  >
    <h3 class="mb-2 text-sm font-semibold">
      Topic landscape
    </h3>
    <p
      v-if="status === 'loading'"
      class="text-xs text-muted"
    >
      Loading…
    </p>
    <p
      v-else-if="status === 'missing'"
      class="text-xs text-muted"
    >
      Topic clusters not yet built for this corpus.
    </p>
    <p
      v-else-if="error"
      class="text-xs text-danger"
    >
      {{ error }}
    </p>
    <div
      v-else
      class="grid gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr))]"
    >
      <button
        v-for="(cl, i) in clusters ?? []"
        :key="cl.graph_compound_parent_id ?? cl.cluster_id ?? String(i)"
        type="button"
        class="rounded border border-border bg-elevated p-2 text-left text-sm hover:bg-overlay"
        @click="$emit('go-graph')"
      >
        <div class="font-semibold">
          {{ cl.canonical_label?.trim() || cl.cil_alias_target_topic_id || 'Cluster' }}
        </div>
        <div class="mt-1 text-[10px] text-muted">
          {{ cl.members?.length ?? 0 }} topics
        </div>
      </button>
    </div>
    <p
      v-if="insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ insight }}
    </p>
  </section>
</template>
