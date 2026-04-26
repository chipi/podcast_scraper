<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  fetchTopicClustersFromApi,
  type TopicClustersCluster,
} from '../../api/corpusTopicClustersApi'
import { useShellStore } from '../../stores/shell'
import { clusterDisplayLabel } from '../../utils/topicClusterDisplay'

/**
 * UXS-006 §5.2 Topic landscape — #656 Stage A.
 *
 * The API fetch + loading / missing / error states were already wired.
 * Stage A polishes the grid for the post-#655 cluster data:
 *   - hover title tooltip so the full ``canonical_label`` is accessible
 *     when the card truncates
 *   - CSS ``truncate`` on the label so long canonical phrases don't
 *     blow out the 12rem grid cell
 *   - schema warning banner (soft — mirrors the API layer, which
 *     returns ``schemaWarning`` for unknown ``schema_version``)
 *   - ``role="list"`` + explicit ``aria-label`` so screen readers
 *     announce the surface correctly
 *   - visible focus ring on each cluster card (keyboard users)
 *
 * Click behavior stays scoped: ``go-graph`` is emitted with the
 * cluster target id so the graph tab opens on that specific cluster
 * when navigation wiring arrives — no consumers rely on the
 * argumentless form today.
 */

const emit = defineEmits<{
  (e: 'go-graph', targetTopicId?: string): void
}>()

const shell = useShellStore()
const clusters = ref<TopicClustersCluster[] | null>(null)
const status = ref<'idle' | 'loading' | 'missing' | 'error'>('idle')
const error = ref<string | null>(null)
const schemaWarning = ref<string | null>(null)

async function load(): Promise<void> {
  const p = shell.corpusPath.trim()
  if (!p || !shell.healthStatus) {
    clusters.value = null
    status.value = 'idle'
    schemaWarning.value = null
    return
  }
  status.value = 'loading'
  error.value = null
  schemaWarning.value = null
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
  schemaWarning.value = r.schemaWarning ?? null
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

function onClusterActivate(c: TopicClustersCluster): void {
  const target = c.cil_alias_target_topic_id || c.canonical_topic_id || undefined
  emit('go-graph', target)
}
</script>

<template>
  <section
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="intelligence-topic-landscape"
    aria-labelledby="topic-landscape-heading"
  >
    <h3
      id="topic-landscape-heading"
      class="mb-2 text-sm font-semibold"
    >
      Topic landscape
    </h3>
    <p
      v-if="schemaWarning"
      class="mb-2 rounded border border-warning/40 bg-warning/10 px-2 py-1 text-[10px] text-warning"
      role="status"
    >
      {{ schemaWarning }}
    </p>
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
      role="list"
      aria-label="Topic clusters"
    >
      <button
        v-for="(cl, i) in clusters ?? []"
        :key="cl.graph_compound_parent_id ?? cl.cluster_id ?? String(i)"
        type="button"
        role="listitem"
        class="rounded border border-border bg-elevated p-2 text-left text-sm transition hover:bg-overlay focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        :title="clusterDisplayLabel(cl)"
        :aria-label="`${clusterDisplayLabel(cl)} — ${cl.members?.length ?? 0} topics`"
        @click="onClusterActivate(cl)"
      >
        <div class="truncate font-semibold">
          {{ clusterDisplayLabel(cl) }}
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
