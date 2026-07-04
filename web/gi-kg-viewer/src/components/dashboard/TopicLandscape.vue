<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  fetchTopicClustersFromApi,
  type TopicClustersCluster,
} from '../../api/corpusTopicClustersApi'
import { useShellStore } from '../../stores/shell'
import { clusterDisplayLabel } from '../../utils/topicClusterDisplay'
import { graphCompoundParentIdFromCluster } from '../../utils/topicClustersOverlay'

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
 * Click: emit ``go-graph`` with the best graph focus id — prefer
 * ``graph_compound_parent_id`` / ``tc:…`` so the rail matches **NodeDetail**
 * (TopicCluster) and Cytoscape can select the compound; optional second arg is
 * a member ``topic:…`` id for ``requestFocusNode`` fallback when the compound
 * is not yet in the rendered graph.
 */

const emit = defineEmits<{
  (e: 'go-graph', targetId?: string, focusFallbackId?: string): void
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

const view = ref<'bubbles' | 'grid'>('bubbles')

/** Clusters as size-ranked bubbles: circle **area** ∝ topic count (diameter ∝ √count), clamped to
 *  a tappable range, so the clusters covering the most topics read as the biggest at a glance (#3). */
const bubbles = computed(() => {
  const c = clusters.value ?? []
  const maxN = Math.max(1, ...c.map((cl) => cl.members?.length ?? 0))
  return c
    .map((cl, i) => ({
      key: cl.graph_compound_parent_id ?? cl.cluster_id ?? String(i),
      cl,
      n: cl.members?.length ?? 0,
    }))
    .sort((a, b) => b.n - a.n)
    .map((b) => ({ ...b, d: Math.round(44 + (120 - 44) * Math.sqrt(b.n / maxN)) }))
})

function firstMemberTopicId(c: TopicClustersCluster): string | undefined {
  const m = c.members?.find((x) => (x.topic_id ?? '').trim().length > 0)
  return m?.topic_id?.trim()
}

/** Prefer TopicCluster compound id for graph + NodeDetail; else CIL topic id. */
function clusterActivateTargets(c: TopicClustersCluster): {
  primary?: string
  fallback?: string
} {
  const compound = graphCompoundParentIdFromCluster(c)
  if (compound) {
    return { primary: compound, fallback: firstMemberTopicId(c) }
  }
  const topic =
    (c.cil_alias_target_topic_id ?? '').trim() ||
    (c.canonical_topic_id ?? '').trim() ||
    firstMemberTopicId(c)
  return { primary: topic || undefined, fallback: undefined }
}

function onClusterActivate(c: TopicClustersCluster): void {
  const { primary, fallback } = clusterActivateTargets(c)
  emit('go-graph', primary, fallback)
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
    <template v-else>
      <!-- View toggle: bubbles (size = topic coverage) vs the compact grid. -->
      <div
        class="mb-2 inline-flex rounded-md border border-border p-0.5 text-[10px]"
        role="tablist"
        aria-label="Topic landscape view"
      >
        <button
          v-for="opt in (['bubbles', 'grid'] as const)"
          :key="opt"
          type="button"
          role="tab"
          :aria-selected="view === opt"
          :data-testid="`topic-landscape-view-${opt}`"
          class="rounded px-2 py-0.5 font-medium capitalize transition-colors"
          :class="view === opt ? 'bg-primary text-primary-foreground' : 'text-muted hover:bg-overlay'"
          @click="view = opt"
        >
          {{ opt }}
        </button>
      </div>

      <!-- Bubbles: circle area ∝ topics covered (#3). -->
      <div
        v-if="view === 'bubbles'"
        class="flex max-h-72 flex-wrap items-center gap-2 overflow-y-auto pr-1"
        role="list"
        aria-label="Topic clusters sized by topics covered"
        data-testid="topic-landscape-bubbles"
      >
        <button
          v-for="b in bubbles"
          :key="b.key"
          type="button"
          role="listitem"
          class="flex shrink-0 items-center justify-center rounded-full border border-primary/40 bg-primary/10 text-center transition hover:bg-primary/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          :style="{ width: `${b.d}px`, height: `${b.d}px` }"
          :title="`${clusterDisplayLabel(b.cl)} — ${b.n} topics`"
          :aria-label="`${clusterDisplayLabel(b.cl)}, ${b.n} topics`"
          @click="onClusterActivate(b.cl)"
        >
          <span class="pointer-events-none flex max-w-full flex-col items-center px-1 leading-tight">
            <span
              v-if="b.d >= 72"
              class="max-w-full truncate text-[10px] font-semibold"
            >{{ clusterDisplayLabel(b.cl) }}</span>
            <span class="text-[11px] font-bold">{{ b.n }}</span>
          </span>
        </button>
      </div>

      <!-- Grid: the compact label + count cards. -->
      <div
        v-else
        class="max-h-72 overflow-y-auto pr-1 grid gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr))]"
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
    </template>
    <p
      v-if="insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ insight }}
    </p>
  </section>
</template>
