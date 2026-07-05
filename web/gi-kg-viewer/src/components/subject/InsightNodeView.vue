<script setup lang="ts">
/**
 * InsightNodeView — an insight's OWN content, resolved from
 * /api/relational/insight-detail. Lets an out-of-slice insight (e.g. a
 * corpus-wide timeline-mention drill) render its text + supporting quotes +
 * ABOUT topics + MENTIONS entities instead of an empty "Node" rail — the same
 * out-of-slice pattern as PodcastNodeView. Topics/entities are click-through
 * (focusTopic / focusEntity). The resolved text is emitted up so the rail
 * header shows the claim, not the opaque insight: hash.
 */
import { computed, ref, watch } from 'vue'
import {
  fetchInsightDetail,
  type InsightDetailResponse,
  type RelatedNode,
} from '../../api/relationalApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { stripLayerPrefixesForCil } from '../../utils/mergeGiKg'
import { StaleGeneration } from '../../utils/staleGeneration'

const props = withDefaults(defineProps<{ subjectIdOverride?: string }>(), {
  subjectIdOverride: '',
})
const emit = defineEmits<{ resolved: [string] }>()

const shell = useShellStore()
const subject = useSubjectStore()

const insightId = computed(() =>
  stripLayerPrefixesForCil(props.subjectIdOverride?.trim() || subject.graphNodeCyId?.trim() || ''),
)

const detail = ref<InsightDetailResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const gate = new StaleGeneration()

async function load(id: string): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!id || !root || !shell.healthStatus) {
    detail.value = null
    return
  }
  const seq = gate.bump()
  loading.value = true
  error.value = null
  detail.value = null
  try {
    const res = await fetchInsightDetail(root, id)
    if (gate.isStale(seq)) return
    if (res.error) {
      error.value = res.error
      return
    }
    detail.value = res
    emit('resolved', res.text ?? '')
  } catch (e) {
    if (gate.isStale(seq)) return
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gate.isCurrent(seq)) loading.value = false
  }
}
watch(insightId, (id) => void load(id), { immediate: true })

const quotes = computed<RelatedNode[]>(() => detail.value?.quotes ?? [])
const topics = computed<RelatedNode[]>(() => detail.value?.topics ?? [])
const entities = computed<RelatedNode[]>(() => detail.value?.entities ?? [])

function shortId(id: string): string {
  return id.replace(/^(?:person|topic|org):/, '').replace(/[-_]+/g, ' ').trim() || id
}
</script>

<template>
  <div class="min-w-0" data-testid="insight-node-view">
    <p v-if="loading" class="text-[11px] text-muted" data-testid="insight-node-loading">Loading…</p>
    <p v-else-if="error" class="text-[11px] text-warning">{{ error }}</p>
    <template v-else-if="detail">
      <!-- The claim itself -->
      <p
        class="break-words text-sm leading-snug text-surface-foreground"
        data-testid="insight-node-text"
      >
        {{ detail.text }}
      </p>
      <div v-if="detail.insight_type || detail.grounded" class="mt-1 flex flex-wrap items-center gap-1.5">
        <span
          v-if="detail.insight_type"
          class="rounded bg-overlay px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-muted"
          >{{ detail.insight_type }}</span
        >
        <span
          v-if="detail.grounded"
          class="rounded bg-gi/15 px-1.5 py-0.5 text-[9px] font-semibold text-gi"
          >Grounded</span
        >
      </div>

      <section v-if="topics.length" class="mt-3" data-testid="insight-node-topics">
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">About</h3>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="t in topics"
            :key="t.id"
            type="button"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs text-primary transition hover:bg-elevated"
            @click="subject.focusTopic(t.id)"
          >
            {{ t.text || shortId(t.id) }}
          </button>
        </div>
      </section>

      <section v-if="entities.length" class="mt-3" data-testid="insight-node-entities">
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Mentions</h3>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="e in entities"
            :key="e.id"
            type="button"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs text-primary transition hover:bg-elevated"
            @click="subject.focusEntity(e.id)"
          >
            {{ e.text || shortId(e.id) }}
          </button>
        </div>
      </section>

      <section v-if="quotes.length" class="mt-3" data-testid="insight-node-quotes">
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Supporting quotes
        </h3>
        <ul class="space-y-1.5">
          <li
            v-for="q in quotes"
            :key="q.id"
            class="rounded border border-border bg-elevated/40 px-2 py-1 text-[11px] leading-snug text-surface-foreground"
          >
            “{{ q.text }}”
          </li>
        </ul>
      </section>
    </template>
  </div>
</template>
