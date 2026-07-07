<script setup lang="ts">
/**
 * RFC-088 chunk-9 follow-up: graph-view companion panel listing
 * enrichment-layer edges (topic_similarity + topic_consensus).
 *
 * Cytoscape-integrating the new edges would require deep changes to
 * graph construction, dedup, and focus state. This panel keeps that
 * graph untouched and instead surfaces the edges as a scannable list
 * directly above/beside the canvas — every row is clickable and pivots
 * the subject store, which the rest of the viewer already follows.
 *
 * Two sections:
 *   - Topic similarity (top-K per topic, from topic_similarity.json)
 *   - Consensus (per-Topic cross-Person corroborating Insight pairs, from
 *     topic_consensus.json — the reimagined NLI enricher, ADR-108)
 *
 * Self-loading via the chunk-8 cache composable so re-mounting is
 * free.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import { useSubjectStore } from '../../stores/subject'
import { stripLayerPrefixesForCil } from '../../utils/mergeGiKg'

interface Props {
  corpusPath: string
}
const props = defineProps<Props>()

interface SimNeighbour {
  topic_id: string
  topic_label?: string
  similarity?: number
}
interface SimTopic {
  topic_id: string
  topic_label?: string
  top_k: SimNeighbour[]
}
interface SimData {
  topic_count?: number
  topics: SimTopic[]
}

interface Consensus {
  topic_id: string
  person_a_id: string
  person_b_id: string
  person_a_name?: string
  person_b_name?: string
  insight_a_id: string
  insight_b_id: string
  consensus_score: number
  model_id?: string
}
interface ConsensusData {
  consensus: Consensus[]
}

const similarity = ref<SimData | null>(null)
const consensus = ref<ConsensusData | null>(null)
const loaded = ref(false)
const error = ref<string | null>(null)

const subject = useSubjectStore()

const focusedTopicId = computed(() => {
  // Topics now focus as graph nodes (unified node view); derive the CIL topic id
  // from the cy id so the sim list still narrows to the focused topic.
  const gn = subject.graphNodeCyId?.trim()
  return gn ? stripLayerPrefixesForCil(gn) : ''
})
const focusedPersonId = computed(() => {
  // Persons now focus as graph nodes too; derive the CIL person id from the cy id
  // so the consensus list still narrows to the focused person.
  const gn = subject.graphNodeCyId?.trim()
  return gn ? stripLayerPrefixesForCil(gn) : ''
})

const SIM_TOP_N = 5
const CONSENSUS_TOP_N = 10

/**
 * Sim edges for the panel: when a topic is focused, narrow to that
 * topic's neighbours; otherwise show the highest-similarity edges
 * across the corpus (top SIM_TOP_N pairs by score).
 */
const visibleSimRows = computed<{ a: SimTopic; n: SimNeighbour }[]>(() => {
  const topics = similarity.value?.topics ?? []
  const focused = focusedTopicId.value
  if (focused) {
    const t = topics.find((x) => x.topic_id === focused)
    if (!t) return []
    return t.top_k.slice(0, SIM_TOP_N).map((n) => ({ a: t, n }))
  }
  // No focus → show top-N pairs by similarity across the whole corpus.
  const flat: { a: SimTopic; n: SimNeighbour }[] = []
  for (const t of topics) for (const n of t.top_k) flat.push({ a: t, n })
  flat.sort((x, y) => (y.n.similarity ?? 0) - (x.n.similarity ?? 0))
  return flat.slice(0, SIM_TOP_N)
})

/**
 * Consensus for the panel: when a person is focused, narrow to
 * corroborations involving them; otherwise show the strongest globally.
 */
const visibleConsensus = computed<Consensus[]>(() => {
  const rows = consensus.value?.consensus ?? []
  const focused = focusedPersonId.value
  let filtered = rows
  if (focused) {
    filtered = rows.filter(
      (r) => r.person_a_id === focused || r.person_b_id === focused,
    )
  }
  return filtered
    .slice()
    .sort((x, y) => y.consensus_score - x.consensus_score)
    .slice(0, CONSENSUS_TOP_N)
})

const hasAny = computed(
  () => visibleSimRows.value.length > 0 || visibleConsensus.value.length > 0,
)

// Collapsed by default so the panel doesn't bury the graph canvas; a one-line
// header toggles it, and when open the content is height-capped + scrollable.
const expanded = ref(false)
const summary = computed(() => {
  const parts: string[] = []
  if (visibleSimRows.value.length) parts.push(`${visibleSimRows.value.length} similar`)
  const c = visibleConsensus.value.length
  if (c) parts.push(`${c} consensus`)
  return parts.join(' · ')
})

async function load(): Promise<void> {
  const root = props.corpusPath?.trim()
  if (!root) {
    similarity.value = null
    consensus.value = null
    loaded.value = true
    return
  }
  error.value = null
  try {
    const [sim, con] = await Promise.all([
      fetchCachedCorpusEnvelope<SimData>(root, 'topic_similarity').catch(() => null),
      fetchCachedCorpusEnvelope<ConsensusData>(root, 'topic_consensus').catch(() => null),
    ])
    similarity.value = sim?.data ?? null
    consensus.value = con?.data ?? null
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : String(exc)
  } finally {
    loaded.value = true
  }
}

onMounted(load)
watch(() => props.corpusPath, () => void load())
</script>

<template>
  <section
    v-if="loaded && hasAny"
    class="rounded border border-default bg-overlay/40 p-2 text-[11px]"
    aria-label="Enrichment-layer edges"
    data-testid="enrichment-edges-panel"
  >
    <div class="flex items-center justify-between gap-2">
      <button
        type="button"
        class="flex min-w-0 items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted hover:text-surface-foreground"
        :aria-expanded="expanded"
        data-testid="enrichment-edges-toggle"
        @click="expanded = !expanded"
      >
        <span
          aria-hidden="true"
          class="inline-block transition-transform"
          :class="expanded ? 'rotate-90' : ''"
        >▸</span>
        <span>Enrichment edges</span>
        <span
          v-if="summary"
          class="truncate font-normal normal-case text-muted/70"
        >· {{ summary }}</span>
      </button>
      <button
        v-if="(focusedTopicId || focusedPersonId) && expanded"
        type="button"
        class="shrink-0 text-[9px] text-muted hover:underline"
        title="Clear subject focus"
        @click="subject.clearSubject?.()"
      >
        clear focus
      </button>
    </div>

    <div
      v-show="expanded"
      class="mt-1 max-h-[45vh] overflow-y-auto pr-0.5"
      data-testid="enrichment-edges-body"
    >
    <!-- Topic similarity edges -->
    <div
      v-if="visibleSimRows.length"
      class="mb-2"
      data-testid="enrichment-edges-similarity"
    >
      <p class="mb-1 text-[10px] text-muted">
        Topic similarity
        <span v-if="focusedTopicId">· neighbours of focused topic</span>
        <span v-else>· top pairs corpus-wide</span>
      </p>
      <ul class="flex flex-col gap-1">
        <li
          v-for="row in visibleSimRows"
          :key="`${row.a.topic_id}::${row.n.topic_id}`"
          class="flex items-center gap-2 rounded border border-default bg-overlay px-2 py-0.5"
          :data-testid="`enrichment-edges-sim-${row.a.topic_id}--${row.n.topic_id}`"
        >
          <button
            type="button"
            class="font-mono hover:underline"
            @click="subject.focusTopic(row.a.topic_id)"
          >{{ row.a.topic_label || row.a.topic_id }}</button>
          <span class="text-muted">~</span>
          <button
            type="button"
            class="font-mono hover:underline"
            @click="subject.focusTopic(row.n.topic_id)"
          >{{ row.n.topic_label || row.n.topic_id }}</button>
          <span
            v-if="row.n.similarity != null"
            class="ml-auto rounded bg-overlay-2 px-1 text-[9px] text-muted"
            :title="`Cosine similarity ${row.n.similarity.toFixed(4)}`"
          >{{ row.n.similarity.toFixed(2) }}</span>
        </li>
      </ul>
    </div>

    <!-- Consensus edges -->
    <div v-if="visibleConsensus.length" data-testid="enrichment-edges-consensus">
      <p class="mb-1 text-[10px] text-muted">
        Consensus
        <span v-if="focusedPersonId">· involving focused person</span>
        <span v-else>· strongest corpus-wide</span>
      </p>
      <ul class="flex flex-col gap-1">
        <li
          v-for="row in visibleConsensus"
          :key="`${row.insight_a_id}::${row.insight_b_id}`"
          class="flex items-center gap-2 rounded border border-emerald-700/50 bg-emerald-900/20 px-2 py-0.5"
          :data-testid="`enrichment-edges-consensus-${row.insight_a_id}--${row.insight_b_id}`"
        >
          <button
            type="button"
            class="font-mono hover:underline"
            @click="subject.focusPerson(row.person_a_id)"
          >{{ row.person_a_name || row.person_a_id }}</button>
          <span class="text-emerald-300">🤝</span>
          <button
            type="button"
            class="font-mono hover:underline"
            @click="subject.focusPerson(row.person_b_id)"
          >{{ row.person_b_name || row.person_b_id }}</button>
          <span class="text-muted">on</span>
          <button
            type="button"
            class="font-mono hover:underline"
            @click="subject.focusTopic(row.topic_id)"
          >{{ row.topic_id }}</button>
          <span
            class="ml-auto rounded bg-emerald-800/30 px-1 text-[9px] text-emerald-200"
            :title="`Consensus score ${row.consensus_score.toFixed(4)} · ${row.model_id ?? ''}`"
          >{{ row.consensus_score.toFixed(2) }}</span>
        </li>
      </ul>
    </div>
    <p
      v-if="error"
      class="mt-1 text-[9px] text-rose-300"
      data-testid="enrichment-edges-error"
    >
      {{ error }}
    </p>
    </div>
  </section>
</template>
