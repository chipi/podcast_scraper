<script setup lang="ts">
/**
 * Result-set operator bar — Search v3 §S4a (RFC-107 §7.4). Sits between
 * the results-count row and the hit cards. Four operator chips:
 *
 *   * **Cluster**  — groups the hit set (server-side; S4b, disabled here).
 *   * **Timeline** — client-only histogram of hits by publish month,
 *                    rendered inline below the bar.
 *   * **On graph** — pins every hit's episode/topic/entity to the graph
 *                    canvas as ``search-hit`` highlights + a set bbox,
 *                    then switches to the Graph tab.
 *   * **Consensus** — cross-speaker corroboration pairs (server-side; S4b,
 *                     disabled here).
 *
 * S4a scope: bar shell + Timeline + On-graph. S4b lands Cluster + Consensus
 * (server aggregation via ``operator=cluster`` / ``operator=consensus`` on
 * ``/api/search``); both chips render an honest "Coming in S4b" tooltip
 * today. The bar reads exclusively from the current ``search.results``
 * (or the caller's ``visible-hits`` prop) — it does NOT re-fetch.
 */
import { computed } from 'vue'
import type {
  CompareSubjectRef,
  SearchClusterGroup,
  SearchCompareResponse,
  SearchConsensusPair,
  SearchHit,
} from '../../api/searchApi'
import type { SubjectMentionsTimeline } from '../../utils/subjectMentionsTimeline'
import SubjectTimelineChart from '../subject/SubjectTimelineChart.vue'
import CompareOperatorPanel from './CompareOperatorPanel.vue'

const props = defineProps<{
  /**
   * Hit set the operator applies to. Callers typically pass ``visibleResults``
   * (the filtered set after the evidence-tier toggle) so the operator's view
   * matches what the user sees below.
   */
  visibleHits: SearchHit[]
  /**
   * Search v3 §S4b — the server's most-recent cluster response for the
   * current query, or null. Rendered inline in the Cluster panel.
   */
  clusters?: SearchClusterGroup[] | null
  /**
   * Search v3 §S4b — the server's most-recent consensus response for the
   * current query, or null. Rendered inline in the Consensus panel.
   */
  consensusPairs?: SearchConsensusPair[] | null
  /**
   * Search v3 §S4b / §S8 — which operator is currently mid-flight, if any.
   * The bar renders a small "…" affordance beside the active chip.
   */
  operatorLoading?: 'cluster' | 'consensus' | 'compare' | null
  /**
   * Search v3 §S4b — the last operator-fetch error (mapped human string),
   * or null.
   */
  operatorError?: string | null
  /**
   * Search v3 §S8 — Compare state. Panel-level state stays here so the
   * bar (as the operator surface) knows when to enable / disable the
   * Compare chip and can pass the response through to the panel.
   */
  compareResult?: SearchCompareResponse | null
  compareLoading?: boolean
  compareError?: string | null
}>()

const emit = defineEmits<{
  /**
   * On-graph: request the App host focus every hit's canonical id on the
   * graph canvas as a ``search-hit`` highlight set, then switch to the
   * Graph main tab. Payload carries the raw episode / node ids ready for
   * ``graphNavigation.setLibraryEpisodeHighlights``.
   */
  'focus-set': [ids: string[]]
  /**
   * Search v3 §S4b — request the server-side Cluster operator over the
   * current query. Parent triggers ``search.runOperator('cluster')``.
   */
  'run-cluster': []
  /**
   * Search v3 §S4b — request the server-side Consensus operator over the
   * current query. Parent triggers ``search.runOperator('consensus')``.
   */
  'run-consensus': []
  /**
   * Search v3 §S8 — request a compare against 2 picker-selected
   * subjects (from the current visible hits). Parent triggers
   * ``search.runCompare(subjectA, subjectB)``.
   */
  'run-compare': [payload: { subjectA: CompareSubjectRef; subjectB: CompareSubjectRef }]
  /** Parent triggers ``search.clearCompare()``. */
  'clear-compare': []
}>()

type OperatorId = 'cluster' | 'timeline' | 'graph' | 'consensus' | 'compare'

const active = defineModel<OperatorId | null>('active', { default: null })

const clusterActive = computed(() => active.value === 'cluster')
const timelineActive = computed(() => active.value === 'timeline')
const consensusActive = computed(() => active.value === 'consensus')
const compareActive = computed(() => active.value === 'compare')

/**
 * Search v3 §S8 — the Compare chip enables when at least 2 distinct
 * comparable subjects appear across the visible hits' metadata (persons,
 * topics, episodes, feeds). Mirrors the discovery walk in
 * ``CompareOperatorPanel``; kept in sync so the chip disables in the same
 * state the picker would render "no candidates".
 */
const compareCandidateCount = computed<number>(() => {
  const seen = new Set<string>()
  const bump = (kind: string, id: string): void => {
    const trimmed = id.trim()
    if (!trimmed) return
    seen.add(`${kind}::${trimmed}`)
  }
  for (const hit of props.visibleHits) {
    const md = (hit.metadata ?? {}) as Record<string, unknown>
    const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
    if (docType === 'kg_topic') {
      const src = typeof md.source_id === 'string' ? md.source_id : ''
      bump('topic', src)
    } else {
      const topicLabel = typeof md.topic_label === 'string' ? md.topic_label : ''
      if (topicLabel) bump('topic', topicLabel)
    }
    const speaker = typeof md.speaker_name === 'string'
      ? md.speaker_name
      : typeof md.speaker === 'string'
        ? md.speaker
        : ''
    if (speaker) bump('person', speaker)
    const supporting = Array.isArray(md.supporting_quotes) ? md.supporting_quotes : []
    for (const raw of supporting) {
      const q = (raw ?? {}) as Record<string, unknown>
      const name = typeof q.speaker_name === 'string' ? q.speaker_name : ''
      if (name) bump('person', name)
    }
    const epId = typeof md.episode_id === 'string' ? md.episode_id : ''
    if (epId) bump('episode', epId)
    const feedId = typeof md.feed_id === 'string' ? md.feed_id : ''
    if (feedId) bump('feed', feedId)
  }
  return seen.size
})

const compareChipDisabled = computed(() => compareCandidateCount.value < 2)

// ---------- Timeline (client-only) ---------------------------------------

/**
 * YYYY-MM buckets built from ``metadata.publish_date`` (ISO ``YYYY-MM-DD``).
 * Hits with no date bucket into the undated tally. We reuse the shipped
 * ``SubjectTimelineChart`` (bar variant) rather than authoring a new chart
 * — the timeline shape lets us plug straight in.
 */
const timeline = computed<SubjectMentionsTimeline>(() => {
  const monthCount = new Map<string, number>()
  let undated = 0
  const episodeIds = new Set<string>()

  for (const hit of props.visibleHits) {
    const md = (hit.metadata ?? {}) as Record<string, unknown>
    const ep = md.episode_id
    if (typeof ep === 'string' && ep.trim()) episodeIds.add(ep.trim())
    const raw = md.publish_date
    if (typeof raw !== 'string' || !raw.trim()) {
      undated += 1
      continue
    }
    const ymd = raw.slice(0, 7) // YYYY-MM
    if (!/^\d{4}-\d{2}$/.test(ymd)) {
      undated += 1
      continue
    }
    monthCount.set(ymd, (monthCount.get(ymd) ?? 0) + 1)
  }
  const months = Array.from(monthCount.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([ymd, count]) => ({ ymd, count }))
  const total = months.reduce((acc, m) => acc + m.count, 0)
  return {
    months,
    total,
    undated,
    episodeCount: episodeIds.size,
    insightIds: [],
    quoteIds: [],
  }
})

// ---------- On-graph (set focus) -----------------------------------------

/**
 * Canonical id per hit, preferring topic / entity source ids so the graph
 * highlights the hit's semantic anchor rather than always resolving to the
 * episode. Falls back to the episode_id when no source id is meaningful.
 */
function graphIdForHit(hit: SearchHit): string | null {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
  const src = typeof md.source_id === 'string' ? md.source_id.trim() : ''
  if (src && (docType === 'kg_topic' || docType === 'kg_entity')) return src
  const ep = typeof md.episode_id === 'string' ? md.episode_id.trim() : ''
  return ep || null
}

const graphSetIds = computed<string[]>(() => {
  const ids: string[] = []
  const seen = new Set<string>()
  for (const hit of props.visibleHits) {
    const id = graphIdForHit(hit)
    if (id && !seen.has(id)) {
      seen.add(id)
      ids.push(id)
    }
  }
  return ids
})

const graphChipDisabled = computed(() => graphSetIds.value.length === 0)
const graphChipLabel = computed(() =>
  graphChipDisabled.value ? 'On graph (no ids)' : `On graph (${graphSetIds.value.length})`,
)

function onClusterClick(): void {
  if (clusterActive.value) {
    active.value = null
    return
  }
  active.value = 'cluster'
  emit('run-cluster')
}

function onTimelineClick(): void {
  active.value = timelineActive.value ? null : 'timeline'
}

function onGraphClick(): void {
  if (graphChipDisabled.value) return
  active.value = 'graph'
  emit('focus-set', graphSetIds.value)
}

function onConsensusClick(): void {
  if (consensusActive.value) {
    active.value = null
    return
  }
  active.value = 'consensus'
  emit('run-consensus')
}

function onCompareClick(): void {
  if (compareChipDisabled.value) return
  active.value = compareActive.value ? null : 'compare'
}

function onCompareRun(payload: {
  subjectA: CompareSubjectRef
  subjectB: CompareSubjectRef
}): void {
  emit('run-compare', payload)
}

function onCompareClear(): void {
  emit('clear-compare')
}
</script>

<template>
  <div
    class="flex flex-col gap-2"
    data-testid="result-set-operator-bar"
    aria-label="Result-set operators"
  >
    <div
      class="flex flex-wrap items-center gap-1.5"
      role="group"
      aria-label="Operator chips"
    >
      <button
        type="button"
        class="rounded border px-2 py-0.5 text-[10px] font-medium leading-none transition-colors"
        :class="
          clusterActive
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="clusterActive"
        data-testid="operator-chip-cluster"
        title="Group hits by topic / theme cluster (server-side; over-fetches top_k × 3 for grouping)."
        @click="onClusterClick"
      >
        {{ operatorLoading === 'cluster' ? 'Cluster…' : 'Cluster' }}
      </button>
      <button
        type="button"
        class="rounded border px-2 py-0.5 text-[10px] font-medium leading-none transition-colors"
        :class="
          timelineActive
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="timelineActive"
        data-testid="operator-chip-timeline"
        title="Show a mentions-by-month histogram of the current hit set."
        @click="onTimelineClick"
      >
        Timeline
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] font-medium leading-none text-muted transition-colors hover:bg-overlay disabled:cursor-not-allowed disabled:opacity-40"
        :disabled="graphChipDisabled"
        data-testid="operator-chip-graph"
        :title="
          graphChipDisabled
            ? 'On graph: no hits resolve to a graph id (episode / topic / entity).'
            : `Pin every hit's canonical id (episode / topic / entity) on the graph canvas and switch to the Graph tab. ${graphSetIds.length} distinct ids.`
        "
        @click="onGraphClick"
      >
        {{ graphChipLabel }}
      </button>
      <button
        type="button"
        class="rounded border px-2 py-0.5 text-[10px] font-medium leading-none transition-colors"
        :class="
          consensusActive
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="consensusActive"
        data-testid="operator-chip-consensus"
        title="Cross-speaker corroboration pairs from enrichments/topic_consensus.json (ADR-108, precision ~0.91 on prod-v2)."
        @click="onConsensusClick"
      >
        {{ operatorLoading === 'consensus' ? 'Consensus…' : 'Consensus' }}
      </button>
      <button
        type="button"
        class="rounded border px-2 py-0.5 text-[10px] font-medium leading-none transition-colors disabled:cursor-not-allowed disabled:opacity-40"
        :class="
          compareActive
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="compareActive"
        :disabled="compareChipDisabled"
        data-testid="operator-chip-compare"
        :title="
          compareChipDisabled
            ? 'Compare: fewer than 2 comparable subjects in the current hit set.'
            : `Compare two subjects (person / topic / episode / feed) from the current hit set — server wraps build_briefing_pack twice.`
        "
        @click="onCompareClick"
      >
        {{ operatorLoading === 'compare' ? 'Compare…' : 'Compare' }}
      </button>
    </div>

    <p
      v-if="operatorError"
      class="text-[10px] text-danger"
      data-testid="operator-error"
    >
      {{ operatorError }}
    </p>

    <div
      v-if="timelineActive"
      class="rounded border border-border bg-canvas p-2"
      data-testid="operator-timeline-panel"
      aria-label="Timeline of the current hit set"
    >
      <SubjectTimelineChart
        :timeline="timeline"
        variant="dots"
        value-label="Hits"
        empty-text="No dated publish months in the current hit set."
        aria-label="Hits by publish month"
      />
      <p
        v-if="timeline.undated > 0"
        class="mt-1 text-[10px] text-muted"
        data-testid="operator-timeline-undated"
      >
        {{ timeline.undated }}
        {{ timeline.undated === 1 ? 'hit' : 'hits' }} without a publish date not shown.
      </p>
    </div>

    <div
      v-if="clusterActive"
      class="rounded border border-border bg-canvas p-2"
      data-testid="operator-cluster-panel"
      aria-label="Cluster grouping of the current hit set"
    >
      <p
        v-if="operatorLoading === 'cluster' && !clusters"
        class="text-[10px] text-muted"
        data-testid="operator-cluster-loading"
      >
        Loading clusters…
      </p>
      <p
        v-else-if="!clusters || !clusters.length"
        class="text-[10px] text-muted"
        data-testid="operator-cluster-empty"
      >
        No clusters — no hit resolves to a topic or theme cluster surface.
      </p>
      <ul
        v-else
        class="flex flex-col gap-1"
        data-testid="operator-cluster-list"
      >
        <li
          v-for="c in clusters"
          :key="c.cluster_id ?? 'ungrouped'"
          class="rounded border border-border/60 bg-surface px-2 py-1 text-[11px] text-surface-foreground"
        >
          <div class="flex items-center gap-2">
            <span
              class="rounded px-1 py-px text-[9px] font-medium uppercase leading-none tracking-wide text-muted"
              :class="c.cluster_kind === 'ungrouped' ? 'bg-overlay' : 'bg-primary/15 text-primary'"
              :title="`Cluster kind: ${c.cluster_kind}`"
            >
              {{ c.cluster_kind === 'ungrouped' ? 'Other' : c.cluster_kind.replace('_', ' ') }}
            </span>
            <span class="truncate font-medium">{{ c.label }}</span>
            <span class="ml-auto shrink-0 text-[10px] text-muted">
              {{ c.size }} {{ c.size === 1 ? 'hit' : 'hits' }}
            </span>
          </div>
        </li>
      </ul>
    </div>

    <div
      v-if="consensusActive"
      class="rounded border border-border bg-canvas p-2"
      data-testid="operator-consensus-panel"
      aria-label="Cross-speaker corroboration pairs"
    >
      <p
        v-if="operatorLoading === 'consensus' && !consensusPairs"
        class="text-[10px] text-muted"
        data-testid="operator-consensus-loading"
      >
        Loading consensus pairs…
      </p>
      <p
        v-else-if="!consensusPairs || !consensusPairs.length"
        class="text-[10px] text-muted"
        data-testid="operator-consensus-empty"
      >
        No corroboration pairs for topics in this hit set (or the corpus has no
        <code>enrichments/topic_consensus.json</code> yet).
      </p>
      <ul
        v-else
        class="flex flex-col gap-1.5"
        data-testid="operator-consensus-list"
      >
        <li
          v-for="p in consensusPairs"
          :key="`${p.topic_id}-${p.insight_a_id}-${p.insight_b_id}`"
          class="rounded border border-border/60 bg-surface px-2 py-1.5 text-[11px] text-surface-foreground"
        >
          <p class="mb-1 flex items-center gap-1.5">
            <span
              class="rounded bg-primary/15 px-1 py-px text-[9px] font-medium uppercase leading-none tracking-wide text-primary"
            >Topic</span>
            <span class="truncate font-medium">{{ p.topic_label ?? p.topic_id }}</span>
          </p>
          <p class="mb-1 line-clamp-2 italic">
            <span class="font-medium">{{ p.person_a_label ?? p.person_a_id }}:</span>
            {{ p.insight_a_text || '(no text)' }}
          </p>
          <p class="mb-1 line-clamp-2 italic">
            <span class="font-medium">{{ p.person_b_label ?? p.person_b_id }}:</span>
            {{ p.insight_b_text || '(no text)' }}
          </p>
          <p class="text-[10px] text-muted">
            <span title="Lower is stronger agreement (ADR-108)">
              contradiction: {{ p.contradiction_score.toFixed(2) }}
            </span>
            <span
              v-if="p.cosine_similarity != null"
              title="Higher is same-question (embedding cosine)"
              class="ml-2"
            >
              cosine: {{ p.cosine_similarity.toFixed(2) }}
            </span>
          </p>
        </li>
      </ul>
    </div>

    <CompareOperatorPanel
      v-if="compareActive"
      :visible-hits="visibleHits"
      :compare-result="compareResult ?? null"
      :compare-loading="Boolean(compareLoading)"
      :compare-error="compareError ?? null"
      @run-compare="onCompareRun"
      @clear-compare="onCompareClear"
    />
  </div>
</template>
