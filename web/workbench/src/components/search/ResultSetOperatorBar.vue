<script setup lang="ts">
/**
 * Result-set operator bar — Search v3 §S4a (RFC-107 §7.4). Sits between
 * the results-count row and the hit cards. Three operator chips:
 *
 *   * **Timeline** — client-only histogram of hits by publish month,
 *                    rendered inline below the bar.
 *   * **On graph** — pins every hit's episode/topic/entity to the graph
 *                    canvas as ``search-hit`` highlights + a set bbox,
 *                    then switches to the Graph tab.
 *
 * The server-side Cluster and Consensus operators are private features (ADR-170) and are not part
 * of this viewer. The bar reads exclusively from the current ``search.results`` (or the caller's
 * ``visible-hits`` prop) — it does NOT re-fetch.
 */
import { computed } from 'vue'
import { searchOperators, type SearchOperator } from '../../extensions'
import type { SearchHit } from '../../api/searchApi'
import type { SubjectMentionsTimeline } from '../../utils/subjectMentionsTimeline'
import SubjectTimelineChart from '../subject/SubjectTimelineChart.vue'

const props = defineProps<{
  /**
   * Hit set the operator applies to. Callers typically pass ``visibleResults``
   * (the filtered set after the evidence-tier toggle) so the operator's view
   * matches what the user sees below.
   */
  visibleHits: SearchHit[]
}>()

const emit = defineEmits<{
  /**
   * On-graph: request the App host focus every hit's canonical id on the
   * graph canvas as a ``search-hit`` highlight set, then switch to the
   * Graph main tab. Payload carries the raw episode / node ids ready for
   * ``graphNavigation.setLibraryEpisodeHighlights``.
   */
  'focus-set': [ids: string[]]
}>()

/** The viewer's own chips, or the id of a registered operator (extension point `searchOperators`). */
const active = defineModel<string | null>('active', { default: null })

const BUILT_IN_ORDER = { timeline: 10, graph: 20 } as const
/** Registered operators whose order falls in [from, to). */
function registeredBetween(from: number, to: number): SearchOperator[] {
  return searchOperators.list().filter((o) => (o.order ?? 0) >= from && (o.order ?? 0) < to)
}
const activeRegistered = computed(() => searchOperators.list().find((o) => o.id === active.value))
const registeredError = computed(() => {
  for (const o of searchOperators.list()) {
    const e = o.error?.()
    if (e) return e
  }
  return null
})

function opAvailable(op: SearchOperator): boolean {
  return op.available?.(props.visibleHits) ?? true
}

function onRegisteredClick(op: SearchOperator): void {
  if (!opAvailable(op)) return
  if (active.value === op.id) {
    active.value = null
    return
  }
  active.value = op.id
  op.activate?.()
}

const timelineActive = computed(() => active.value === 'timeline')

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

function onTimelineClick(): void {
  active.value = timelineActive.value ? null : 'timeline'
}

function onGraphClick(): void {
  if (graphChipDisabled.value) return
  active.value = 'graph'
  emit('focus-set', graphSetIds.value)
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
        v-for="op in registeredBetween(-Infinity, BUILT_IN_ORDER.timeline)"
        :key="op.id"
        type="button"
        class="rounded border px-2 py-0.5 text-[10px] font-medium leading-none transition-colors disabled:cursor-not-allowed disabled:opacity-40"
        :class="
          active === op.id
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="active === op.id"
        :data-testid="`operator-chip-${op.id}`"
        :disabled="!opAvailable(op)"
        :title="opAvailable(op) ? op.title : (op.unavailableTitle ?? op.title)"
        @click="onRegisteredClick(op)"
      >
        {{ op.busy?.() ? `${op.label}…` : op.label }}
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
        v-for="op in registeredBetween(BUILT_IN_ORDER.timeline, BUILT_IN_ORDER.graph)"
        :key="op.id"
        type="button"
        class="rounded border px-2 py-0.5 text-[10px] font-medium leading-none transition-colors disabled:cursor-not-allowed disabled:opacity-40"
        :class="
          active === op.id
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="active === op.id"
        :data-testid="`operator-chip-${op.id}`"
        :disabled="!opAvailable(op)"
        :title="opAvailable(op) ? op.title : (op.unavailableTitle ?? op.title)"
        @click="onRegisteredClick(op)"
      >
        {{ op.busy?.() ? `${op.label}…` : op.label }}
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
        v-for="op in registeredBetween(BUILT_IN_ORDER.graph, Infinity)"
        :key="op.id"
        type="button"
        class="rounded border px-2 py-0.5 text-[10px] font-medium leading-none transition-colors disabled:cursor-not-allowed disabled:opacity-40"
        :class="
          active === op.id
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="active === op.id"
        :data-testid="`operator-chip-${op.id}`"
        :disabled="!opAvailable(op)"
        :title="opAvailable(op) ? op.title : (op.unavailableTitle ?? op.title)"
        @click="onRegisteredClick(op)"
      >
        {{ op.busy?.() ? `${op.label}…` : op.label }}
      </button>
    </div>

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

    <p
      v-if="registeredError"
      class="text-[10px] text-danger"
      data-testid="operator-error"
    >
      {{ registeredError }}
    </p>

    <component
      :is="activeRegistered.panel"
      v-if="activeRegistered?.panel"
      :visible-hits="visibleHits"
    />

  </div>
</template>
