<script setup lang="ts">
import { computed, inject } from 'vue'
import { resultCardBadges, resultCardSections } from '../../extensions'
import type { SearchHit } from '../../api/searchApi'
import { corpusGraphBaselineLoaderKey } from '../../corpusGraphBaseline'
import SearchResultRowIcon from './SearchResultRowIcon.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { truncate } from '../../utils/formatting'
import {
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
  SEARCH_RESULT_GRAPH_BUTTON_CLASS,
} from '../../utils/searchResultActionStyles'
import { graphNodeIdFromSearchHit } from '../../utils/searchFocus'
import { isKgSurfaceMultiEpisodeDedupe } from '../../utils/searchHitKgDedupe'
import { sourceMetadataRelativePathFromSearchHit } from '../../utils/searchHitLibrary'
import { useSubjectStore } from '../../stores/subject'

const props = defineProps<{
  hit: SearchHit
  /** Corpus Library API + corpus path — required to offer episode handoff (**L**) in the subject rail. */
  libraryOpensEnabled: boolean
}>()

const emit = defineEmits<{
  focus: [SearchHit]
  /**
   * Row-body click (Search v3 §S4-shell followup) — opens the Episode
   * subject panel in the right rail. Fires when the user clicks anywhere
   * on the card body OUTSIDE the inner action buttons / links (those all
   * call ``.stopPropagation()``). Retired: the standalone **L** button.
   */
  'open-library': [SearchHit]
}>()

const subject = useSubjectStore()
const artifacts = useArtifactsStore()
const graphExplorer = useGraphExplorerStore()
const loadCorpusGraphBaseline = inject(corpusGraphBaselineLoaderKey, null)

async function ensureDefaultCorpusGraphIfNeeded(): Promise<void> {
  if (!loadCorpusGraphBaseline) return
  if (graphExplorer.graphTabOpenedThisSession && artifacts.selectedRelPaths.length > 0) {
    return
  }
  await loadCorpusGraphBaseline()
}

const docType = computed(() => String(props.hit.metadata?.doc_type ?? '?'))

/** PRD-033 FR1.1 — retrieval tier badge (insight / segment / aux). */
const sourceTier = computed(() => String(props.hit.source_tier ?? 'aux'))
const TIER_LABELS: Record<string, string> = {
  insight: 'Insight',
  segment: 'Transcript',
  aux: 'Reference',
}
const tierLabel = computed(() => TIER_LABELS[sourceTier.value] ?? 'Reference')

/** FR1.5 — a kg_entity / kg_topic hit links its canonical id to its Detail panel. */
const detailEntity = computed((): { id: string; kind: 'person' | 'topic' } | null => {
  const dt = docType.value
  const sourceId = props.hit.metadata?.source_id
  const id = typeof sourceId === 'string' ? sourceId.trim() : ''
  if (!id) return null
  if (dt === 'kg_topic' || id.startsWith('topic:')) return { id, kind: 'topic' }
  if (dt === 'kg_entity') return { id, kind: 'person' }
  return null
})

function focusDetailEntity(): void {
  const e = detailEntity.value
  if (!e) return
  if (e.kind === 'topic') subject.focusTopic(e.id)
  else subject.focusPerson(e.id)
  void ensureDefaultCorpusGraphIfNeeded()
}

/** One line when server joined ``topic_clusters.json`` for this ``kg_topic`` row. */
const topicClusterSummary = computed((): string | null => {
  const tc = props.hit.metadata?.topic_cluster
  if (tc == null || typeof tc !== 'object') return null
  const o = tc as Record<string, unknown>
  const label = o.canonical_label
  const gpid = o.graph_compound_parent_id
  const bits: string[] = []
  if (typeof label === 'string' && label.trim()) bits.push(label.trim())
  if (typeof gpid === 'string' && gpid.trim()) bits.push(gpid.trim())
  return bits.length ? bits.join(' · ') : null
})

const episodeId = computed(() => {
  const e = props.hit.metadata?.episode_id
  return typeof e === 'string' ? e : null
})

const focusable = computed(() => graphNodeIdFromSearchHit(props.hit) != null)

const kgMultiEpDedupe = computed(() => isKgSurfaceMultiEpisodeDedupe(props.hit))

const libraryMetaPath = computed(() => sourceMetadataRelativePathFromSearchHit(props.hit))

/**
 * Row is click-to-open when the hit has a default action:
 *   - kg_topic / kg_entity → open Topic / Person subject panel (from
 *     ``detailEntity`` — internally calls the subject store).
 *   - Otherwise → emit ``open-library`` to open the Episode subject
 *     panel (requires a metadata path on the hit).
 *
 * Retired: standalone **L** and **S** buttons AND the standalone "Open
 * Topic panel →" / "Open Person panel →" text link — the whole row body
 * is the affordance for both routes (Search v3 §S4-shell followup +
 * 2026-07-22 UX cleanup).
 */
const rowOpensDetailEntity = computed(() => detailEntity.value != null)
const rowOpensLibrary = computed(
  () =>
    props.libraryOpensEnabled &&
    libraryMetaPath.value != null &&
    !kgMultiEpDedupe.value,
)
const rowClickable = computed(
  () => rowOpensDetailEntity.value || rowOpensLibrary.value,
)
const rowAriaLabel = computed<string | undefined>(() => {
  if (rowOpensDetailEntity.value) {
    const e = detailEntity.value!
    return e.kind === 'topic' ? 'Open Topic panel' : 'Open Person panel'
  }
  if (rowOpensLibrary.value) return 'Open episode in subject panel'
  return undefined
})

const showEpisodeChip = computed(
  () => Boolean(episodeId.value) && !kgMultiEpDedupe.value,
)

const hasActions = computed(() => focusable.value)

const showRightChips = computed(() => hasActions.value || showEpisodeChip.value)

const graphButtonTooltip = computed((): string => {
  if (!focusable.value) {
    return 'Show on graph'
  }
  if (kgMultiEpDedupe.value) {
    const n = Number(props.hit.metadata?.kg_surface_match_count)
    const k = Number.isFinite(n) ? n : 2
    return (
      `Show on graph — merged across ${k} episodes; uses this row’s node id in the current graph. ` +
      'Turn off “Merge duplicate KG surfaces” in Advanced search to get L/E per episode.'
    )
  }
  return 'Show on graph'
})

const episodeIdTooltip = computed((): string | undefined => {
  const id = episodeId.value
  if (!id) return undefined
  return (
    `Episode id (corpus-stable, from metadata / vector index): ${id}. ` +
    'Same episode across chunks in this search; not sent for L (that uses the metadata file path).'
  )
})

/** Native title tooltip; same idea as digest topic-hit score (vector similarity). */
const SEARCH_HIT_SCORE_TOOLTIP =
  'Vector similarity from the semantic search index for your query (higher = closer match). ' +
  'Depends on the embedding model; use only to rank hits within this search, not across index rebuilds or models.'

function onGraphClick(ev: MouseEvent): void {
  ev.stopPropagation()
  if (!focusable.value) return
  emit('focus', props.hit)
}

function onEpisodeIdChipClick(ev: MouseEvent): void {
  ev.stopPropagation()
}

/**
 * Row-body click / keyboard activation. Inner buttons stop propagation, so
 * this only fires for background clicks (title bar chrome, body text, or
 * blank space). Keyboard: Enter / Space when the article is focused.
 *
 * Routing (2026-07-22 UX cleanup): kg_topic / kg_entity hits open the
 * Topic / Person subject panel directly; everything else emits
 * ``open-library`` (Episode panel via the metadata path). The standalone
 * "Open Topic panel →" / "Open Person panel →" text link was retired
 * because the whole row now IS that affordance.
 */
function activateRowDefaultAction(): void {
  if (rowOpensDetailEntity.value) {
    focusDetailEntity()
    return
  }
  if (rowOpensLibrary.value) {
    emit('open-library', props.hit)
  }
}

function onRowClick(): void {
  if (!rowClickable.value) return
  activateRowDefaultAction()
}

function onRowKeydown(ev: KeyboardEvent): void {
  if (!rowClickable.value) return
  if (ev.key !== 'Enter' && ev.key !== ' ') return
  if (ev.defaultPrevented) return
  ev.preventDefault()
  activateRowDefaultAction()
}
</script>

<template>
  <article
    class="rounded border border-border bg-elevated p-2 text-xs text-elevated-foreground"
    :class="rowClickable && 'cursor-pointer transition-colors hover:border-primary/50 hover:bg-overlay focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary'"
    :role="rowClickable ? 'button' : undefined"
    :tabindex="rowClickable ? 0 : undefined"
    :aria-label="rowAriaLabel"
    @click="onRowClick"
    @keydown="onRowKeydown"
  >
    <div class="mb-1 flex min-w-0 flex-wrap items-center gap-2">
      <SearchResultRowIcon
        :doc-type="docType"
        :source-tier="sourceTier"
        data-testid="search-result-tier"
        :data-tier="tierLabel"
        :aria-label="`${tierLabel} · ${docType}`"
      />
      <component
        :is="badge.component"
        v-for="badge in resultCardBadges.list()"
        :key="badge.id"
        :hit="hit"
      />
      <span
        class="cursor-help rounded bg-overlay px-1 py-px font-mono text-[9px] leading-none text-muted"
        :title="SEARCH_HIT_SCORE_TOOLTIP"
        :aria-label="`Similarity ${hit.score.toFixed(4)}. ${SEARCH_HIT_SCORE_TOOLTIP}`"
      >{{ hit.score.toFixed(4) }}</span>
      <div
        v-if="showRightChips"
        class="ml-auto flex shrink-0 items-center gap-1"
      >
        <button
          v-if="focusable"
          type="button"
          :class="SEARCH_RESULT_GRAPH_BUTTON_CLASS"
          :aria-label="graphButtonTooltip"
          :title="graphButtonTooltip"
          @click="onGraphClick"
        >
          G
        </button>
        <button
          v-if="showEpisodeChip"
          type="button"
          :class="SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS"
          :aria-label="episodeIdTooltip"
          :title="episodeIdTooltip"
          @click="onEpisodeIdChipClick"
        >
          E
        </button>
      </div>
    </div>
    <p
      v-if="topicClusterSummary"
      class="mt-1 text-[10px] leading-snug text-muted"
    >
      Topic cluster: {{ topicClusterSummary }}
    </p>
    <p class="leading-snug text-surface-foreground">
      {{ truncate(hit.text || '(no text)', 320) }}
    </p>
    <component
      :is="section.component"
      v-for="section in resultCardSections.list()"
      :key="section.id"
      :hit="hit"
    />
  </article>
</template>
