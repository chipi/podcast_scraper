<script setup lang="ts">
import { computed, inject, ref } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import { corpusGraphBaselineLoaderKey } from '../../corpusGraphBaseline'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { truncate } from '../../utils/formatting'
import {
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
  SEARCH_RESULT_GRAPH_BUTTON_CLASS,
} from '../../utils/searchResultActionStyles'
import { graphNodeIdFromSearchHit } from '../../utils/searchFocus'
import { quoteAttributionDisplayFromId } from '../../utils/parsing'
import {
  GI_QUOTE_SPEAKER_UNAVAILABLE_HINT,
  liftedQuotePayloadHasUsableTiming,
  SEARCH_LIFTED_QUOTE_SPEAKER_UNAVAILABLE_TESTID,
  SUPPORTING_QUOTE_SPEAKER_UNAVAILABLE_TESTID,
} from '../../utils/transcriptSourceDisplay'
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

/** #674 item 4 — Supporting-quote speaker name → Person Landing in the rail.
 *  Auto-loads the corpus graph baseline so the panel has data even if the
 *  user hasn't visited Graph yet. */
function focusPersonFromSpeakerId(rawId: unknown): void {
  const id = typeof rawId === 'string' ? rawId.trim() : ''
  if (!id) return
  subject.focusPerson(id)
  void ensureDefaultCorpusGraphIfNeeded()
}

const docType = computed(() => String(props.hit.metadata?.doc_type ?? '?'))

/**
 * RFC-088 chunk-5 follow-up: related topics decorated onto the hit by
 * the QueryEnricher chain when /api/search?enrich_results=true. Shape:
 * metadata.query_enrichments.related_topics = [
 *   { topic_id, topic_label, similarity }
 * ]
 */
interface RelatedTopicChip {
  topic_id: string
  topic_label?: string
  similarity?: number
}
const relatedTopics = computed<RelatedTopicChip[]>(() => {
  const qe = (props.hit.metadata as Record<string, unknown> | undefined)?.query_enrichments
  if (!qe || typeof qe !== 'object') return []
  const rt = (qe as Record<string, unknown>).related_topics
  if (!Array.isArray(rt)) return []
  return rt
    .filter((r): r is Record<string, unknown> => r != null && typeof r === 'object')
    .map((r) => ({
      topic_id: String(r.topic_id ?? ''),
      topic_label: typeof r.topic_label === 'string' ? r.topic_label : undefined,
      similarity: typeof r.similarity === 'number' ? r.similarity : undefined,
    }))
    .filter((r) => r.topic_id)
})

function focusRelatedTopic(topicId: string): void {
  if (topicId) subject.focusTopic(topicId)
}

/** PRD-033 FR1.1 — retrieval tier badge (insight / segment / aux). */
const sourceTier = computed(() => String(props.hit.source_tier ?? 'aux'))
const TIER_LABELS: Record<string, string> = {
  insight: 'Insight',
  segment: 'Transcript',
  aux: 'Reference',
}
const tierLabel = computed(() => TIER_LABELS[sourceTier.value] ?? 'Reference')
const tierClass = computed(() => {
  switch (sourceTier.value) {
    case 'insight':
      return 'bg-primary/15 text-primary'
    case 'segment':
      return 'bg-success/15 text-success'
    default:
      return 'bg-overlay text-muted'
  }
})

/**
 * PRD-033 FR1.5 — when a transcript hit lifts a linked insight, the lifted insight
 * also exposes the canonical id of *both* the segment hit and the insight: the card
 * already shows both, making it a compound (insight + grounding segment) result.
 */
const isCompound = computed(
  () =>
    sourceTier.value === 'segment' &&
    props.hit.lifted != null &&
    typeof props.hit.lifted === 'object',
)

function liftedEntityId(block: unknown): string {
  if (block == null || typeof block !== 'object') return ''
  const id = (block as Record<string, unknown>).id
  return typeof id === 'string' ? id.trim() : ''
}

/** FR1.5 — lifted topic name → Topic Entity View in the rail. */
function focusTopicFromLifted(block: unknown): void {
  const id = liftedEntityId(block)
  if (!id) return
  subject.focusTopic(id)
  void ensureDefaultCorpusGraphIfNeeded()
}

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
 * Row is click-to-open (right-rail Episode panel) when the hit resolves to
 * a metadata path. Retired: standalone **L** and **S** buttons — the whole
 * row body is the affordance now (Search v3 §S4-shell followup).
 */
const rowClickable = computed(
  () =>
    props.libraryOpensEnabled &&
    libraryMetaPath.value != null &&
    !kgMultiEpDedupe.value,
)

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

const quotes = computed(() => {
  const raw = props.hit.supporting_quotes
  if (!Array.isArray(raw) || raw.length === 0) return []
  return raw.filter(
    (q): q is Record<string, unknown> => q != null && typeof q === 'object',
  )
})

const quotesOpen = ref(false)
const liftedOpen = ref(true)

/** Optional enrichment on transcript chunk hits (linked GI insight). */
const lifted = computed((): Record<string, unknown> | null => {
  const raw = props.hit.lifted
  if (raw == null || typeof raw !== 'object') return null
  return raw as Record<string, unknown>
})

function liftedInsightText(ins: unknown): string {
  if (ins == null || typeof ins !== 'object') return ''
  const t = (ins as Record<string, unknown>).text
  return typeof t === 'string' ? t : ''
}

function liftedInsightId(ins: unknown): string {
  if (ins == null || typeof ins !== 'object') return ''
  const id = (ins as Record<string, unknown>).id
  return typeof id === 'string' ? id : ''
}

function liftedEntityLabel(block: unknown, fallback: string): string {
  if (block == null || typeof block !== 'object') return fallback
  const o = block as Record<string, unknown>
  const dn = o.display_name
  if (typeof dn === 'string' && dn.trim()) return dn
  const id = o.id
  if (typeof id === 'string' && id.trim()) return id
  return fallback
}

/** True when **lifted** payload includes a non-empty speaker display block. */
function liftedHasUsableSpeaker(L: Record<string, unknown>): boolean {
  const s = L.speaker
  if (s == null || typeof s !== 'object') return false
  return liftedEntityLabel(s, '').trim().length > 0
}

const liftedQuoteTimeLabel = computed((): string => {
  const L = lifted.value
  if (!L) return '—'
  const q = L.quote
  if (q == null || typeof q !== 'object') return '—'
  const o = q as Record<string, unknown>
  const a = Number(o.timestamp_start_ms)
  const b = Number(o.timestamp_end_ms)
  if (!Number.isFinite(a) && !Number.isFinite(b)) return '—'
  const s = Number.isFinite(a) ? (a / 1000).toFixed(1) : '?'
  const e = Number.isFinite(b) ? (b / 1000).toFixed(1) : '?'
  return `${s}s – ${e}s`
})

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
 */
function onRowClick(): void {
  if (!rowClickable.value) return
  emit('open-library', props.hit)
}

function onRowKeydown(ev: KeyboardEvent): void {
  if (!rowClickable.value) return
  if (ev.key !== 'Enter' && ev.key !== ' ') return
  if (ev.defaultPrevented) return
  ev.preventDefault()
  emit('open-library', props.hit)
}
</script>

<template>
  <article
    class="rounded border border-border bg-elevated p-2 text-xs text-elevated-foreground"
    :class="rowClickable && 'cursor-pointer transition-colors hover:border-primary/50 hover:bg-overlay focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary'"
    :role="rowClickable ? 'button' : undefined"
    :tabindex="rowClickable ? 0 : undefined"
    :aria-label="rowClickable ? 'Open episode in subject panel' : undefined"
    @click="onRowClick"
    @keydown="onRowKeydown"
  >
    <div class="mb-1 flex min-w-0 flex-wrap items-center gap-2">
      <span class="font-mono text-[10px] text-primary">{{ docType }}</span>
      <span
        class="rounded px-1 py-px text-[9px] font-medium uppercase leading-none tracking-wide"
        :class="tierClass"
        data-testid="search-result-tier"
        :title="`Retrieval tier: ${tierLabel}`"
      >{{ tierLabel }}</span>
      <span
        v-if="isCompound"
        class="rounded bg-primary/15 px-1 py-px text-[9px] font-medium leading-none text-primary"
        data-testid="search-result-compound"
        title="Compound result: a transcript segment merged with its linked GI insight."
      >+ insight</span>
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
    <p
      v-if="detailEntity"
      class="mt-1"
      @click.stop
    >
      <button
        type="button"
        class="rounded text-[10px] font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        data-testid="search-result-entity-link"
        :title="`Open ${detailEntity.kind === 'topic' ? 'Topic' : 'Person'} panel`"
        @click="focusDetailEntity"
      >
        Open {{ detailEntity.kind === 'topic' ? 'Topic' : 'Person' }} panel →
      </button>
    </p>

    <div
      v-if="lifted"
      class="mt-2 border-t border-border pt-2"
      role="region"
      aria-label="Lifted GI insight"
      @click.stop
    >
      <button
        type="button"
        class="text-[10px] text-primary underline hover:text-surface-foreground"
        @click="liftedOpen = !liftedOpen"
      >
        {{ liftedOpen ? 'Hide' : 'Show' }} linked GI insight
      </button>
      <div
        v-if="liftedOpen"
        class="mt-1 space-y-1 rounded border border-primary/25 bg-canvas/80 px-2 py-1.5 text-[11px] leading-snug text-surface-foreground"
      >
        <p
          v-if="liftedInsightId(lifted.insight)"
          class="font-mono text-[10px] text-muted"
        >
          {{ liftedInsightId(lifted.insight) }}
        </p>
        <p v-if="liftedInsightText(lifted.insight)">
          {{ truncate(liftedInsightText(lifted.insight), 280) }}
        </p>
        <p
          v-if="lifted.speaker || lifted.topic"
          class="text-[10px] text-muted"
        >
          <span v-if="lifted.speaker">
            Speaker:
            <button
              v-if="liftedEntityId(lifted.speaker)"
              type="button"
              class="rounded text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              data-testid="search-result-lifted-speaker-link"
              :title="`Open Person panel for ${liftedEntityLabel(lifted.speaker, '—')}`"
              @click.stop="focusPersonFromSpeakerId(liftedEntityId(lifted.speaker))"
            >{{ liftedEntityLabel(lifted.speaker, '—') }}</button>
            <template v-else>{{ liftedEntityLabel(lifted.speaker, '—') }}</template>
          </span>
          <span v-if="lifted.speaker && lifted.topic"> · </span>
          <span v-if="lifted.topic">
            Topic:
            <button
              v-if="liftedEntityId(lifted.topic)"
              type="button"
              class="rounded text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              data-testid="search-result-lifted-topic-link"
              :title="`Open Topic panel for ${liftedEntityLabel(lifted.topic, '—')}`"
              @click.stop="focusTopicFromLifted(lifted.topic)"
            >{{ liftedEntityLabel(lifted.topic, '—') }}</button>
            <template v-else>{{ liftedEntityLabel(lifted.topic, '—') }}</template>
          </span>
        </p>
        <p
          v-if="lifted.quote && typeof lifted.quote === 'object'"
          class="text-[10px] text-muted"
        >
          Quote time: {{ liftedQuoteTimeLabel }}
        </p>
        <p
          v-if="
            lifted.quote &&
            typeof lifted.quote === 'object' &&
            !liftedHasUsableSpeaker(lifted) &&
            liftedQuotePayloadHasUsableTiming(lifted.quote)
          "
          class="mt-0.5 text-[10px] leading-snug text-muted/80"
          :data-testid="SEARCH_LIFTED_QUOTE_SPEAKER_UNAVAILABLE_TESTID"
        >
          {{ GI_QUOTE_SPEAKER_UNAVAILABLE_HINT }}
        </p>
      </div>
    </div>

    <div
      v-if="quotes.length"
      class="mt-1.5"
      @click.stop
    >
      <button
        type="button"
        class="text-[10px] text-muted underline hover:text-surface-foreground"
        @click="quotesOpen = !quotesOpen"
      >
        {{ quotesOpen ? 'Hide' : 'Show' }} {{ quotes.length }} supporting
        {{ quotes.length === 1 ? 'quote' : 'quotes' }}
      </button>
      <div
        v-if="quotesOpen"
        class="mt-1 space-y-1"
      >
        <blockquote
          v-for="(q, i) in quotes"
          :key="i"
          class="border-l-2 border-primary/40 pl-2 text-[11px] leading-snug text-muted"
        >
          <p>{{ truncate(String(q.text ?? ''), 300) }}</p>
          <p
            v-if="q.speaker_name || q.speaker_id"
            class="mt-0.5 text-[10px] font-medium text-primary"
          >
            —
            <button
              v-if="typeof q.speaker_id === 'string' && q.speaker_id.trim()"
              type="button"
              class="rounded text-left text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              data-testid="search-result-speaker-link"
              :title="`Open Person panel for ${
                String(q.speaker_name ?? '') ||
                  quoteAttributionDisplayFromId(q.speaker_id)
              }`"
              @click.stop="focusPersonFromSpeakerId(q.speaker_id)"
            >{{
              String(q.speaker_name ?? '') ||
                quoteAttributionDisplayFromId(q.speaker_id)
            }}</button>
            <span v-else>{{
              String(q.speaker_name ?? '') ||
                quoteAttributionDisplayFromId(
                  typeof q.speaker_id === 'string' ? q.speaker_id : '',
                )
            }}</span>
            <span
              v-if="q.timestamp_start_ms != null"
              class="font-normal text-muted"
            >
              ({{ (Number(q.timestamp_start_ms) / 1000).toFixed(1) }}s{{ q.timestamp_end_ms != null ? ` – ${(Number(q.timestamp_end_ms) / 1000).toFixed(1)}s` : '' }})
            </span>
          </p>
          <p
            v-else-if="String(q.text ?? '').trim()"
            class="mt-0.5 text-[10px] leading-snug text-muted/80"
            :data-testid="SUPPORTING_QUOTE_SPEAKER_UNAVAILABLE_TESTID"
          >
            {{ GI_QUOTE_SPEAKER_UNAVAILABLE_HINT }}
          </p>
        </blockquote>
      </div>
    </div>
    <!-- RFC-088 chunk-5: related-topic chips from the QueryEnricher chain
         (only present when the request used ?enrich_results=true and the
         corpus has an enrichments/topic_similarity.json envelope). -->
    <div
      v-if="relatedTopics.length"
      class="mt-1.5 flex flex-wrap items-center gap-1 border-t border-border/60 pt-1.5"
      data-testid="search-result-related-topics"
      aria-label="Related topics from the enrichment layer"
    >
      <span class="text-[9px] uppercase tracking-wider text-muted">Related</span>
      <button
        v-for="r in relatedTopics"
        :key="r.topic_id"
        type="button"
        class="rounded border border-default bg-overlay px-2 py-0.5 text-[10px] hover:bg-overlay-2"
        :data-testid="`search-result-related-topic-${r.topic_id}`"
        :title="r.similarity != null ? `Cosine similarity ${r.similarity.toFixed(2)}` : undefined"
        @click="focusRelatedTopic(r.topic_id)"
      >
        {{ r.topic_label || r.topic_id }}
        <span v-if="r.similarity != null" class="ml-1 text-muted">·{{ r.similarity.toFixed(2) }}</span>
      </button>
    </div>
  </article>
</template>
