<script setup lang="ts">
import { computed, ref } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import { truncate } from '../../utils/formatting'
import {
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
  SEARCH_RESULT_GRAPH_BUTTON_CLASS,
} from '../../utils/searchResultActionStyles'
import { graphNodeIdFromSearchHit } from '../../utils/searchFocus'
import { quoteAttributionDisplayFromId } from '../../utils/parsing'
import { isKgSurfaceMultiEpisodeDedupe } from '../../utils/searchHitKgDedupe'
import { sourceMetadataRelativePathFromSearchHit } from '../../utils/searchHitLibrary'

const props = defineProps<{
  hit: SearchHit
  /** Corpus Library API + corpus path — required to offer Library navigation. */
  libraryOpensEnabled: boolean
}>()

const emit = defineEmits<{
  focus: [SearchHit]
  'open-library': [SearchHit]
  'open-episode-summary': [SearchHit]
}>()

const docType = computed(() => String(props.hit.metadata?.doc_type ?? '?'))

const episodeId = computed(() => {
  const e = props.hit.metadata?.episode_id
  return typeof e === 'string' ? e : null
})

const focusable = computed(() => graphNodeIdFromSearchHit(props.hit) != null)

const kgMultiEpDedupe = computed(() => isKgSurfaceMultiEpisodeDedupe(props.hit))

const libraryMetaPath = computed(() => sourceMetadataRelativePathFromSearchHit(props.hit))

const openLibrary = computed(
  () =>
    props.libraryOpensEnabled &&
    libraryMetaPath.value != null &&
    !kgMultiEpDedupe.value,
)

const openEpisodeSummary = computed(
  () =>
    props.libraryOpensEnabled &&
    libraryMetaPath.value != null &&
    !kgMultiEpDedupe.value,
)

const showEpisodeChip = computed(
  () => Boolean(episodeId.value) && !kgMultiEpDedupe.value,
)

const hasActions = computed(
  () => focusable.value || openLibrary.value || openEpisodeSummary.value,
)

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
    'Same episode across chunks in this search; not sent when opening Library (that uses the metadata file path).'
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

function onGraphClick(ev: MouseEvent): void {
  ev.stopPropagation()
  if (!focusable.value) return
  emit('focus', props.hit)
}

function onLibraryClick(ev: MouseEvent): void {
  ev.stopPropagation()
  if (!openLibrary.value) return
  emit('open-library', props.hit)
}

function onEpisodeSummaryClick(ev: MouseEvent): void {
  ev.stopPropagation()
  if (!openEpisodeSummary.value) return
  emit('open-episode-summary', props.hit)
}

function onEpisodeIdChipClick(ev: MouseEvent): void {
  ev.stopPropagation()
}
</script>

<template>
  <article
    class="rounded border border-border bg-elevated p-2 text-xs text-elevated-foreground"
  >
    <div class="mb-1 flex min-w-0 flex-wrap items-center gap-2">
      <span class="font-mono text-[10px] text-primary">{{ docType }}</span>
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
          v-if="openLibrary"
          type="button"
          class="flex size-6 shrink-0 items-center justify-center rounded-sm bg-primary text-[11px] font-semibold leading-none text-primary-foreground hover:opacity-90"
          aria-label="Open episode in Library"
          title="Open in Library"
          @click="onLibraryClick"
        >
          L
        </button>
        <button
          v-if="openEpisodeSummary"
          type="button"
          class="flex size-6 shrink-0 items-center justify-center rounded-sm border border-border bg-canvas text-[10px] font-semibold leading-none text-surface-foreground hover:bg-overlay"
          aria-label="Episode summary in right panel"
          title="Episode summary (right panel)"
          @click="onEpisodeSummaryClick"
        >
          S
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
    <p class="leading-snug text-surface-foreground">
      {{ truncate(hit.text || '(no text)', 320) }}
    </p>

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
            {{
              String(q.speaker_name ?? '') ||
                quoteAttributionDisplayFromId(
                  typeof q.speaker_id === 'string' ? q.speaker_id : '',
                )
            }}
            <span
              v-if="q.timestamp_start_ms != null"
              class="font-normal text-muted"
            >
              ({{ (Number(q.timestamp_start_ms) / 1000).toFixed(1) }}s{{ q.timestamp_end_ms != null ? ` – ${(Number(q.timestamp_end_ms) / 1000).toFixed(1)}s` : '' }})
            </span>
          </p>
        </blockquote>
      </div>
    </div>
  </article>
</template>
