<script setup lang="ts">
import { computed, ref } from 'vue'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSearchStore } from '../../stores/search'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { SearchHit } from '../../api/searchApi'
import { graphNodeIdFromSearchHit } from '../../utils/searchFocus'
import { sourceMetadataRelativePathFromSearchHit } from '../../utils/searchHitLibrary'
import ResultCard from './ResultCard.vue'
import SearchFilterBar from './SearchFilterBar.vue'
import SearchResultsVizDialog from './SearchResultsVizDialog.vue'
import HelpTip from '../shared/HelpTip.vue'

const emit = defineEmits<{
  'go-graph': []
  'open-library-episode': [payload: { metadata_relative_path: string }]
  'open-episode-summary': [hit: SearchHit]
}>()

const shell = useShellStore()
const search = useSearchStore()
const nav = useGraphNavigationStore()
const subject = useSubjectStore()

const queryRef = ref<HTMLTextAreaElement | null>(null)
const advancedDialogRef = ref<HTMLDialogElement | null>(null)
const vizDialogRef = ref<InstanceType<typeof SearchResultsVizDialog> | null>(null)

function openAdvancedSearch(): void {
  advancedDialogRef.value?.showModal()
}

function closeAdvancedSearch(): void {
  advancedDialogRef.value?.close()
}

function openSearchResultsViz(): void {
  vizDialogRef.value?.open()
}

function onAdvancedDialogClick(e: MouseEvent): void {
  const el = advancedDialogRef.value
  if (el && e.target === el) {
    el.close()
  }
}

function focusQuery(): void {
  const el = queryRef.value
  if (!el) return
  el.focus()
  try {
    el.select()
  } catch {
    /* ignore */
  }
  el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
}

defineExpose({ focusQuery })

/** L when health + corpus path; per-hit path still required (see ResultCard). */
const libraryOpensEnabled = computed(() =>
  Boolean(shell.healthStatus && shell.hasCorpusPath),
)

const searchFieldsEnabled = computed(
  () => Boolean(shell.healthStatus && shell.hasCorpusPath),
)

const searchFieldDisabledTitle = computed(() => {
  if (!shell.hasCorpusPath) {
    return 'Set corpus path in the status bar to enable search'
  }
  if (!shell.healthStatus) {
    return 'Requires a healthy API connection'
  }
  return ''
})

const enhancedSearchChipClass = computed(() => {
  const base =
    'inline-flex items-center rounded border px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide'
  if (search.enrichmentCallFailed) {
    return `${base} border-warning text-warning`
  }
  return `${base} border-gi text-gi`
})

/** Optional ``tc:…`` compound to widen the graph camera bbox (selection stays on the leaf). */
function topicClusterCompoundIdForCamera(hit: SearchHit): string | null {
  const tc = hit.metadata?.topic_cluster
  if (tc == null || typeof tc !== 'object') return null
  const g = (tc as Record<string, unknown>).graph_compound_parent_id
  return typeof g === 'string' && g.trim() ? g.trim() : null
}

function onFocusHit(hit: SearchHit): void {
  const id = graphNodeIdFromSearchHit(hit)
  if (!id) return
  subject.focusGraphNode(id)
  const tcParent = topicClusterCompoundIdForCamera(hit)
  nav.requestFocusNode(id, undefined, tcParent ? [tcParent] : undefined)
  emit('go-graph')
}

function onOpenLibraryHit(hit: SearchHit): void {
  if (!libraryOpensEnabled.value) return
  const rel = sourceMetadataRelativePathFromSearchHit(hit)
  if (!rel) return
  emit('open-library-episode', { metadata_relative_path: rel })
}

function onOpenEpisodeSummaryHit(hit: SearchHit): void {
  emit('open-episode-summary', hit)
}

async function onSubmit(): Promise<void> {
  await search.runSearch(shell.corpusPath)
}

/** Enter runs search (same as **Search**); Shift+Enter inserts a newline. */
function onQueryKeydown(e: KeyboardEvent): void {
  if (e.key !== 'Enter' || e.shiftKey) return
  if (e.ctrlKey || e.metaKey || e.altKey) return
  if (e.defaultPrevented || e.isComposing) return
  if (!searchFieldsEnabled.value || search.loading) return
  e.preventDefault()
  void onSubmit()
}

/** Advanced feed field: catalog id for API; Library handoff can show catalog title until edited. */
const advancedFeedUi = computed({
  get() {
    if (
      search.feedFilterHandoffPristine &&
      search.feedFilterDisplayLabel?.trim()
    ) {
      return search.feedFilterDisplayLabel.trim()
    }
    return search.filters.feed
  },
  set(v: string) {
    search.commitFeedFilterUiInput(v)
  },
})

const advancedFeedInputTitle = computed(() => {
  if (
    search.feedFilterHandoffPristine &&
    search.feedFilterDisplayLabel?.trim() &&
    search.filters.feed.trim()
  ) {
    return `Search filters by catalog feed id substring (${search.filters.feed.trim()}).`
  }
  return 'Substring match on catalog feed_id in index metadata.'
})

const advancedFeedCombinedTitle = computed(() =>
  searchFieldsEnabled.value ? advancedFeedInputTitle.value : searchFieldDisabledTitle.value,
)
</script>

<template>
  <!-- Grid: chrome row auto-height, results row fills remainder (reliable scroll; flex-1 column was not bounding height). -->
  <section
    class="grid min-h-0 min-w-0 max-w-full flex-1 grid-rows-[auto_minmax(0,1fr)] overflow-x-hidden rounded-lg border border-border bg-surface p-4"
  >
    <div class="min-w-0">
    <div class="mb-2 flex shrink-0 flex-wrap items-center gap-1.5">
      <h2 class="text-sm font-medium text-surface-foreground">
        Semantic search
      </h2>
      <span
        v-if="shell.enrichedSearchAvailable"
        data-testid="search-enhanced-chip"
        :class="enhancedSearchChipClass"
        :title="
          search.enrichmentCallFailed
            ? 'Last search reported enrichment failure — vector hits are still shown.'
            : 'Semantic search enrichment is available on this server.'
        "
      >
        Enhanced
      </span>
      <HelpTip>
        <p class="font-medium text-surface-foreground">
          How semantic search works
        </p>
        <ul class="mt-1.5 list-disc space-y-1 pl-4 text-muted">
          <li>
            Reads the FAISS index under
            <code class="rounded bg-canvas px-0.5 text-[10px]">&lt;corpus&gt;/search/</code>
            — build it with
            <code class="rounded bg-canvas px-0.5 text-[10px]">podcast index</code>.
          </li>
          <li>
            Queries are embedded with the same model as the index (must be available locally).
          </li>
          <li>
            <strong>Since (date)</strong> limits hits to episodes on or after that day (UTC).
            <strong>Top‑k</strong> is the max number of vector hits to return. Other filters are in
            <strong>Advanced search</strong> (doc types, feed filter, speaker, grounded, embedding
            model). The feed field matches <strong>catalog feed id</strong> in the index; after
            <strong>Prefill semantic search</strong> from Library it shows the <strong>feed title</strong>
            from the catalog when known (hover for the id). Non-default choices appear as a read-only
            summary under the link.
          </li>
          <li>
            <strong>Library</strong> → <strong>Prefill semantic search</strong> scopes by feed (see above)
            and fills the query like <strong>Similar episodes</strong> (title + bullets, else episode title
            — not full prose summary), with per-field clipping and a short total cap so huge recap text in
            metadata does not fill the box;
            <strong>Digest</strong> → <strong>Search topic</strong> sets query and Since from the digest
            window. Run <strong>Search</strong> for vector hits.
          </li>
          <li>
            <strong>G</strong> — focus the mapped GI/KG node and switch to <strong>Graph</strong> when
            needed; <strong>L</strong> — open the episode in the <strong>subject panel</strong> (main tab
            unchanged) when the corpus path is set, the API is healthy, and the hit includes
            <code class="rounded bg-canvas px-0.5 text-[10px]">source_metadata_relative_path</code>
            (stamp it with a vector index rebuild). Catalog errors for that episode still surface on the
            <strong>Library</strong> tab if you open it separately.
          </li>
          <li>
            <strong>Search result insights</strong> (after a search) — one scrollable modal: dominant-type
            takeaway line; <strong>Doc types</strong> and <strong>Publish month</strong> side-by-side;
            episodes and feeds show <strong>titles</strong> from search metadata, or the API fills them from
            <code class="rounded bg-canvas px-0.5 text-[10px]">*.metadata.json</code> when the index row has
            no title fields (needs resolvable <code class="rounded bg-canvas px-0.5 text-[10px]">source_metadata_relative_path</code>);
            hover for full text + stable id; top rows + tail counts; similarity bars vs strongest hit; terms
            (heuristic, not KG).
          </li>
          <li>
            <strong>Advanced search</strong> — <strong>Merge duplicate KG surfaces</strong> (on by default)
            collapses <code class="rounded bg-canvas px-0.5 text-[10px]">kg_entity</code> /
            <code class="rounded bg-canvas px-0.5 text-[10px]">kg_topic</code> rows that share the same
            embedded text (like graph Entity/Topic dedupe). Merged rows show <strong>G</strong> only
            (<strong>L</strong> / <strong>E</strong> would imply a single episode); turn merge off for
            per-episode <strong>L</strong> and <strong>E</strong>.
          </li>
        </ul>
      </HelpTip>
    </div>
    <p
      v-if="!shell.hasCorpusPath"
      class="mb-2 shrink-0 text-xs text-muted"
    >
      Set corpus path in the status bar to enable search.
    </p>
    <p
      v-else-if="!shell.healthStatus"
      class="mb-2 shrink-0 text-xs text-muted"
    >
      Requires the API.
    </p>
    <form
      id="semantic-search-form"
      class="shrink-0 space-y-2"
      @submit.prevent="onSubmit"
    >
      <textarea
        id="search-q"
        ref="queryRef"
        v-model="search.query"
        rows="2"
        class="w-full rounded border border-border bg-elevated px-2 py-1.5 text-sm text-elevated-foreground placeholder:text-muted"
        placeholder="Natural language…"
        aria-label="Search query"
        :disabled="!searchFieldsEnabled"
        :title="searchFieldDisabledTitle"
        @keydown="onQueryKeydown"
      />
    </form>
    <div class="mt-2 shrink-0">
      <SearchFilterBar
        :enabled="searchFieldsEnabled"
        :disabled-title="searchFieldDisabledTitle"
        @open-more="openAdvancedSearch"
      />
    </div>
    <div class="mt-2 flex shrink-0 flex-wrap gap-2">
      <button
        type="submit"
        form="semantic-search-form"
        class="rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
        :disabled="!searchFieldsEnabled || search.loading"
        :title="searchFieldDisabledTitle"
      >
        {{ search.loading ? 'Searching…' : 'Search' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-3 py-1.5 text-sm hover:bg-overlay disabled:opacity-40"
        :disabled="!searchFieldsEnabled"
        :title="searchFieldDisabledTitle"
        @click="search.clearResults()"
      >
        Clear
      </button>
    </div>
    <dialog
      ref="advancedDialogRef"
      class="shrink-0 w-[min(100%,24rem)] max-h-[min(90vh,32rem)] overflow-y-auto rounded-lg border border-border bg-surface p-4 text-surface-foreground shadow-xl [&::backdrop]:bg-black/40"
      aria-labelledby="advanced-search-title"
      @click="onAdvancedDialogClick"
    >
      <div class="mb-3 flex items-start justify-between gap-3">
        <h2 id="advanced-search-title" class="text-sm font-medium text-surface-foreground">
          Advanced search
        </h2>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
          @click="closeAdvancedSearch"
        >
          Close
        </button>
      </div>
      <div class="space-y-2">
        <label class="flex cursor-pointer items-center gap-2 text-xs text-muted">
          <input
            v-model="search.filters.groundedOnly"
            type="checkbox"
            class="rounded border-border"
            :disabled="!searchFieldsEnabled"
            :title="searchFieldDisabledTitle"
          >
          Grounded insights only
        </label>
        <label class="block text-xs text-muted" for="search-advanced-feed">
          Feed
          <input
            id="search-advanced-feed"
            v-model="advancedFeedUi"
            type="text"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            placeholder="Catalog feed id substring"
            :title="advancedFeedCombinedTitle"
            :disabled="!searchFieldsEnabled"
            autocomplete="off"
          >
        </label>
        <label class="block text-xs text-muted">
          Speaker contains
          <input
            v-model="search.filters.speaker"
            type="text"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            :disabled="!searchFieldsEnabled"
            :title="searchFieldDisabledTitle"
          >
        </label>
        <label class="block text-xs text-muted">
          Embedding model
          <input
            v-model="search.filters.embeddingModel"
            type="text"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            placeholder="(server default)"
            :disabled="!searchFieldsEnabled"
            :title="searchFieldDisabledTitle"
          >
        </label>
        <label class="flex cursor-pointer items-center gap-2 text-xs text-muted">
          <input
            v-model="search.filters.dedupeKgSurfaces"
            type="checkbox"
            class="rounded border-border"
            :disabled="!searchFieldsEnabled"
            :title="searchFieldDisabledTitle"
          >
          Merge duplicate KG surfaces (kg_entity / kg_topic)
        </label>
      </div>
    </dialog>
    <SearchResultsVizDialog
      ref="vizDialogRef"
      :hits="search.results"
    />
    </div>
    <div
      data-testid="semantic-search-results-scroll"
      class="min-h-0 min-w-0 overflow-x-hidden overflow-y-auto overscroll-y-contain [scrollbar-gutter:stable] [scrollbar-width:thin] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border"
    >
      <p
        v-if="search.error"
        class="mt-2 text-xs text-danger"
      >
        {{ search.error }}
      </p>
      <p
        v-if="search.apiError"
        class="mt-2 text-xs text-warning"
      >
        {{ search.apiError }}
      </p>
      <div
        v-if="search.results.length"
        class="mt-3 space-y-2"
      >
        <div class="flex flex-wrap items-center gap-x-3 gap-y-1">
          <p class="text-xs font-medium text-muted">
            {{ search.results.length }}
            {{ search.results.length === 1 ? 'result' : 'results' }}
          </p>
          <p
            v-if="search.liftStats && search.liftStats.transcript_hits_returned > 0"
            class="text-[10px] text-muted"
            title="Transcript lift coverage for this result page (rows returned vs rows with a linked GI insight)."
          >
            Lift:
            {{ search.liftStats.lift_applied }} /
            {{ search.liftStats.transcript_hits_returned }}
            transcript rows linked to GI
          </p>
          <button
            type="button"
            class="text-xs text-primary underline decoration-primary/60 underline-offset-2 hover:decoration-primary"
            @click="openSearchResultsViz"
          >
            Search result insights
          </button>
        </div>
        <ResultCard
          v-for="(h, i) in search.results"
          :key="`${h.doc_id}-${i}`"
          :hit="h"
          :library-opens-enabled="libraryOpensEnabled"
          @focus="onFocusHit"
          @open-library="onOpenLibraryHit"
          @open-episode-summary="onOpenEpisodeSummaryHit"
        />
      </div>
    </div>
  </section>
</template>
