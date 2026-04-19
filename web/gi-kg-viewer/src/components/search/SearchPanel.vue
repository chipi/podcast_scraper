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

const typeOptions = [
  { value: 'insight', label: 'Insights' },
  { value: 'quote', label: 'Quotes' },
  { value: 'kg_entity', label: 'KG entities' },
  { value: 'kg_topic', label: 'KG topics' },
  { value: 'summary', label: 'Summary bullets' },
  { value: 'transcript', label: 'Transcript chunks' },
] as const

const advancedFilterSummaryLines = computed(() => {
  const f = search.filters
  const lines: string[] = []
  const topK = Number(f.topK)
  if (Number.isFinite(topK) && topK !== 10) {
    lines.push(`Top‑k: ${topK}`)
  }
  if (f.groundedOnly) {
    lines.push('Grounded insights only')
  }
  const feed = f.feed.trim()
  if (feed) {
    const feedTitle = search.feedFilterDisplayLabel?.trim()
    const showTitle = Boolean(
      search.feedFilterHandoffPristine && feedTitle,
    )
    lines.push(showTitle ? `Feed: ${feedTitle}` : `Feed: ${feed}`)
  }
  const speaker = f.speaker.trim()
  if (speaker) {
    lines.push(`Speaker: ${speaker}`)
  }
  const embedding = f.embeddingModel.trim()
  if (embedding) {
    lines.push(`Embedding model: ${embedding}`)
  }
  if (f.types.length > 0) {
    const labels = f.types
      .map((v) => typeOptions.find((o) => o.value === v)?.label ?? v)
      .join(', ')
    lines.push(`Doc types: ${labels}`)
  }
  if (!f.dedupeKgSurfaces) {
    lines.push('Merge duplicate KG surfaces: off')
  }
  return lines
})

const hasAdvancedFilterSummary = computed(
  () => advancedFilterSummaryLines.value.length > 0,
)

/** L when health + corpus path; per-hit path still required (see ResultCard). */
const libraryOpensEnabled = computed(() =>
  Boolean(shell.healthStatus && shell.hasCorpusPath),
)

function toggleType(v: string): void {
  const i = search.filters.types.indexOf(v)
  if (i >= 0) {
    search.filters.types.splice(i, 1)
  } else {
    search.filters.types.push(v)
  }
}

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
</script>

<template>
  <section class="rounded-lg border border-border bg-surface p-4">
    <div class="mb-2 flex items-center gap-1.5">
      <h2 class="text-sm font-medium text-surface-foreground">
        Semantic search
      </h2>
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
      v-if="!shell.healthStatus"
      class="mb-2 text-xs text-muted"
    >
      Requires the API.
    </p>
    <form
      id="semantic-search-form"
      class="space-y-2"
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
        :disabled="!shell.healthStatus"
      />
      <div class="flex max-w-full flex-nowrap items-end gap-3">
        <div class="w-[min(100%,10.5rem)] shrink-0">
          <label class="block text-xs text-muted" for="search-since-date">Since (date)</label>
          <input
            id="search-since-date"
            v-model="search.filters.since"
            type="date"
            class="w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            :disabled="!shell.healthStatus"
          >
        </div>
        <div class="w-[4.75rem] shrink-0">
          <label class="block text-xs text-muted" for="search-top-k">Top‑k</label>
          <input
            id="search-top-k"
            v-model.number="search.filters.topK"
            type="number"
            min="1"
            max="100"
            class="w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            :disabled="!shell.healthStatus"
          >
        </div>
      </div>
    </form>
    <div class="mt-2">
      <button
        type="button"
        class="text-xs text-primary underline decoration-primary/60 underline-offset-2 hover:decoration-primary disabled:opacity-40"
        :disabled="!shell.healthStatus"
        @click="openAdvancedSearch"
      >
        Advanced search
      </button>
    </div>
    <div
      v-if="hasAdvancedFilterSummary"
      role="region"
      aria-label="Active advanced filters"
      class="mt-2 rounded border border-border bg-elevated/60 px-2 py-1.5"
    >
      <p class="text-[10px] font-medium uppercase tracking-wide text-muted">
        Advanced filters
      </p>
      <ul class="mt-1 space-y-0.5 text-[10px] leading-snug text-muted">
        <li
          v-for="(line, i) in advancedFilterSummaryLines"
          :key="i"
        >
          {{ line }}
        </li>
      </ul>
    </div>
    <div class="mt-2 flex flex-wrap gap-2">
      <button
        type="submit"
        form="semantic-search-form"
        class="rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
        :disabled="!shell.healthStatus || search.loading"
      >
        {{ search.loading ? 'Searching…' : 'Search' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-3 py-1.5 text-sm hover:bg-overlay disabled:opacity-40"
        :disabled="!shell.healthStatus"
        @click="search.clearResults()"
      >
        Clear
      </button>
    </div>
    <dialog
      ref="advancedDialogRef"
      class="w-[min(100%,24rem)] max-h-[min(90vh,32rem)] overflow-y-auto rounded-lg border border-border bg-surface p-4 text-surface-foreground shadow-xl [&::backdrop]:bg-black/40"
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
            :disabled="!shell.healthStatus"
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
            :title="advancedFeedInputTitle"
            :disabled="!shell.healthStatus"
            autocomplete="off"
          >
        </label>
        <label class="block text-xs text-muted">
          Speaker contains
          <input
            v-model="search.filters.speaker"
            type="text"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            :disabled="!shell.healthStatus"
          >
        </label>
        <label class="block text-xs text-muted">
          Embedding model
          <input
            v-model="search.filters.embeddingModel"
            type="text"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            placeholder="(server default)"
            :disabled="!shell.healthStatus"
          >
        </label>
        <label class="flex cursor-pointer items-center gap-2 text-xs text-muted">
          <input
            v-model="search.filters.dedupeKgSurfaces"
            type="checkbox"
            class="rounded border-border"
            :disabled="!shell.healthStatus"
          >
          Merge duplicate KG surfaces (kg_entity / kg_topic)
        </label>
        <fieldset class="space-y-1">
          <legend class="text-xs font-medium text-muted">
            Doc types (empty = all)
          </legend>
          <div class="flex flex-wrap gap-2">
            <label
              v-for="opt in typeOptions"
              :key="opt.value"
              class="flex cursor-pointer items-center gap-1 text-xs"
            >
              <input
                type="checkbox"
                class="rounded border-border"
                :checked="search.filters.types.includes(opt.value)"
                :disabled="!shell.healthStatus"
                @change="toggleType(opt.value)"
              >
              {{ opt.label }}
            </label>
          </div>
        </fieldset>
      </div>
    </dialog>
    <SearchResultsVizDialog
      ref="vizDialogRef"
      :hits="search.results"
    />
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
  </section>
</template>
