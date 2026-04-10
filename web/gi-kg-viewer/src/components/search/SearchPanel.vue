<script setup lang="ts">
import { ref } from 'vue'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSearchStore } from '../../stores/search'
import { useShellStore } from '../../stores/shell'
import type { SearchHit } from '../../api/searchApi'
import { graphNodeIdFromSearchHit } from '../../utils/searchFocus'
import ResultCard from './ResultCard.vue'
import HelpTip from '../shared/HelpTip.vue'

const emit = defineEmits<{ 'go-graph': [] }>()

const shell = useShellStore()
const search = useSearchStore()
const nav = useGraphNavigationStore()

const queryRef = ref<HTMLTextAreaElement | null>(null)

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

function toggleType(v: string): void {
  const i = search.filters.types.indexOf(v)
  if (i >= 0) {
    search.filters.types.splice(i, 1)
  } else {
    search.filters.types.push(v)
  }
}

function onFocusHit(hit: SearchHit): void {
  const id = graphNodeIdFromSearchHit(hit)
  if (!id) return
  nav.requestFocusNode(id)
  emit('go-graph')
}

async function onSubmit(): Promise<void> {
  await search.runSearch(shell.corpusPath)
}
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
            Use filters to narrow by doc type, feed id substring (matches catalog
            <code class="rounded bg-canvas px-0.5 text-[10px]">feed_id</code>), date, speaker, or
            grounded status.
          </li>
          <li>
            <strong>Library</strong> / <strong>Digest</strong> → <strong>Prefill semantic search</strong>
            sets the feed filter and fills the query from summary text (or title / bullets); run
            <strong>Search</strong> for vector hits.
          </li>
          <li>"Show on graph" works when the hit maps to a node in your loaded artifacts.</li>
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
      class="space-y-2"
      @submit.prevent="onSubmit"
    >
      <p
        v-if="search.libraryHandoffHint"
        class="rounded border border-border bg-elevated px-2 py-1.5 text-xs text-muted"
        role="status"
      >
        {{ search.libraryHandoffHint }}
      </p>
      <label class="block text-xs text-muted" for="search-q">Query</label>
      <textarea
        id="search-q"
        ref="queryRef"
        v-model="search.query"
        rows="2"
        class="w-full rounded border border-border bg-elevated px-2 py-1.5 text-sm text-elevated-foreground placeholder:text-muted"
        placeholder="Natural language…"
        :disabled="!shell.healthStatus"
      />
      <div class="grid grid-cols-2 gap-2">
        <label class="text-xs text-muted">
          Top‑k
          <input
            v-model.number="search.filters.topK"
            type="number"
            min="1"
            max="100"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            :disabled="!shell.healthStatus"
          >
        </label>
        <label class="flex cursor-pointer items-center gap-2 text-xs text-muted">
          <input
            v-model="search.filters.groundedOnly"
            type="checkbox"
            class="rounded border-border"
            :disabled="!shell.healthStatus"
          >
          Grounded insights only
        </label>
      </div>
      <div class="flex flex-wrap gap-2">
        <label class="text-xs text-muted">
          Feed id (substring)
          <input
            v-model="search.filters.feed"
            type="text"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            :disabled="!shell.healthStatus"
            autocomplete="off"
          >
        </label>
        <label class="text-xs text-muted">
          Since (date)
          <input
            v-model="search.filters.since"
            type="date"
            class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
            :disabled="!shell.healthStatus"
          >
        </label>
      </div>
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
      <div class="flex flex-wrap gap-2">
        <button
          type="submit"
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
    </form>
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
      <p class="text-xs font-medium text-muted">
        {{ search.results.length }} result(s) for “{{ search.lastSubmittedQuery }}”
      </p>
      <ResultCard
        v-for="(h, i) in search.results"
        :key="`${h.doc_id}-${i}`"
        :hit="h"
        @focus="onFocusHit"
      />
    </div>
  </section>
</template>
