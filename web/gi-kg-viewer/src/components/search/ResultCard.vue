<script setup lang="ts">
import { computed, ref } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import { truncate } from '../../utils/formatting'
import { graphNodeIdFromSearchHit } from '../../utils/searchFocus'

const props = defineProps<{
  hit: SearchHit
}>()

const emit = defineEmits<{ focus: [SearchHit] }>()

const docType = computed(() => String(props.hit.metadata?.doc_type ?? '?'))

const episodeId = computed(() => {
  const e = props.hit.metadata?.episode_id
  return typeof e === 'string' ? e : null
})

const focusable = computed(() => graphNodeIdFromSearchHit(props.hit) != null)

const quotes = computed(() => {
  const raw = props.hit.supporting_quotes
  if (!Array.isArray(raw) || raw.length === 0) return []
  return raw.filter(
    (q): q is Record<string, unknown> => q != null && typeof q === 'object',
  )
})

const quotesOpen = ref(false)

function onFocus(): void {
  if (focusable.value) emit('focus', props.hit)
}
</script>

<template>
  <article
    class="rounded border border-border bg-elevated p-2 text-xs text-elevated-foreground"
  >
    <div class="mb-1 flex flex-wrap items-center gap-2">
      <span class="font-mono text-[10px] text-primary">{{ docType }}</span>
      <span class="text-muted">score {{ hit.score.toFixed(4) }}</span>
      <span
        v-if="episodeId"
        class="text-muted"
      >ep {{ episodeId }}</span>
      <button
        v-if="focusable"
        type="button"
        class="ml-auto rounded border border-border px-2 py-0.5 text-[10px] font-medium hover:bg-overlay"
        @click="onFocus"
      >
        Show on graph
      </button>
    </div>
    <p class="leading-snug text-surface-foreground">
      {{ truncate(hit.text || '(no text)', 320) }}
    </p>

    <div
      v-if="quotes.length"
      class="mt-1.5"
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
            v-if="q.speaker_id"
            class="mt-0.5 text-[10px] font-medium text-primary"
          >
            — {{ q.speaker_id }}
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
