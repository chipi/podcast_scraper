<script setup lang="ts">
/**
 * Search-results merged card for a set of transcript hits that all point
 * at the same episode. Emitted by the ``collapseTranscriptHitsByEpisode``
 * helper. Replaces the flat "N cards, one per chunk" view with:
 *
 *   - Episode chrome at top (title / feed / publish date).
 *   - Per-chunk excerpts as sub-rows so the user still sees WHERE we
 *     found the query in the transcript.
 *   - Row click → open Episode subject panel (same default action as
 *     the regular ResultCard for episode rows).
 *
 * The user's workflow: click the row to open the episode in the right
 * rail, then click "View transcript" on the rail to open the raw
 * transcript file and locate the chunk they saw here.
 */
import { computed } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import type { TranscriptClusterHit } from '../../utils/collapseTranscriptHitsByEpisode'
import { truncate } from '../../utils/formatting'

const props = defineProps<{
  cluster: TranscriptClusterHit
  /** Passthrough — same rule as ResultCard: only opens Library when the
   *  server said so + the hit has a metadata path (checked here via
   *  ``cluster.metadataRelativePath``). */
  libraryOpensEnabled: boolean
}>()

const emit = defineEmits<{
  /** Row-body click opens the Episode subject panel. Payload is the
   *  cluster's top-scoring member so the caller can reuse the existing
   *  ``open-library`` handler that takes a ``SearchHit``. */
  'open-library': [SearchHit]
  /** Member sub-row click — request the parent open the shared
   *  ``TranscriptViewerDialog`` seeked to that chunk's start timestamp.
   *  Payload carries the metadata path (so the parent can resolve the
   *  transcript relpath) and the chunk's start ms when available. */
  'open-transcript-at': [payload: {
    metadataRelativePath: string
    episodeTitle: string
    audioSeekStartMs: number | null
    hit: SearchHit
  }]
}>()

const rowClickable = computed(
  () => props.libraryOpensEnabled && props.cluster.metadataRelativePath != null,
)

const memberCountLabel = computed<string>(() => {
  const n = props.cluster.members.length
  return n === 1 ? '1 transcript match' : `${n} transcript matches`
})

function timestampLabelForMember(hit: SearchHit): string | null {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const a = Number(md.timestamp_start_ms)
  if (!Number.isFinite(a)) return null
  const s = Math.floor(a / 1000)
  const mm = Math.floor(s / 60)
  const ss = s % 60
  return `${mm}:${ss.toString().padStart(2, '0')}`
}

function memberChunkStartMs(hit: SearchHit): number | null {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const raw = Number(md.timestamp_start_ms)
  return Number.isFinite(raw) ? raw : null
}

function onMemberClick(hit: SearchHit): void {
  const metaRel = props.cluster.metadataRelativePath
  if (!metaRel) return
  emit('open-transcript-at', {
    metadataRelativePath: metaRel,
    episodeTitle: props.cluster.episodeTitle,
    audioSeekStartMs: memberChunkStartMs(hit),
    hit,
  })
}

function onRowClick(): void {
  if (!rowClickable.value) return
  emit('open-library', props.cluster.members[0])
}

function onRowKeydown(ev: KeyboardEvent): void {
  if (!rowClickable.value) return
  if (ev.key !== 'Enter' && ev.key !== ' ') return
  if (ev.defaultPrevented) return
  ev.preventDefault()
  emit('open-library', props.cluster.members[0])
}
</script>

<template>
  <article
    class="rounded border border-border bg-elevated p-2 text-xs text-elevated-foreground"
    :class="rowClickable && 'cursor-pointer transition-colors hover:border-primary/50 hover:bg-overlay focus-visible:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary'"
    :role="rowClickable ? 'button' : undefined"
    :tabindex="rowClickable ? 0 : undefined"
    :aria-label="rowClickable ? 'Open episode in subject panel' : undefined"
    data-testid="search-transcript-cluster"
    @click="onRowClick"
    @keydown="onRowKeydown"
  >
    <div class="mb-1 flex min-w-0 flex-wrap items-center gap-2">
      <span class="font-mono text-[10px] text-primary">episode</span>
      <span
        class="rounded bg-success/15 px-1 py-px text-[9px] font-medium uppercase leading-none tracking-wide text-success"
        title="Transcript matches rolled up under one episode row"
      >Transcript</span>
      <span
        class="rounded bg-primary/15 px-1 py-px text-[9px] font-medium leading-none text-primary"
        data-testid="search-transcript-cluster-count"
      >{{ memberCountLabel }}</span>
      <span
        class="rounded bg-overlay px-1 py-px font-mono text-[9px] leading-none text-muted"
        :title="`Top transcript-match score for this episode (best chunk). ${cluster.members.length} total chunks matched.`"
      >{{ cluster.topScore.toFixed(4) }}</span>
    </div>
    <p
      class="leading-snug font-medium text-surface-foreground"
      data-testid="search-transcript-cluster-episode-title"
    >
      {{ cluster.episodeTitle }}
    </p>
    <p
      v-if="cluster.feedTitle || cluster.publishDate"
      class="mt-0.5 text-[10px] text-muted"
    >
      <span v-if="cluster.feedTitle">{{ cluster.feedTitle }}</span>
      <span v-if="cluster.feedTitle && cluster.publishDate"> · </span>
      <span v-if="cluster.publishDate">{{ cluster.publishDate }}</span>
    </p>
    <ul
      class="mt-2 flex flex-col gap-1"
      aria-label="Chunks where the query matched in the transcript"
      data-testid="search-transcript-cluster-members"
    >
      <li
        v-for="m in cluster.members"
        :key="m.doc_id"
        class="rounded border border-border/60 bg-canvas/60 px-2 py-1 text-[11px] leading-snug text-surface-foreground"
        :class="cluster.metadataRelativePath ? 'cursor-pointer hover:border-primary/50 hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary' : ''"
        :role="cluster.metadataRelativePath ? 'button' : undefined"
        :tabindex="cluster.metadataRelativePath ? 0 : undefined"
        :aria-label="cluster.metadataRelativePath ? 'Open transcript at this position' : undefined"
        :title="cluster.metadataRelativePath ? 'Open transcript at this position (in-app viewer, audio seeks to timestamp when available).' : undefined"
        data-testid="search-transcript-cluster-member"
        @click.stop="onMemberClick(m)"
        @keydown.enter.stop.prevent="onMemberClick(m)"
        @keydown.space.stop.prevent="onMemberClick(m)"
      >
        <p
          v-if="timestampLabelForMember(m)"
          class="mb-0.5 flex items-center gap-1 text-[9px] uppercase tracking-wide text-muted"
        >
          <span>@ {{ timestampLabelForMember(m) }}</span>
          <span class="font-mono text-muted/80">· {{ m.score.toFixed(4) }}</span>
        </p>
        <p>{{ truncate(m.text || '(no text)', 240) }}</p>
      </li>
    </ul>
  </article>
</template>
