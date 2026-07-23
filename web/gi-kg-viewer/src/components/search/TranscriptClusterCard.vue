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
import SearchResultRowIcon from './SearchResultRowIcon.vue'

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
  return n === 1 ? '1 match on this episode' : `${n} matches on this episode`
})

/**
 * Distinct list of "what matched on this episode" — displayed in the
 * cluster header so the user sees at a glance whether we found the
 * query in the title, description, summary, or the transcript (and how
 * many of each). Ordered for stable rendering:
 *   Title · Description · Summary · Summary bullet · Transcript
 *
 * Each entry counts how many of the cluster's members contributed. The
 * order is fixed (not by member frequency) so the row header stays
 * predictable across similar queries.
 */
interface MatchedFieldBreakdown {
  label: string
  count: number
}
const MATCHED_FIELD_ORDER = [
  'Title',
  'Description',
  'Summary',
  'Summary bullet',
  'Transcript',
] as const

const matchedFieldsSummary = computed<MatchedFieldBreakdown[]>(() => {
  const counts = new Map<string, number>()
  for (const m of props.cluster.members) {
    const label = matchedFieldLabelForMember(m)
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  const out: MatchedFieldBreakdown[] = []
  for (const label of MATCHED_FIELD_ORDER) {
    const c = counts.get(label)
    if (c) out.push({ label, count: c })
  }
  // Any labels outside the fixed order (future doc_types) get appended
  // in insertion order so nothing silently disappears.
  for (const [label, count] of counts) {
    if (!MATCHED_FIELD_ORDER.includes(label as (typeof MATCHED_FIELD_ORDER)[number])) {
      out.push({ label, count })
    }
  }
  return out
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

/**
 * "matched: X" chip label for a cluster member. Uses ``matched_field``
 * when present (kept in the indexer row metadata for future consumers),
 * otherwise derives from the row's ``doc_type`` — the aux table's
 * schema does not round-trip metadata dicts so real hits arrive without
 * ``matched_field`` and the doc_type itself carries the same signal.
 */
function matchedFieldLabelForMember(hit: SearchHit): string {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const raw = typeof md.matched_field === 'string' ? md.matched_field : ''
  switch (raw) {
    case 'title':
      return 'Title'
    case 'description':
      return 'Description'
    case 'summary':
      return 'Summary'
    case 'summary_bullet':
      return 'Summary bullet'
    case 'transcript':
      return 'Transcript'
    default:
      break
  }
  const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
  switch (docType) {
    case 'transcript':
      return 'Transcript'
    case 'episode_title':
      return 'Title'
    case 'episode_description':
      return 'Description'
    case 'summary_short':
      return 'Summary'
    case 'summary':
      return 'Summary bullet'
    default:
      return 'Match'
  }
}

function memberIsTranscript(hit: SearchHit): boolean {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  return md.doc_type === 'transcript'
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
      <SearchResultRowIcon
        doc-type="episode"
        source-tier="segment"
        :subtitle="cluster.episodeTitle"
        data-testid="search-result-tier"
        data-tier="Transcript"
      />
      <span
        class="rounded bg-primary/15 px-1 py-px text-[9px] font-medium leading-none text-primary"
        data-testid="search-transcript-cluster-count"
      >{{ memberCountLabel }}</span>
      <span
        class="rounded bg-overlay px-1 py-px font-mono text-[9px] leading-none text-muted"
        :title="`Top match score on this episode (best member). ${cluster.members.length} total matches.`"
      >{{ cluster.topScore.toFixed(4) }}</span>
    </div>
    <p
      class="leading-snug font-medium text-surface-foreground"
      data-testid="search-transcript-cluster-episode-title"
    >
      {{ cluster.episodeTitle }}
    </p>
    <p
      class="mt-1 flex flex-wrap items-center gap-1 text-[10px] uppercase tracking-wide text-muted"
      data-testid="search-transcript-cluster-matched-fields"
      :aria-label="`Matched on: ${matchedFieldsSummary.map((m) => m.count > 1 ? `${m.label} ×${m.count}` : m.label).join(', ')}`"
    >
      <span class="text-muted/80">Matched:</span>
      <span
        v-for="m in matchedFieldsSummary"
        :key="m.label"
        class="rounded bg-primary/15 px-1 py-px font-medium leading-none text-primary"
        :data-match-field="m.label"
      >{{ m.count > 1 ? `${m.label} ×${m.count}` : m.label }}</span>
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
        :class="memberIsTranscript(m) && cluster.metadataRelativePath ? 'cursor-pointer hover:border-primary/50 hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary' : ''"
        :role="memberIsTranscript(m) && cluster.metadataRelativePath ? 'button' : undefined"
        :tabindex="memberIsTranscript(m) && cluster.metadataRelativePath ? 0 : undefined"
        :aria-label="memberIsTranscript(m) && cluster.metadataRelativePath ? 'Open transcript at this position' : undefined"
        :title="memberIsTranscript(m) && cluster.metadataRelativePath ? 'Open transcript at this position (in-app viewer, audio seeks to timestamp when available).' : undefined"
        data-testid="search-transcript-cluster-member"
        :data-match-field="(m.metadata?.matched_field as string | undefined) ?? ''"
        @click.stop="memberIsTranscript(m) && onMemberClick(m)"
        @keydown.enter.stop.prevent="memberIsTranscript(m) && onMemberClick(m)"
        @keydown.space.stop.prevent="memberIsTranscript(m) && onMemberClick(m)"
      >
        <p class="mb-0.5 flex items-center gap-1 text-[9px] uppercase tracking-wide text-muted">
          <span
            class="rounded bg-primary/15 px-1 py-px font-medium text-primary"
            data-testid="search-transcript-cluster-member-field"
          >
            {{ matchedFieldLabelForMember(m) }}
          </span>
          <span
            v-if="timestampLabelForMember(m)"
          >@ {{ timestampLabelForMember(m) }}</span>
          <span class="font-mono text-muted/80">· {{ m.score.toFixed(4) }}</span>
        </p>
        <p>{{ truncate(m.text || '(no text)', 240) }}</p>
      </li>
    </ul>
  </article>
</template>
