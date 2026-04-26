<script setup lang="ts">
/**
 * #672 — Topic / Entity rail panel ("TEV"). Reads the focused subject id
 * from the subject store, looks it up in the merged GI/KG slice, and shows
 * name + aliases + a monthly mentions timeline + the linked insights/quotes.
 *
 * Mounts inside ``SubjectRail`` for ``subject.kind === 'topic'``. Entity
 * subjects still flow through ``focusGraphNode`` → existing ``NodeDetail``.
 */
import { computed } from 'vue'
import type { RawGraphNode } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useSubjectStore } from '../../stores/subject'
import {
  findRawNodeInArtifact,
  findRawNodeInArtifactByIdOrPrefixed,
} from '../../utils/parsing'
import { logicalEpisodeIdFromGraphNodeId } from '../../utils/graphEpisodeMetadata'
import { buildSubjectMentionsTimeline } from '../../utils/subjectMentionsTimeline'
import SubjectTimelineChart from './SubjectTimelineChart.vue'

/** Mentions list cap — the rail panel is narrow and the timeline above
 *  already covers volume; 25 is enough for "what is this subject about?"
 *  scanning before the user opens the full graph. */
const TOPIC_ENTITY_VIEW_MENTIONS_CAP = 25

const emit = defineEmits<{
  goGraph: []
  closeSubject: []
  openLibraryEpisode: [{ metadata_relative_path: string }]
  prefillSemanticSearch: [{ query: string }]
}>()

const artifacts = useArtifactsStore()
const subject = useSubjectStore()

const subjectId = computed(() => subject.topicId?.trim() || '')

const subjectNode = computed<RawGraphNode | null>(() => {
  const art = artifacts.displayArtifact
  const id = subjectId.value
  if (!art || !id) return null
  return findRawNodeInArtifactByIdOrPrefixed(art, id)
})

/** Actual graph node id (prefixed form after KG merge) for edge lookups. */
const subjectGraphNodeId = computed<string | null>(() => {
  const n = subjectNode.value
  if (!n || n.id == null) return subjectId.value || null
  return String(n.id)
})

const subjectKindLabel = computed(() => {
  const t = subjectNode.value?.type
  if (t === 'Topic') return 'Topic'
  if (t === 'Entity' || t === 'Person') return 'Entity'
  return 'Subject'
})

const subjectName = computed(() => {
  const n = subjectNode.value
  if (!n) return subjectId.value
  const p = n.properties as Record<string, unknown> | undefined
  const label = typeof p?.label === 'string' ? p.label.trim() : ''
  if (label) return label
  const name = typeof p?.name === 'string' ? p.name.trim() : ''
  if (name) return name
  return subjectId.value
})

const subjectAliases = computed(() => {
  const a = subjectNode.value?.properties?.aliases
  if (!Array.isArray(a)) return ''
  const parts = a
    .filter((x): x is string => typeof x === 'string' && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(', ') : ''
})

const subjectDescription = computed(() => {
  const d = subjectNode.value?.properties?.description
  return typeof d === 'string' && d.trim() ? d.trim() : ''
})

const timeline = computed(() =>
  buildSubjectMentionsTimeline(artifacts.displayArtifact, subjectGraphNodeId.value),
)

interface MentionRow {
  id: string
  type: 'Insight' | 'Quote'
  text: string
  episodeId: string | null
  episodeTitle: string | null
  publishDate: string | null
}

const mentionRows = computed<MentionRow[]>(() => {
  const art = artifacts.displayArtifact
  if (!art) return []
  const ids = [
    ...timeline.value.insightIds.map((id) => ({ id, type: 'Insight' as const })),
    ...timeline.value.quoteIds.map((id) => ({ id, type: 'Quote' as const })),
  ]
  const episodes = new Map<string, RawGraphNode>()
  for (const n of art.data?.nodes ?? []) {
    if (!n || String(n.type) !== 'Episode') continue
    const lid = logicalEpisodeIdFromGraphNodeId(String(n.id ?? ''))
    if (lid) episodes.set(lid, n)
  }
  const rows: MentionRow[] = []
  for (const { id, type } of ids) {
    const n = findRawNodeInArtifact(art, id)
    if (!n) continue
    const p = n.properties as Record<string, unknown> | undefined
    const text =
      type === 'Insight'
        ? (typeof p?.text === 'string' && p.text.trim()) ||
          (typeof p?.label === 'string' && p.label.trim()) ||
          ''
        : (typeof p?.text === 'string' && p.text.trim()) || ''
    const episodeId = typeof p?.episode_id === 'string' ? p.episode_id.trim() : null
    const ep = episodeId ? episodes.get(episodeId) ?? null : null
    const epP = ep?.properties as Record<string, unknown> | undefined
    const episodeTitle =
      typeof epP?.episode_title === 'string' && epP.episode_title.trim()
        ? epP.episode_title.trim()
        : typeof epP?.title === 'string' && epP.title.trim()
          ? epP.title.trim()
          : null
    const pd =
      typeof epP?.publish_date === 'string' && epP.publish_date.trim()
        ? epP.publish_date.trim().slice(0, 10)
        : null
    rows.push({ id, type, text, episodeId, episodeTitle, publishDate: pd })
  }
  rows.sort((a, b) => {
    if (a.publishDate && b.publishDate) {
      if (a.publishDate < b.publishDate) return 1
      if (a.publishDate > b.publishDate) return -1
    } else if (a.publishDate) return -1
    else if (b.publishDate) return 1
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
  })
  return rows
})

function onPrefillSearch(): void {
  const q = subjectName.value.trim()
  if (!q) return
  emit('prefillSemanticSearch', { query: q })
}
</script>

<template>
  <div
    class="mx-3 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
    role="region"
    aria-label="Topic or entity"
    data-testid="topic-entity-view"
  >
    <div class="mt-1 flex shrink-0 items-baseline gap-2 border-b border-border pb-2">
      <span
        class="text-[10px] font-semibold uppercase tracking-wider text-muted"
        data-testid="topic-entity-view-kind"
      >{{ subjectKindLabel }}</span>
      <h2
        class="min-w-0 flex-1 truncate text-xs font-semibold text-surface-foreground"
        data-testid="topic-entity-view-name"
        :title="subjectName"
      >
        {{ subjectName }}
      </h2>
    </div>
    <div class="min-h-0 flex-1 space-y-3 overflow-y-auto px-1 py-2">
      <p
        v-if="subjectAliases"
        class="text-[11px] text-muted"
        data-testid="topic-entity-view-aliases"
      >
        Aliases: {{ subjectAliases }}
      </p>
      <p
        v-if="subjectDescription"
        class="text-[11px] leading-snug text-surface-foreground"
        data-testid="topic-entity-view-description"
      >
        {{ subjectDescription }}
      </p>
      <p
        v-if="timeline.total > 0 || timeline.undated > 0"
        class="text-[10px] text-muted"
        data-testid="topic-entity-view-stats"
      >
        {{ timeline.total }} dated mention{{ timeline.total === 1 ? '' : 's' }}
        across {{ timeline.episodeCount }} episode{{ timeline.episodeCount === 1 ? '' : 's' }}
        ({{ timeline.insightIds.length }} insight{{ timeline.insightIds.length === 1 ? '' : 's' }},
        {{ timeline.quoteIds.length }} quote{{ timeline.quoteIds.length === 1 ? '' : 's' }}<span
          v-if="timeline.undated > 0"
        >; {{ timeline.undated }} undated</span>).
      </p>
      <section aria-label="Mentions by month">
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Mentions by month
        </h3>
        <SubjectTimelineChart
          :timeline="timeline"
          aria-label="Mentions by month for this subject"
        />
      </section>
      <section
        v-if="mentionRows.length > 0"
        aria-label="Linked insights and quotes"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Mentions
        </h3>
        <ul
          class="space-y-1.5"
          data-testid="topic-entity-view-mentions"
        >
          <li
            v-for="row in mentionRows.slice(0, TOPIC_ENTITY_VIEW_MENTIONS_CAP)"
            :key="row.id"
            class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
          >
            <p class="font-medium text-surface-foreground">
              <span
                class="mr-1 inline-block rounded bg-overlay px-1 py-0.5 text-[9px] uppercase tracking-wider text-muted"
              >{{ row.type }}</span>
              {{ row.text || row.id }}
            </p>
            <p
              v-if="row.episodeTitle || row.publishDate"
              class="mt-0.5 text-[10px] text-muted"
            >
              <span v-if="row.episodeTitle">{{ row.episodeTitle }}</span>
              <span
                v-if="row.episodeTitle && row.publishDate"
              > · </span>
              <span v-if="row.publishDate">{{ row.publishDate }}</span>
            </p>
          </li>
        </ul>
        <p
          v-if="mentionRows.length > TOPIC_ENTITY_VIEW_MENTIONS_CAP"
          class="mt-1 text-[10px] text-muted"
          data-testid="topic-entity-view-mentions-overflow"
        >
          + {{ mentionRows.length - TOPIC_ENTITY_VIEW_MENTIONS_CAP }} more
        </p>
      </section>
      <p
        v-else-if="timeline.total === 0 && timeline.undated === 0"
        class="text-[11px] text-muted"
        data-testid="topic-entity-view-empty"
      >
        No insights or quotes link to this subject in the loaded graph.
      </p>
      <div class="flex shrink-0 flex-wrap gap-2 pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="topic-entity-view-go-graph"
          @click="emit('goGraph')"
        >
          Open in graph
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="topic-entity-view-prefill-search"
          @click="onPrefillSearch"
        >
          Prefill semantic search
        </button>
      </div>
    </div>
  </div>
</template>
