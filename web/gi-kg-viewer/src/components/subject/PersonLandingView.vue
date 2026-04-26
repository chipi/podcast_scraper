<script setup lang="ts">
/**
 * #672 — Person Landing rail panel. Reads ``subject.personId`` and shows
 * a Profile / Positions tab pair: Profile holds basic identity info +
 * mentions timeline; Positions lists ``SPOKEN_BY`` quotes attributed to
 * this person with episode context.
 */
import { computed, ref, watch } from 'vue'
import type { RawGraphNode } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useSubjectStore } from '../../stores/subject'
import {
  countPersonEntityIncidentEdges,
  findRawNodeInArtifact,
  normalizeGiEdgeType,
} from '../../utils/parsing'
import { logicalEpisodeIdFromGraphNodeId } from '../../utils/graphEpisodeMetadata'
import { buildSubjectMentionsTimeline } from '../../utils/subjectMentionsTimeline'
import SubjectTimelineChart from './SubjectTimelineChart.vue'

const emit = defineEmits<{
  goGraph: []
  closeSubject: []
  prefillSemanticSearch: [{ query: string }]
}>()

const artifacts = useArtifactsStore()
const subject = useSubjectStore()

type PersonTab = 'profile' | 'positions'
const activeTab = ref<PersonTab>('profile')

watch(
  () => subject.personId,
  () => {
    activeTab.value = 'profile'
  },
)

const personId = computed(() => subject.personId?.trim() || '')

const personNode = computed<RawGraphNode | null>(() => {
  const art = artifacts.displayArtifact
  const id = personId.value
  if (!art || !id) return null
  return findRawNodeInArtifact(art, id)
})

const personName = computed(() => {
  const n = personNode.value
  if (!n) return personId.value
  const p = n.properties as Record<string, unknown> | undefined
  const name = typeof p?.name === 'string' ? p.name.trim() : ''
  if (name) return name
  const label = typeof p?.label === 'string' ? p.label.trim() : ''
  return label || personId.value
})

const personAliases = computed(() => {
  const a = personNode.value?.properties?.aliases
  if (!Array.isArray(a)) return ''
  const parts = a
    .filter((x): x is string => typeof x === 'string' && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(', ') : ''
})

const personDescription = computed(() => {
  const d = personNode.value?.properties?.description
  return typeof d === 'string' && d.trim() ? d.trim() : ''
})

const edgeCounts = computed(() =>
  countPersonEntityIncidentEdges(artifacts.displayArtifact, personId.value),
)

const timeline = computed(() =>
  buildSubjectMentionsTimeline(artifacts.displayArtifact, personId.value),
)

interface PositionRow {
  id: string
  text: string
  episodeId: string | null
  episodeTitle: string | null
  publishDate: string | null
}

const positionRows = computed<PositionRow[]>(() => {
  const art = artifacts.displayArtifact
  const pid = personId.value
  if (!art || !pid) return []
  const episodes = new Map<string, RawGraphNode>()
  for (const n of art.data?.nodes ?? []) {
    if (!n || String(n.type) !== 'Episode') continue
    const lid = logicalEpisodeIdFromGraphNodeId(String(n.id ?? ''))
    if (lid) episodes.set(lid, n)
  }
  const seen = new Set<string>()
  const rows: PositionRow[] = []
  for (const e of art.data?.edges ?? []) {
    if (!e) continue
    const ty = normalizeGiEdgeType(e.type)
    if (ty !== 'spoken_by') continue
    const to = String(e.to ?? '').trim()
    if (to !== pid) continue
    const quoteId = String(e.from ?? '').trim()
    if (!quoteId || seen.has(quoteId)) continue
    seen.add(quoteId)
    const q = findRawNodeInArtifact(art, quoteId)
    if (!q || String(q.type) !== 'Quote') continue
    const p = q.properties as Record<string, unknown> | undefined
    const text =
      typeof p?.text === 'string' && p.text.trim() ? p.text.trim() : ''
    const episodeId =
      typeof p?.episode_id === 'string' && p.episode_id.trim()
        ? p.episode_id.trim()
        : null
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
    rows.push({ id: quoteId, text, episodeId, episodeTitle, publishDate: pd })
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

function tabClass(active: boolean): string {
  const base =
    'flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors'
  return active
    ? `${base} bg-primary text-primary-foreground`
    : `${base} text-elevated-foreground hover:bg-overlay`
}

function onPrefillSearch(): void {
  const q = personName.value.trim()
  if (!q) return
  emit('prefillSemanticSearch', { query: q })
}
</script>

<template>
  <div
    class="mx-3 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
    role="region"
    aria-label="Person"
    data-testid="person-landing-view"
  >
    <div class="mt-1 flex shrink-0 items-baseline gap-2 border-b border-border pb-2">
      <span
        class="text-[10px] font-semibold uppercase tracking-wider text-muted"
      >Person</span>
      <h2
        class="min-w-0 flex-1 truncate text-xs font-semibold text-surface-foreground"
        data-testid="person-landing-view-name"
        :title="personName"
      >
        {{ personName }}
      </h2>
    </div>
    <nav
      class="flex shrink-0 gap-1 border-b border-border bg-elevated/50 px-2 py-1.5"
      role="tablist"
      aria-label="Person sections"
    >
      <button
        id="person-landing-tab-profile"
        type="button"
        role="tab"
        :class="tabClass(activeTab === 'profile')"
        :aria-selected="activeTab === 'profile'"
        aria-controls="person-landing-panel-profile"
        :tabindex="activeTab === 'profile' ? 0 : -1"
        data-testid="person-landing-tab-profile"
        @click="activeTab = 'profile'"
      >
        Profile
      </button>
      <button
        id="person-landing-tab-positions"
        type="button"
        role="tab"
        :class="tabClass(activeTab === 'positions')"
        :aria-selected="activeTab === 'positions'"
        aria-controls="person-landing-panel-positions"
        :tabindex="activeTab === 'positions' ? 0 : -1"
        data-testid="person-landing-tab-positions"
        @click="activeTab = 'positions'"
      >
        Positions
      </button>
    </nav>
    <div
      v-show="activeTab === 'profile'"
      id="person-landing-panel-profile"
      role="tabpanel"
      aria-labelledby="person-landing-tab-profile"
      data-testid="person-landing-panel-profile"
      class="min-h-0 flex-1 space-y-3 overflow-y-auto px-1 py-2"
    >
      <p
        v-if="personAliases"
        class="text-[11px] text-muted"
        data-testid="person-landing-aliases"
      >
        Aliases: {{ personAliases }}
      </p>
      <p
        v-if="personDescription"
        class="text-[11px] leading-snug text-surface-foreground"
        data-testid="person-landing-description"
      >
        {{ personDescription }}
      </p>
      <p
        class="text-[10px] text-muted"
        data-testid="person-landing-edge-counts"
      >
        In this graph: {{ edgeCounts.spokenByQuotes }}
        attributed quote{{ edgeCounts.spokenByQuotes === 1 ? '' : 's' }} ·
        {{ edgeCounts.spokeInEpisodes }}
        episode link{{ edgeCounts.spokeInEpisodes === 1 ? '' : 's' }}.
      </p>
      <section aria-label="Mentions by month">
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Mentions by month
        </h3>
        <SubjectTimelineChart
          :timeline="timeline"
          aria-label="Mentions by month for this person"
        />
      </section>
      <div class="flex shrink-0 flex-wrap gap-2 pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="person-landing-go-graph"
          @click="emit('goGraph')"
        >
          Open in graph
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="person-landing-prefill-search"
          @click="onPrefillSearch"
        >
          Prefill semantic search
        </button>
      </div>
    </div>
    <div
      v-show="activeTab === 'positions'"
      id="person-landing-panel-positions"
      role="tabpanel"
      aria-labelledby="person-landing-tab-positions"
      data-testid="person-landing-panel-positions"
      class="min-h-0 flex-1 space-y-2 overflow-y-auto px-1 py-2"
    >
      <p
        v-if="positionRows.length === 0"
        class="text-[11px] text-muted"
        data-testid="person-landing-positions-empty"
      >
        No attributed quotes in the loaded graph.
      </p>
      <ul
        v-else
        class="space-y-1.5"
        data-testid="person-landing-positions"
      >
        <li
          v-for="row in positionRows.slice(0, 50)"
          :key="row.id"
          class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
        >
          <blockquote class="border-l-2 border-primary/40 pl-2 text-surface-foreground">
            {{ row.text || row.id }}
          </blockquote>
          <p
            v-if="row.episodeTitle || row.publishDate"
            class="mt-0.5 text-[10px] text-muted"
          >
            <span v-if="row.episodeTitle">{{ row.episodeTitle }}</span>
            <span v-if="row.episodeTitle && row.publishDate"> · </span>
            <span v-if="row.publishDate">{{ row.publishDate }}</span>
          </p>
        </li>
      </ul>
      <p
        v-if="positionRows.length > 50"
        class="text-[10px] text-muted"
        data-testid="person-landing-positions-overflow"
      >
        + {{ positionRows.length - 50 }} more
      </p>
    </div>
  </div>
</template>
