<script setup lang="ts">
/**
 * #672 — Topic / Entity rail panel ("TEV"). Reads the focused subject id
 * from the subject store, looks it up in the merged GI/KG slice, and shows
 * name + aliases + a monthly mentions timeline + the linked insights/quotes.
 *
 * Mounts inside ``SubjectRail`` for ``subject.kind === 'topic'``. Entity
 * subjects still flow through ``focusGraphNode`` → existing ``NodeDetail``.
 */
import { computed, ref, watch } from 'vue'
import type { RawGraphNode } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import {
  fetchCrossShow,
  fetchRelatedTopics,
  fetchTopicEntities,
  fetchWhoSaid,
  type RelatedNode,
} from '../../api/relationalApi'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import { StaleGeneration } from '../../utils/staleGeneration'
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
const shell = useShellStore()
const subject = useSubjectStore()

const subjectId = computed(() => subject.topicId?.trim() || '')

/**
 * PRD-033 FR4.2 — cross-show coverage + key voices for this topic, from the
 * relational-query layer (RFC-094). Fetched async on subject change, skeleton-first
 * (RFC-094 OQ-3); a StaleGeneration gate drops responses for a superseded subject.
 */
interface CrossShowRow {
  showId: string
  insight: RelatedNode
}
interface VoiceRow {
  personId: string
  insights: RelatedNode[]
}
const crossShowLoading = ref(false)
const crossShowError = ref<string | null>(null)
const crossShowRows = ref<CrossShowRow[]>([])
const voicesLoading = ref(false)
const voicesError = ref<string | null>(null)
const voiceRows = ref<VoiceRow[]>([])
const entitiesLoading = ref(false)
const entityRows = ref<RelatedNode[]>([])
const relatedTopicRows = ref<RelatedNode[]>([]) // #1055 — topics that share insights
const relationalGate = new StaleGeneration()

// RFC-088 chunk 6c: enrichment-layer signals for the focused topic.
interface CoOccurrenceRow {
  topic_id: string
  topic_label?: string
  episode_count: number
}
interface VelocityRow {
  topic_id: string
  topic_label?: string
  velocity_last_over_6mo: number
  monthly_counts: Record<string, number>
  total: number
}
const cooccurrenceRows = ref<CoOccurrenceRow[]>([])
const velocityRow = ref<VelocityRow | null>(null)
const velocityEffectiveLastMonth = ref<string | null>(null)
const enrichmentLoaded = ref(false)

function currentYearMonthUtc(): string {
  const d = new Date()
  const y = d.getUTCFullYear()
  const m = String(d.getUTCMonth() + 1).padStart(2, '0')
  return `${y}-${m}`
}
const COOCC_TOP_N = 8

function shortId(id: string): string {
  return id.replace(/^(podcast|person|topic|org):/, '').replace(/[-_]/g, ' ').trim() || id
}

function resetRelational(): void {
  crossShowRows.value = []
  crossShowError.value = null
  voiceRows.value = []
  voicesError.value = null
  entityRows.value = []
  relatedTopicRows.value = []
  cooccurrenceRows.value = []
  velocityRow.value = null
  velocityEffectiveLastMonth.value = null
  enrichmentLoaded.value = false
}

async function loadEnrichmentSignals(topicId: string): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!topicId || !root) return
  try {
    const [coOcc, velocity] = await Promise.all([
      fetchCachedCorpusEnvelope<{ pairs: Array<{ topic_a_id: string; topic_b_id: string; topic_a_label?: string; topic_b_label?: string; episode_count: number }> }>(
        root,
        'topic_cooccurrence_corpus',
      ).catch(() => null),
      fetchCachedCorpusEnvelope<{ topics: VelocityRow[]; effective_last_month?: string | null }>(
        root,
        'temporal_velocity',
      ).catch(() => null),
    ])
    enrichmentLoaded.value = true
    if (coOcc?.data?.pairs) {
      const partners: CoOccurrenceRow[] = []
      for (const p of coOcc.data.pairs) {
        if (p.topic_a_id === topicId) {
          partners.push({ topic_id: p.topic_b_id, topic_label: p.topic_b_label, episode_count: p.episode_count })
        } else if (p.topic_b_id === topicId) {
          partners.push({ topic_id: p.topic_a_id, topic_label: p.topic_a_label, episode_count: p.episode_count })
        }
      }
      partners.sort((a, b) => b.episode_count - a.episode_count)
      cooccurrenceRows.value = partners.slice(0, COOCC_TOP_N)
    }
    if (velocity?.data?.topics) {
      velocityRow.value = velocity.data.topics.find((t) => t.topic_id === topicId) ?? null
      velocityEffectiveLastMonth.value = velocity.data.effective_last_month ?? null
    }
  } catch {
    /* enrichment signals are best-effort; never break the rail */
  }
}

async function loadRelational(topicId: string): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!topicId || !root || !shell.healthStatus) {
    resetRelational()
    return
  }
  const seq = relationalGate.bump()
  crossShowLoading.value = true
  voicesLoading.value = true
  entitiesLoading.value = true
  resetRelational()
  try {
    const [cross, who, ents, related] = await Promise.all([
      fetchCrossShow(root, topicId).catch((e) => ({ error: String(e?.message ?? e), groups: {} })),
      fetchWhoSaid(root, topicId).catch((e) => ({ error: String(e?.message ?? e), groups: {} })),
      fetchTopicEntities(root, topicId).catch(() => ({ results: [] as RelatedNode[] })),
      fetchRelatedTopics(root, topicId).catch(() => ({ results: [] as RelatedNode[] })),
    ])
    if (relationalGate.isStale(seq)) return
    relatedTopicRows.value = related.results ?? []
    crossShowError.value = cross.error ?? null
    crossShowRows.value = Object.entries(cross.groups ?? {})
      .map(([showId, insights]) => ({ showId, insight: insights[0] }))
      .filter((r): r is CrossShowRow => r.insight != null)
    voicesError.value = who.error ?? null
    voiceRows.value = Object.entries(who.groups ?? {})
      .map(([personId, insights]) => ({ personId, insights }))
      .filter((r) => r.insights.length > 0)
    entityRows.value = ents.results ?? []
  } finally {
    if (relationalGate.isCurrent(seq)) {
      crossShowLoading.value = false
      voicesLoading.value = false
      entitiesLoading.value = false
    }
  }
}

watch(subjectId, (id) => void loadRelational(id), { immediate: true })
watch(subjectId, (id) => void loadEnrichmentSignals(id), { immediate: true })

function onClickVoice(personId: string): void {
  if (personId) subject.focusPerson(personId)
}

function onClickEntity(entity: RelatedNode): void {
  if (!entity.id) return
  if (entity.type === 'person') subject.focusPerson(entity.id)
  else subject.focusEntity(entity.id)
}

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
  // RFC-097 v3.0: Organization is a first-class typed node — render under
  // the shared "Entity" label alongside Person / legacy Entity. Without
  // this branch, Organization subjects fall through to the generic
  // "Subject" label.
  if (t === 'Entity' || t === 'Person' || t === 'Organization') return 'Entity'
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
    class="mx-3 flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden"
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
    <div class="min-h-0 w-full min-w-0 flex-1 space-y-3 overflow-y-auto py-2">
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
      <!-- RFC-088 chunk 6c: enrichment-layer signals (velocity + co-occurrence chips). -->
      <section
        v-if="enrichmentLoaded && (velocityRow || cooccurrenceRows.length)"
        class="w-full min-w-0 rounded border border-default bg-overlay/40 p-2"
        aria-label="Enrichment signals"
        data-testid="topic-entity-view-enrichment-signals"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Enrichment signals
        </h3>
        <div
          v-if="velocityRow"
          class="mb-1 flex items-center gap-2 text-[10px]"
          data-testid="topic-entity-view-velocity"
        >
          <span class="text-muted">Velocity (last / 6-mo avg):</span>
          <span
            class="rounded px-2 py-0.5 font-mono"
            :class="velocityRow.velocity_last_over_6mo > 1.5 ? 'bg-emerald-700/30 text-emerald-300' : velocityRow.velocity_last_over_6mo < 0.5 ? 'bg-rose-700/30 text-rose-300' : 'bg-overlay text-muted'"
          >{{ velocityRow.velocity_last_over_6mo.toFixed(2) }}×</span>
          <span class="text-muted">· {{ velocityRow.total }} mentions / 12-mo</span>
          <span
            v-if="velocityEffectiveLastMonth && velocityEffectiveLastMonth !== currentYearMonthUtc()"
            class="text-muted italic"
            data-testid="topic-entity-view-velocity-as-of"
            :title="`Corpus data ends at ${velocityEffectiveLastMonth}; velocity is computed against that month rather than the current calendar month.`"
          >· as of {{ velocityEffectiveLastMonth }}</span>
        </div>
        <div v-if="cooccurrenceRows.length" data-testid="topic-entity-view-cooccurrence">
          <p class="mb-1 text-[10px] text-muted">Co-occurs with</p>
          <div class="flex flex-wrap gap-1">
            <button
              v-for="r in cooccurrenceRows"
              :key="r.topic_id"
              type="button"
              class="rounded border border-default bg-overlay px-2 py-0.5 text-[10px] hover:bg-overlay-2"
              :data-testid="`topic-entity-view-cooccurrence-chip-${r.topic_id}`"
              @click="subject.focusTopic(r.topic_id)"
            >
              {{ r.topic_label || shortId(r.topic_id) }}
              <span class="ml-1 text-muted">·{{ r.episode_count }}</span>
            </button>
          </div>
        </div>
      </section>
      <section
        class="w-full min-w-0"
        aria-label="Mentions by month"
      >
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
        class="w-full min-w-0"
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

      <!-- PRD-033 FR4.2 — cross-show coverage (the corpus differentiator). -->
      <section
        v-if="crossShowLoading || crossShowError || crossShowRows.length"
        class="w-full min-w-0"
        aria-label="Across shows"
        data-testid="tev-cross-show"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Across shows
        </h3>
        <p
          v-if="crossShowLoading"
          data-testid="tev-cross-show-loading"
          class="text-[10px] text-muted"
        >
          Loading…
        </p>
        <p
          v-else-if="crossShowError"
          class="text-[10px] text-warning"
        >
          {{ crossShowError }}
        </p>
        <ul
          v-else
          class="space-y-1.5"
          data-testid="tev-cross-show-list"
        >
          <li
            v-for="row in crossShowRows"
            :key="row.showId"
            data-testid="tev-cross-show-row"
            class="rounded border-l-2 border-primary/40 pl-2 text-[11px] leading-snug"
          >
            <p class="font-semibold text-surface-foreground">{{ shortId(row.showId) }}</p>
            <p
              class="text-muted line-clamp-2"
              :title="row.insight.text"
            >
              {{ row.insight.text }}
            </p>
          </li>
        </ul>
      </section>

      <!-- #1055 — related topics (topics that share insights with this one). -->
      <section
        v-if="relatedTopicRows.length"
        class="w-full min-w-0"
        aria-label="Related topics"
        data-testid="tev-related-topics"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Related topics
        </h3>
        <div class="flex flex-wrap gap-1">
          <span
            v-for="t in relatedTopicRows"
            :key="t.id"
            class="rounded bg-overlay px-1.5 py-0.5 text-[10px] text-surface-foreground"
            data-testid="tev-related-topic-chip"
          >{{ t.text }}</span>
        </div>
      </section>

      <!-- PRD-033 FR4.2 — key voices (Person→Insight for this topic). -->
      <section
        v-if="voicesLoading || voicesError || voiceRows.length"
        class="w-full min-w-0"
        aria-label="Key voices"
        data-testid="tev-voices"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Key voices
        </h3>
        <p
          v-if="voicesLoading"
          class="text-[10px] text-muted"
        >
          Loading…
        </p>
        <p
          v-else-if="voicesError"
          class="text-[10px] text-warning"
        >
          {{ voicesError }}
        </p>
        <ul
          v-else
          class="space-y-1.5"
          data-testid="tev-voices-list"
        >
          <li
            v-for="row in voiceRows"
            :key="row.personId"
            data-testid="tev-voice-row"
            class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
          >
            <button
              type="button"
              data-testid="tev-voice-link"
              class="rounded font-semibold text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              :title="`Open Person panel for ${shortId(row.personId)}`"
              @click="onClickVoice(row.personId)"
            >{{ shortId(row.personId) }}</button>
            <span class="ml-1 text-[10px] text-muted">({{ row.insights.length }})</span>
            <p
              v-if="row.insights[0]?.text"
              class="mt-0.5 text-muted line-clamp-2"
              :title="row.insights[0]?.text"
            >
              {{ row.insights[0]?.text }}
            </p>
          </li>
        </ul>
      </section>

      <!-- PRD-033 FR4.2 — entities involved (Insight→MENTIONS→Entity), most-mentioned first. -->
      <section
        v-if="entitiesLoading || entityRows.length"
        class="w-full min-w-0"
        aria-label="Entities involved"
        data-testid="tev-entities"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Entities involved
        </h3>
        <p
          v-if="entitiesLoading"
          class="text-[10px] text-muted"
        >
          Loading…
        </p>
        <div
          v-else
          class="flex flex-wrap gap-1"
          data-testid="tev-entities-list"
        >
          <button
            v-for="ent in entityRows"
            :key="ent.id"
            type="button"
            data-testid="tev-entity-chip"
            class="rounded border border-border bg-elevated/40 px-1.5 py-0.5 text-[10px] text-surface-foreground hover:border-primary hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            :title="`Open panel for ${ent.text || shortId(ent.id)}`"
            @click="onClickEntity(ent)"
          >{{ ent.text || shortId(ent.id) }}</button>
        </div>
      </section>

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
