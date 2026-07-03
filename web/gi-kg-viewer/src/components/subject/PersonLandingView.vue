<script setup lang="ts">
/**
 * #672 — Person Landing rail panel. Reads ``subject.personId`` and shows
 * a Profile / Positions tab pair: Profile holds identity + Connections
 * (topics / co-speakers from the relational layer); Positions lists the
 * person's stated positions and quotes.
 *
 * #909 — the Positions tab's "All positions" lens surfaces the person's quotes
 * across ALL episodes (from the CIL ``person_profile`` endpoint), not just the
 * episodes currently merged into the loaded graph — so it scales and resolves
 * out-of-slice. This is the corpus-wide "what X said across episodes" payoff of
 * the #875/#876 diarization work.
 *
 * #1048 — restructured to be the shared Person Landing shell per PRD-029.
 * Tabs are now "Person Profile" (the aggregate person view; inherits all
 * shipped #672 / #909 / #1055 content) and "Position Tracker" (placeholder
 * for #1049 / PRD-028 Person × Topic over-time drill-in). Identity header
 * gains role + episode count + organization chips. New ranked topic-overview
 * list under Person Profile counts ``ABOUT(Insight→Topic)`` edges whose
 * Insight ``MENTIONS_PERSON`` this Person.
 */
import { computed, ref, watch } from 'vue'
import type { RawGraphNode } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
// #1075 chunk 4 — relational endpoints go through the LRU+TTL panel
// cache so re-focusing a recently-viewed Person feels instant. Same
// signatures as the wrapped fns; the cache invalidates on corpusPath
// change.
import {
  cachedFetchCoSpeakers as fetchCoSpeakers,
  cachedFetchPersonTopics as fetchPersonTopics,
  cachedFetchPositions as fetchPositions,
  cachedFetchPersonProfile as fetchPersonProfile,
} from '../../composables/useRelationalCache'
import type { RelatedNode } from '../../api/relationalApi'
import { StaleGeneration } from '../../utils/staleGeneration'
import {
  findRawNodeInArtifactByIdOrPrefixed,
  personEpisodeAppearances,
  personInsightsByTopic,
  personRoleFromNode,
  rankedPersonOrganizations,
  rankedPersonTopicMentions,
} from '../../utils/parsing'
import { stripLayerPrefixesForCil } from '../../utils/mergeGiKg'
import PositionTrackerPanel from './PositionTrackerPanel.vue'
import PersonInitialAvatar from '../shared/PersonInitialAvatar.vue'


const emit = defineEmits<{
  goGraph: []
  closeSubject: []
  prefillSemanticSearch: [{ query: string }]
}>()

// Embeddable-flat: when folded into NodeDetail's Details tab, ``embedded`` hides
// the standalone header/footer chrome and ``subjectIdOverride`` supplies the id
// (the subject store now opens Person nodes through the unified node view).
const props = withDefaults(
  defineProps<{
    embedded?: boolean
    subjectIdOverride?: string
    // Which slice to render when embedded in NodeDetail: 'profile' = slim
    // identity/context for the Details tab; 'positions' = the two-lens
    // (By topic · All positions) view for the Position Tracker tab. 'full' keeps
    // the whole standalone layout (used by tests).
    view?: 'full' | 'profile' | 'positions'
  }>(),
  { embedded: false, subjectIdOverride: '', view: 'full' },
)

// Position Tracker tab (view='positions') — two lenses mirroring the topic
// Timeline tab's Episodes/Mentions. 'by_topic' = Insights voiced grouped;
// 'all' = the flat Stated positions list, paginated.
const positionsLens = ref<'by_topic' | 'all'>('by_topic')
const POSITIONS_PAGE_SIZE = 10
const positionsPage = ref(1)

const artifacts = useArtifactsStore()
const shell = useShellStore()
const subject = useSubjectStore()

/** Person subject id — from the embed prop (NodeDetail fold) or the store. */
const personId = computed(() => props.subjectIdOverride?.trim() || subject.personId?.trim() || '')

// #1048 — tab vocabulary aligned with PRD-028 / PRD-029. ``profile`` is the
// aggregate Person Profile view (PRD-029); ``position_tracker`` is the
// per-topic drill-in (PRD-028, filled by follow-up #1049).
type PersonTab = 'profile' | 'position_tracker'
const activeTab = ref<PersonTab>('profile')

// Person enrichment signals (grounding rate / co-appearances / contradictions)
// and the mentions-by-month timeline live in the Signals tab
// (NodeEnrichmentSection + the rail's signalsTimeline) — not duplicated here.

watch(personId, () => {
  activeTab.value = 'profile'
})

const personNode = computed<RawGraphNode | null>(() => {
  const art = artifacts.displayArtifact
  const id = personId.value
  if (!art || !id) return null
  return findRawNodeInArtifactByIdOrPrefixed(art, id)
})

/** Actual graph node id (prefixed form after KG merge) for edge lookups. */
const personGraphNodeId = computed<string | null>(() => {
  const n = personNode.value
  if (!n || n.id == null) return personId.value || null
  return String(n.id)
})

/**
 * PRD-033 FR4.1 — the insights this person *stated* (Person→STATES→Insight), from
 * the relational-query layer (RFC-094). Distinct from the in-graph `SPOKEN_BY` quotes
 * below; this is the synthesized-position view. Fetched async on subject change,
 * skeleton-first; a StaleGeneration gate drops superseded responses.
 */
const statedLoading = ref(false)
const statedError = ref<string | null>(null)
const statedRows = ref<RelatedNode[]>([])
const statedGate = new StaleGeneration()

async function loadStatedPositions(rawId: string): Promise<void> {
  const id = rawId.trim()
  const root = shell.corpusPath?.trim()
  if (!id || !root || !shell.healthStatus) {
    statedRows.value = []
    statedError.value = null
    return
  }
  const seq = statedGate.bump()
  statedLoading.value = true
  statedRows.value = []
  statedError.value = null
  try {
    const body = await fetchPositions(root, id)
    if (statedGate.isStale(seq)) return
    statedError.value = body.error ?? null
    statedRows.value = body.results ?? []
  } catch (e) {
    if (statedGate.isStale(seq)) return
    statedError.value = e instanceof Error ? e.message : String(e)
    statedRows.value = []
  } finally {
    if (statedGate.isCurrent(seq)) statedLoading.value = false
  }
}

watch(
  personGraphNodeId,
  (id) => void loadStatedPositions(id ?? ''),
  { immediate: true },
)

/**
 * #1055 — connections: the person's topics (structural graph lens) + co-speakers (people
 * who engage the same topics). Both from the relational layer; skeleton-first, stale-gated.
 */
const topicsRows = ref<RelatedNode[]>([])
const coSpeakersRows = ref<RelatedNode[]>([])
const connectionsGate = new StaleGeneration()

async function loadConnections(rawId: string): Promise<void> {
  const id = rawId.trim()
  const root = shell.corpusPath?.trim()
  if (!id || !root || !shell.healthStatus) {
    topicsRows.value = []
    coSpeakersRows.value = []
    return
  }
  const seq = connectionsGate.bump()
  topicsRows.value = []
  coSpeakersRows.value = []
  try {
    // Relational endpoints canonicalize on the corpus id (person:…); strip the
    // graph layer prefix (g:/k:) first or they return empty (id-format mismatch).
    const canonical = stripLayerPrefixesForCil(id)
    const [t, c] = await Promise.all([
      fetchPersonTopics(root, canonical),
      fetchCoSpeakers(root, canonical),
    ])
    if (connectionsGate.isStale(seq)) return
    topicsRows.value = t.results ?? []
    coSpeakersRows.value = c.results ?? []
  } catch {
    if (connectionsGate.isStale(seq)) return
    topicsRows.value = []
    coSpeakersRows.value = []
  }
}

watch(personGraphNodeId, (id) => void loadConnections(id ?? ''), { immediate: true })

// Prolific speakers (esp. hosts) accumulate many topics + co-speakers. The
// relational endpoints already return these ranked, so cap the display to a
// preview and let the user expand. Collapse again on person change.
const TOPICS_PREVIEW = 12
const COSPEAKERS_PREVIEW = 8
const topicsExpanded = ref(false)
const coSpeakersExpanded = ref(false)
// FB7 — flag which person topics are cluster compounds (from topic_clusters.json)
// so the flat list distinguishes broad cluster topics from specific ones (Digest
// parity). Themes are a per-focused-topic co-occurrence concept, not a per-topic
// attribute, so they're not flagged here.
// A topic is a "cluster" topic when it's a member of any topic_clusters.json
// cluster — matched directly on member topic_id, not the graph-materialised
// compound (which findTopicClusterContextForGraphNode requires and which most
// clusters lack). Returns the cluster's canonical label for the tooltip.
const topicClusterLabelById = computed(() => {
  const m = new Map<string, string>()
  for (const cl of artifacts.topicClustersDoc?.clusters ?? []) {
    if (!cl || typeof cl !== 'object') continue
    const label =
      typeof cl.canonical_label === 'string' && cl.canonical_label.trim()
        ? cl.canonical_label.trim()
        : ''
    for (const mem of Array.isArray(cl.members) ? cl.members : []) {
      const tid = mem && typeof mem.topic_id === 'string' ? mem.topic_id.trim() : ''
      if (tid && !m.has(tid)) m.set(tid, label || tid)
    }
  }
  return m
})
const flaggedTopics = computed(() =>
  topicsRows.value.map((t) => ({
    ...t,
    clusterLabel: topicClusterLabelById.value.get(t.id) ?? null,
  })),
)
const visibleTopics = computed(() =>
  topicsExpanded.value ? flaggedTopics.value : flaggedTopics.value.slice(0, TOPICS_PREVIEW),
)
const visibleCoSpeakers = computed(() =>
  coSpeakersExpanded.value
    ? coSpeakersRows.value
    : coSpeakersRows.value.slice(0, COSPEAKERS_PREVIEW),
)
watch(personGraphNodeId, () => {
  topicsExpanded.value = false
  coSpeakersExpanded.value = false
})

/**
 * #909 — corpus-wide quotes this person spoke across ALL episodes, from the CIL
 * ``person_profile`` endpoint (``GET /api/persons/{id}/brief``). Resolves the
 * person across the whole corpus (incl. #852 name variants), independent of the
 * loaded graph slice — this is the source for the Positions "All positions"
 * lens. Async, skeleton-first, stale-gated; renders only when an API corpus is
 * connected (no-op in local-file mode).
 */
interface CorpusQuoteRow {
  id: string
  text: string
  episodeId: string | null
  episodeTitle: string | null
}

/** Episode title embedded in a quote's transcript_ref, e.g.
 *  ``transcripts/0006 - Trading disruption_20260613-…`` → ``Trading disruption``.
 *  The /brief payload carries no episode title and the quote's episode_id is a
 *  compact hex uuid that doesn't match the graph's dashed ids, so this is the
 *  one readable episode label available client-side. */
function episodeTitleFromTranscriptRef(ref: unknown): string | null {
  if (typeof ref !== 'string') return null
  const m = ref.match(/\/\d+\s*-\s*(.+?)_\d{6,}/)
  return m ? m[1].trim() || null : null
}
const corpusLoading = ref(false)
const corpusError = ref<string | null>(null)
const corpusQuotes = ref<CorpusQuoteRow[]>([])
const corpusGate = new StaleGeneration()

const corpusEpisodeCount = computed(() => {
  const eps = new Set<string>()
  for (const q of corpusQuotes.value) if (q.episodeId) eps.add(q.episodeId)
  return eps.size
})

async function loadCorpusQuotes(rawId: string): Promise<void> {
  const id = rawId.trim()
  const root = shell.corpusPath?.trim()
  if (!id || !root || !shell.healthStatus) {
    corpusQuotes.value = []
    corpusError.value = null
    return
  }
  const seq = corpusGate.bump()
  corpusLoading.value = true
  corpusQuotes.value = []
  corpusError.value = null
  try {
    const body = await fetchPersonProfile(root, id)
    if (corpusGate.isStale(seq)) return
    const rows: CorpusQuoteRow[] = []
    const seen = new Set<string>()
    for (const r of body.quotes ?? []) {
      const node = r?.quote as Record<string, unknown> | undefined
      const qid = node && node.id != null ? String(node.id) : ''
      if (!qid || seen.has(qid)) continue
      seen.add(qid)
      const p = node?.properties as Record<string, unknown> | undefined
      const text = typeof p?.text === 'string' ? p.text.trim() : ''
      const episodeId =
        typeof r.episode_id === 'string' && r.episode_id.trim() ? r.episode_id.trim() : null
      rows.push({ id: qid, text, episodeId, episodeTitle: episodeTitleFromTranscriptRef(p?.transcript_ref) })
    }
    corpusQuotes.value = rows
  } catch (e) {
    if (corpusGate.isStale(seq)) return
    corpusError.value = e instanceof Error ? e.message : String(e)
    corpusQuotes.value = []
  } finally {
    if (corpusGate.isCurrent(seq)) corpusLoading.value = false
  }
}

watch(personId, (id) => void loadCorpusQuotes(id ?? ''), { immediate: true })

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

// #1048 — identity header additions per PRD-029.
const personRole = computed(() => personRoleFromNode(personNode.value))
const personRoleLabel = computed(() => {
  const r = personRole.value
  if (!r) return ''
  return r.charAt(0).toUpperCase() + r.slice(1)
})

// #1048 — ranked topic overview via ABOUT(Insight→Topic) chains whose Insight
// MENTIONS_PERSON this person. Distinct from the relational
// fetchPersonTopics() chips above (which use the structural person→topic lens).
const TOPIC_OVERVIEW_CAP = 10
const rankedTopics = computed(() =>
  rankedPersonTopicMentions(
    artifacts.displayArtifact,
    personGraphNodeId.value,
    TOPIC_OVERVIEW_CAP,
  ),
)

// #1048 — co-mentioned Organizations via MENTIONS_PERSON ⨯ MENTIONS_ORG join
// on the same Insight. Empty until KG fixtures contain MENTIONS_ORG edges
// (RFC-097 v2 strict subset; cloud profiles only).
const ORG_CHIPS_CAP = 10
const rankedOrgs = computed(() =>
  rankedPersonOrganizations(
    artifacts.displayArtifact,
    personGraphNodeId.value,
    ORG_CHIPS_CAP,
  ),
)

// #1050 — Episodes appeared in (UXS-010 section). SPOKE_IN-derived list,
// newest-first, undated entries sink. Empty when no SPOKE_IN edges exist
// for this Person.
const episodeAppearances = computed(() =>
  personEpisodeAppearances(artifacts.displayArtifact, personGraphNodeId.value),
)

// #1050 — Insights voiced grouped by Topic (UXS-010 section). Each Topic
// header reuses the #1049 entry point so the Profile tab is the canonical
// way into the Position Tracker (chains naturally with the Top Topics
// list above — same selectTopicForPositionTracker call).
const insightTopicGroups = computed(() =>
  personInsightsByTopic(artifacts.displayArtifact, personGraphNodeId.value),
)
// Per-group expand state (default collapsed so the Profile tab stays
// scannable on first open; the count + topic name is the summary line).
const expandedTopicGroups = ref<Set<string>>(new Set())
watch(
  () => `${personGraphNodeId.value}`,
  () => {
    expandedTopicGroups.value = new Set()
  },
)
function toggleTopicGroup(topicId: string): void {
  const next = new Set(expandedTopicGroups.value)
  if (next.has(topicId)) next.delete(topicId)
  else next.add(topicId)
  expandedTopicGroups.value = next
}

// "All positions" lens — paginate the corpus-wide quotes (server /brief) so a
// prolific speaker doesn't render hundreds of rows in one scroll (mirrors the
// topic Mentions pager). These scale and resolve out-of-slice, unlike the old
// client-graph SPOKEN_BY walk that only saw the loaded graph.
const positionsTotalPages = computed(() =>
  Math.max(1, Math.ceil(corpusQuotes.value.length / POSITIONS_PAGE_SIZE)),
)
const pagedCorpusQuotes = computed<CorpusQuoteRow[]>(() => {
  const start = (positionsPage.value - 1) * POSITIONS_PAGE_SIZE
  return corpusQuotes.value.slice(start, start + POSITIONS_PAGE_SIZE)
})
watch(personGraphNodeId, () => {
  positionsLens.value = 'by_topic'
  positionsPage.value = 1
})
watch(positionsLens, () => { positionsPage.value = 1 })

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

// #1049 — clicking a ranked-Topic row pivots the Position Tracker tab
// onto that (Person, Topic) pair and switches tabs. The same handler is
// the entry point #1050 will eventually use from Topic group headers.
function onPickTopicForPositionTracker(topicId: string): void {
  if (!topicId.trim()) return
  subject.selectTopicForPositionTracker(topicId)
  // In positions view the arc renders inline; in full standalone, switch the internal tab.
  if (props.view !== 'positions') activeTab.value = 'position_tracker'
}
</script>

<template>
  <div
    class="flex min-h-0 min-w-0 flex-1 flex-col"
    :class="props.embedded ? '' : 'mx-3 overflow-hidden'"
    role="region"
    aria-label="Person"
    data-testid="person-landing-view"
  >
    <div class="mt-1 shrink-0 border-b border-border pb-2">
      <div v-if="!props.embedded" class="flex items-baseline gap-2">
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
        <!-- #1048 — role badge (host / guest / mention) per PRD-029 FR1 -->
        <span
          v-if="personRoleLabel"
          class="rounded bg-overlay px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-surface-foreground"
          data-testid="person-landing-role"
          :data-role="personRole"
          :title="`Role: ${personRoleLabel}`"
        >
          {{ personRoleLabel }}
        </span>
        <button
          type="button"
          class="shrink-0 self-center rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
          data-testid="subject-rail-close"
          aria-label="Close person detail"
          @click="emit('closeSubject')"
        >
          ×
        </button>
      </div>
      <!-- #1048 / #1050 — episode-count signal. Derives from the same
           personEpisodeAppearances list rendered below so the at-a-glance
           count cannot disagree with the section (raw SPOKE_IN edge tallies
           are not deduplicated by target episode; the list helper is). -->
      <p
        v-if="episodeAppearances.length > 0"
        class="mt-1 text-[10px] text-muted"
        data-testid="person-landing-episode-count"
      >
        {{ episodeAppearances.length }}
        episode{{ episodeAppearances.length === 1 ? '' : 's' }}
      </p>
      <!-- #1048 — co-mentioned Organizations (PRD-029 FR1 affiliations) -->
      <div
        v-if="rankedOrgs.length"
        class="mt-1 flex flex-wrap gap-1"
        data-testid="person-landing-organizations"
      >
        <span
          v-for="org in rankedOrgs"
          :key="org.id"
          class="rounded bg-overlay px-1.5 py-0.5 text-[10px] text-surface-foreground"
          data-testid="person-landing-organization-chip"
          :title="`${org.name} · ${org.count} insight${org.count === 1 ? '' : 's'}`"
        >{{ org.name }}</span>
      </div>
    </div>
    <nav
      v-if="!props.embedded"
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
        Person Profile
      </button>
      <button
        id="person-landing-tab-position-tracker"
        type="button"
        role="tab"
        :class="tabClass(activeTab === 'position_tracker')"
        :aria-selected="activeTab === 'position_tracker'"
        aria-controls="person-landing-panel-position-tracker"
        :tabindex="activeTab === 'position_tracker' ? 0 : -1"
        data-testid="person-landing-tab-position-tracker"
        @click="activeTab = 'position_tracker'"
      >
        Position Tracker
      </button>
    </nav>
    <div
      v-show="props.embedded || activeTab === 'profile'"
      id="person-landing-panel-profile"
      role="tabpanel"
      aria-labelledby="person-landing-tab-profile"
      data-testid="person-landing-panel-profile"
      class="min-h-0 flex-1 space-y-3 overflow-y-auto px-1 py-2"
    >
      <!-- Profile-only sections: hidden in positions view -->
      <template v-if="props.view !== 'positions'">
        <!-- Role badge (host / guest / mentioned) — the embedded header (owned
             by NodeDetail) doesn't carry it, so surface it here so the reader
             knows how this person relates to the corpus. Host stands out. -->
        <span
          v-if="personRoleLabel"
          class="inline-flex items-center self-start rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
          :class="personRole === 'host'
            ? 'bg-primary text-primary-foreground'
            : 'bg-overlay text-surface-foreground'"
          data-testid="person-landing-role-embedded"
          :data-role="personRole"
        >{{ personRoleLabel }}</span>
        <!-- Connections first: the substance (topics discussed + who they
             speak with) sits at the top of the panel; identity/enrichment
             metadata follows below. -->
        <section aria-label="Connections" data-testid="person-landing-connections">
          <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
            Topics
          </h3>
          <div
            v-if="topicsRows.length"
            class="flex flex-wrap gap-1"
            data-testid="person-landing-topics"
          >
            <span
              v-for="t in visibleTopics"
              :key="t.id"
              class="inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[10px] text-surface-foreground"
              :class="t.clusterLabel ? '' : 'bg-overlay'"
              :style="t.clusterLabel ? { backgroundColor: 'color-mix(in srgb, var(--ps-kg) 22%, transparent)' } : undefined"
              :title="t.clusterLabel ? `Topic cluster: ${t.clusterLabel}` : undefined"
              :data-cluster="t.clusterLabel ? 'true' : undefined"
              data-testid="person-landing-topic-chip"
            ><span v-if="t.clusterLabel" aria-hidden="true" class="text-[9px] opacity-70">⧉</span>{{ t.text }}</span>
            <button
              v-if="topicsRows.length > TOPICS_PREVIEW"
              type="button"
              class="rounded px-1.5 py-0.5 text-[10px] font-medium text-primary hover:bg-overlay"
              data-testid="person-landing-topics-toggle"
              @click="topicsExpanded = !topicsExpanded"
            >{{ topicsExpanded ? 'Show less' : `+${topicsRows.length - TOPICS_PREVIEW} more` }}</button>
          </div>
          <p v-else class="text-[10px] text-muted" data-testid="person-landing-topics-empty">
            No topics for this voice yet.
          </p>
          <h3 class="mb-1 mt-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
            In the same conversation
          </h3>
          <div
            v-if="coSpeakersRows.length"
            class="flex flex-wrap gap-1"
            data-testid="person-landing-co-speakers"
          >
            <button
              v-for="p in visibleCoSpeakers"
              :key="p.id"
              type="button"
              class="inline-flex items-center gap-1 rounded bg-overlay py-0.5 pl-0.5 pr-1.5 text-[10px] text-surface-foreground hover:bg-overlay-2"
              data-testid="person-landing-co-speaker-chip"
              :title="`Open ${p.text}`"
              @click="subject.focusPerson(p.id)"
            >
              <PersonInitialAvatar :name="p.text" />
              {{ p.text }}
            </button>
            <button
              v-if="coSpeakersRows.length > COSPEAKERS_PREVIEW"
              type="button"
              class="self-center rounded px-1.5 py-0.5 text-[10px] font-medium text-primary hover:bg-overlay"
              data-testid="person-landing-co-speakers-toggle"
              @click="coSpeakersExpanded = !coSpeakersExpanded"
            >{{ coSpeakersExpanded ? 'Show less' : `+${coSpeakersRows.length - COSPEAKERS_PREVIEW} more` }}</button>
          </div>
          <p v-else class="text-[10px] text-muted" data-testid="person-landing-co-speakers-empty">
            No co-speakers share a topic with this voice yet.
          </p>
        </section>
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
      <!-- #1050 — UXS-010 "Episodes appeared in" — SPOKE_IN-derived list,
           newest-first. Replaces the prior numeric-only count signal in the
           identity header (we keep that count for at-a-glance, but expose
           the actual episodes here). -->
      <section
        v-if="episodeAppearances.length"
        aria-label="Episodes appeared in"
        data-testid="person-landing-episodes-appeared"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Episodes appeared in
        </h3>
        <ul
          class="space-y-0.5"
          data-testid="person-landing-episodes-appeared-list"
        >
          <li
            v-for="ep in episodeAppearances"
            :key="ep.episodeId"
            class="flex items-baseline justify-between gap-2 text-[11px] text-surface-foreground"
            data-testid="person-landing-episodes-appeared-row"
            :data-episode-id="ep.episodeId"
          >
            <span class="min-w-0 truncate" :title="ep.title ?? ep.episodeId">{{ ep.title || ep.episodeId }}</span>
            <span
              v-if="ep.publishDate"
              class="shrink-0 text-[10px] text-muted"
              data-testid="person-landing-episodes-appeared-date"
            >{{ ep.publishDate }}</span>
            <span
              v-else
              class="shrink-0 text-[10px] italic text-muted"
              data-testid="person-landing-episodes-appeared-date-unknown"
            >date unknown</span>
          </li>
        </ul>
      </section>

      </template>

      <!-- Topics discussed: only in full view -->
      <!-- #1050 — UXS-010 "Topics discussed" — ranked by ABOUT∩MENTIONS_PERSON
           insight count. Each row is a button that pivots the Position
           Tracker tab to (this Person, that Topic) — #1049 entry point. -->
      <section
        v-if="rankedTopics.length && props.view === 'full'"
        aria-label="Topics discussed"
        data-testid="person-landing-ranked-topics"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Topics discussed
        </h3>
        <ul class="space-y-0.5" data-testid="person-landing-ranked-topics-list">
          <li
            v-for="t in rankedTopics"
            :key="t.id"
            data-testid="person-landing-ranked-topic-row"
          >
            <button
              type="button"
              class="flex w-full items-baseline justify-between gap-2 rounded px-1 py-0.5 text-left text-[11px] text-surface-foreground hover:bg-overlay/60 focus-visible:bg-overlay/60 focus-visible:outline-none"
              data-testid="person-landing-ranked-topic-button"
              :title="`Open Position Tracker for ${t.name}`"
              :aria-label="`Open Position Tracker for ${t.name}`"
              @click="onPickTopicForPositionTracker(t.id)"
            >
              <span class="min-w-0 truncate">{{ t.name }}</span>
              <span
                class="shrink-0 rounded bg-overlay px-1.5 py-0.5 text-[10px] text-muted"
                data-testid="person-landing-ranked-topic-count"
              >{{ t.count }}</span>
            </button>
          </li>
        </ul>
      </section>

      <!-- Two-lens positions view (embedded in NodeDetail Position Tracker tab) -->
      <template v-if="props.view === 'positions'">
        <!-- Arc drill: topic selected → show arc, hide lens toggle -->
        <template v-if="subject.positionTrackerTopicId">
          <PositionTrackerPanel :person-id-override="personId" />
        </template>
        <template v-else>
          <!-- Lens segmented toggle -->
          <div
            role="tablist"
            aria-label="Positions view"
            class="mb-2 inline-flex shrink-0 rounded-md border border-border p-0.5"
            data-testid="person-landing-positions-lens"
          >
            <button
              type="button"
              role="tab"
              class="rounded px-2 py-0.5 text-[10px] font-medium transition-colors"
              :class="positionsLens === 'by_topic' ? 'bg-primary text-primary-foreground' : 'text-muted hover:bg-overlay'"
              :aria-selected="positionsLens === 'by_topic'"
              data-testid="person-landing-positions-lens-by-topic"
              @click="positionsLens = 'by_topic'"
            >
              By topic
            </button>
            <button
              type="button"
              role="tab"
              class="rounded px-2 py-0.5 text-[10px] font-medium transition-colors"
              :class="positionsLens === 'all' ? 'bg-primary text-primary-foreground' : 'text-muted hover:bg-overlay'"
              :aria-selected="positionsLens === 'all'"
              data-testid="person-landing-positions-lens-all"
              @click="positionsLens = 'all'"
            >
              All positions
            </button>
          </div>
          <!-- By topic: Insights voiced (gated by lens) -->
          <div v-show="positionsLens === 'by_topic'" data-testid="person-landing-positions-by-topic">
            <!-- #1050 — UXS-010 "Insights voiced (grouped by Topic)". Each Topic
                 header is a button that opens the Position Tracker for the
                 (Person, Topic) pair — same entry point as the ranked-Topics
                 list above so the user has one consistent affordance. -->
            <section
              v-if="insightTopicGroups.length"
              aria-label="Insights voiced grouped by topic"
              data-testid="person-landing-insights-voiced"
            >
              <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
                Insights voiced
              </h3>
              <ul
                class="space-y-1.5"
                data-testid="person-landing-insights-voiced-list"
              >
                <li
                  v-for="group in insightTopicGroups"
                  :key="group.topicId"
                  class="rounded border border-border bg-elevated/30 px-2 py-1.5"
                  data-testid="person-landing-insights-voiced-group"
                  :data-topic-id="group.topicId"
                >
                  <button
                    type="button"
                    class="flex w-full items-baseline justify-between gap-2 text-left text-[11px] font-semibold text-surface-foreground hover:text-primary focus-visible:text-primary focus-visible:outline-none"
                    :aria-label="`${group.topicName} — ${group.count} insight${group.count === 1 ? '' : 's'}. Open Position Tracker.`"
                    data-testid="person-landing-insights-voiced-topic-button"
                    @click="onPickTopicForPositionTracker(group.topicId)"
                  >
                    <span class="min-w-0 truncate" :title="group.topicName">
                      {{ group.topicName }}
                    </span>
                    <span
                      class="shrink-0 rounded bg-overlay px-1.5 py-0.5 text-[10px] font-normal text-muted"
                      data-testid="person-landing-insights-voiced-topic-count"
                    >{{ group.count }}</span>
                  </button>
                  <button
                    type="button"
                    class="mt-0.5 text-[10px] text-muted underline-offset-2 hover:underline focus-visible:underline"
                    data-testid="person-landing-insights-voiced-toggle"
                    :aria-expanded="expandedTopicGroups.has(group.topicId)"
                    @click="toggleTopicGroup(group.topicId)"
                  >
                    {{ expandedTopicGroups.has(group.topicId) ? 'Hide insights' : 'Show insights' }}
                  </button>
                  <ul
                    v-if="expandedTopicGroups.has(group.topicId)"
                    class="mt-1 space-y-1"
                    data-testid="person-landing-insights-voiced-rows"
                  >
                    <li
                      v-for="ins in group.insights"
                      :key="ins.insightId"
                      class="rounded bg-elevated/60 px-2 py-1 text-[11px] leading-snug text-surface-foreground"
                      data-testid="person-landing-insights-voiced-row"
                      :data-insight-type="ins.insightType ?? 'unknown'"
                    >
                      <p
                        v-if="ins.insightType"
                        class="mb-0.5 text-[9px] uppercase tracking-wider text-muted"
                        data-testid="person-landing-insights-voiced-row-type"
                      >{{ ins.insightType }}</p>
                      <p data-testid="person-landing-insights-voiced-row-text">{{ ins.text || ins.insightId }}</p>
                    </li>
                  </ul>
                </li>
              </ul>
            </section>
          </div>
          <!-- All positions: paginated stated + attributed quotes -->
          <div v-show="positionsLens === 'all'" data-testid="person-landing-positions-all">
            <!-- PRD-033 FR4.1 — synthesized positions this person stated (relational layer). -->
            <section
              v-if="statedLoading || statedError || statedRows.length"
              aria-label="Stated positions"
              data-testid="person-landing-stated"
            >
              <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
                Stated positions
              </h3>
              <p
                v-if="statedLoading"
                data-testid="person-landing-stated-loading"
                class="text-[11px] text-muted"
              >
                Loading…
              </p>
              <p
                v-else-if="statedError"
                class="text-[11px] text-warning"
              >
                {{ statedError }}
              </p>
              <ul
                v-else
                class="space-y-1.5"
                data-testid="person-landing-stated-list"
              >
                <li
                  v-for="row in statedRows"
                  :key="row.id"
                  data-testid="person-landing-stated-row"
                  class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
                >
                  <blockquote class="border-l-2 border-primary/40 pl-2 text-surface-foreground">
                    {{ row.text || row.id }}
                  </blockquote>
                </li>
              </ul>
            </section>

            <h3 class="mt-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
              Quotes across the corpus<span v-if="corpusEpisodeCount">
                · {{ corpusEpisodeCount }} episode{{ corpusEpisodeCount === 1 ? '' : 's' }}</span>
            </h3>
            <p
              v-if="corpusLoading"
              class="text-[11px] text-muted"
              data-testid="person-landing-corpus-loading"
            >
              Loading…
            </p>
            <p
              v-else-if="corpusError"
              class="text-[11px] text-warning"
            >
              {{ corpusError }}
            </p>
            <p
              v-else-if="corpusQuotes.length === 0"
              class="text-[11px] text-muted"
              data-testid="person-landing-positions-empty"
            >
              No attributed quotes for this voice yet.
            </p>
            <ul
              v-else
              class="space-y-1.5"
              data-testid="person-landing-positions"
            >
              <li
                v-for="row in pagedCorpusQuotes"
                :key="row.id"
                data-testid="person-landing-corpus-row"
                class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
              >
                <blockquote class="border-l-2 border-primary/40 pl-2 text-surface-foreground">
                  {{ row.text || row.id }}
                </blockquote>
                <p
                  v-if="row.episodeTitle"
                  class="mt-0.5 text-[10px] text-muted"
                >
                  {{ row.episodeTitle }}
                </p>
              </li>
            </ul>
            <!-- Paginator -->
            <div
              v-if="positionsTotalPages > 1"
              class="mt-2 flex items-center justify-between gap-2 text-[10px]"
              data-testid="person-landing-positions-pager"
            >
              <button
                type="button"
                class="rounded border border-border px-2 py-0.5 text-muted hover:bg-overlay disabled:opacity-40"
                :disabled="positionsPage === 1"
                data-testid="person-landing-positions-pager-prev"
                @click="positionsPage--"
              >
                ← Prev
              </button>
              <span class="text-muted" data-testid="person-landing-positions-pager-info">
                {{ positionsPage }} / {{ positionsTotalPages }}
              </span>
              <button
                type="button"
                class="rounded border border-border px-2 py-0.5 text-muted hover:bg-overlay disabled:opacity-40"
                :disabled="positionsPage >= positionsTotalPages"
                data-testid="person-landing-positions-pager-next"
                @click="positionsPage++"
              >
                Next →
              </button>
            </div>
          </div>
        </template>
      </template>

      <!-- Full view: insights voiced + stated positions stacked (current layout preserved) -->
      <template v-if="props.view === 'full'">
        <!-- #1050 — UXS-010 "Insights voiced (grouped by Topic)". Each Topic
             header is a button that opens the Position Tracker for the
             (Person, Topic) pair — same entry point as the ranked-Topics
             list above so the user has one consistent affordance. -->
        <section
          v-if="insightTopicGroups.length"
          aria-label="Insights voiced grouped by topic"
          data-testid="person-landing-insights-voiced"
        >
          <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
            Insights voiced
          </h3>
          <ul
            class="space-y-1.5"
            data-testid="person-landing-insights-voiced-list"
          >
            <li
              v-for="group in insightTopicGroups"
              :key="group.topicId"
              class="rounded border border-border bg-elevated/30 px-2 py-1.5"
              data-testid="person-landing-insights-voiced-group"
              :data-topic-id="group.topicId"
            >
              <button
                type="button"
                class="flex w-full items-baseline justify-between gap-2 text-left text-[11px] font-semibold text-surface-foreground hover:text-primary focus-visible:text-primary focus-visible:outline-none"
                :aria-label="`${group.topicName} — ${group.count} insight${group.count === 1 ? '' : 's'}. Open Position Tracker.`"
                data-testid="person-landing-insights-voiced-topic-button"
                @click="onPickTopicForPositionTracker(group.topicId)"
              >
                <span class="min-w-0 truncate" :title="group.topicName">
                  {{ group.topicName }}
                </span>
                <span
                  class="shrink-0 rounded bg-overlay px-1.5 py-0.5 text-[10px] font-normal text-muted"
                  data-testid="person-landing-insights-voiced-topic-count"
                >{{ group.count }}</span>
              </button>
              <button
                type="button"
                class="mt-0.5 text-[10px] text-muted underline-offset-2 hover:underline focus-visible:underline"
                data-testid="person-landing-insights-voiced-toggle"
                :aria-expanded="expandedTopicGroups.has(group.topicId)"
                @click="toggleTopicGroup(group.topicId)"
              >
                {{ expandedTopicGroups.has(group.topicId) ? 'Hide insights' : 'Show insights' }}
              </button>
              <ul
                v-if="expandedTopicGroups.has(group.topicId)"
                class="mt-1 space-y-1"
                data-testid="person-landing-insights-voiced-rows"
              >
                <li
                  v-for="ins in group.insights"
                  :key="ins.insightId"
                  class="rounded bg-elevated/60 px-2 py-1 text-[11px] leading-snug text-surface-foreground"
                  data-testid="person-landing-insights-voiced-row"
                  :data-insight-type="ins.insightType ?? 'unknown'"
                >
                  <p
                    v-if="ins.insightType"
                    class="mb-0.5 text-[9px] uppercase tracking-wider text-muted"
                    data-testid="person-landing-insights-voiced-row-type"
                  >{{ ins.insightType }}</p>
                  <p data-testid="person-landing-insights-voiced-row-text">{{ ins.text || ins.insightId }}</p>
                </li>
              </ul>
            </li>
          </ul>
        </section>

        <!-- #1050 — UXS-010 "Episodes appeared in" already in profile-only block above.
             PRD-033 FR4.1 — synthesized positions this person stated (relational layer). -->
        <section
          v-if="statedLoading || statedError || statedRows.length"
          aria-label="Stated positions"
          data-testid="person-landing-stated"
        >
          <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
            Stated positions
          </h3>
          <p
            v-if="statedLoading"
            data-testid="person-landing-stated-loading"
            class="text-[11px] text-muted"
          >
            Loading…
          </p>
          <p
            v-else-if="statedError"
            class="text-[11px] text-warning"
          >
            {{ statedError }}
          </p>
          <ul
            v-else
            class="space-y-1.5"
            data-testid="person-landing-stated-list"
          >
            <li
              v-for="row in statedRows"
              :key="row.id"
              data-testid="person-landing-stated-row"
              class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
            >
              <blockquote class="border-l-2 border-primary/40 pl-2 text-surface-foreground">
                {{ row.text || row.id }}
              </blockquote>
            </li>
          </ul>
        </section>

        <h3 class="mt-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Quotes across the corpus<span v-if="corpusEpisodeCount">
            · {{ corpusEpisodeCount }} episode{{ corpusEpisodeCount === 1 ? '' : 's' }}</span>
        </h3>
        <p
          v-if="corpusQuotes.length === 0"
          class="text-[11px] text-muted"
          data-testid="person-landing-positions-empty"
        >
          No attributed quotes for this voice yet.
        </p>
        <ul
          v-else
          class="space-y-1.5"
          data-testid="person-landing-positions"
        >
          <li
            v-for="row in corpusQuotes"
            :key="row.id"
            data-testid="person-landing-corpus-row"
            class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
          >
            <blockquote class="border-l-2 border-primary/40 pl-2 text-surface-foreground">
              {{ row.text || row.id }}
            </blockquote>
            <p
              v-if="row.episodeTitle"
              class="mt-0.5 text-[10px] text-muted"
            >
              {{ row.episodeTitle }}
            </p>
          </li>
        </ul>
      </template>

      <div v-if="!props.embedded" class="flex shrink-0 flex-wrap gap-2 pt-2">
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
    <!-- #1049 — Position Tracker per PRD-028 / RFC-072 §5A. Embedded, this is a
         peer node-view tab (NodeDetail owns it), so it renders only standalone. -->
    <div
      v-if="!props.embedded"
      v-show="activeTab === 'position_tracker'"
      id="person-landing-panel-position-tracker"
      role="tabpanel"
      aria-labelledby="person-landing-tab-position-tracker"
      data-testid="person-landing-panel-position-tracker"
      class="flex min-h-0 flex-1 flex-col"
    >
      <PositionTrackerPanel :person-id-override="personId" />
    </div>
  </div>
</template>
