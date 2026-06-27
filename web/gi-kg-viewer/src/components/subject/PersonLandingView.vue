<script setup lang="ts">
/**
 * #672 — Person Landing rail panel. Reads ``subject.personId`` and shows
 * a Profile / Positions tab pair: Profile holds basic identity info +
 * mentions timeline; Positions lists ``SPOKEN_BY`` quotes attributed to
 * this person with episode context.
 *
 * #909 — the Positions tab also surfaces an "Across the corpus" section: the
 * person's quotes across ALL episodes (from the CIL ``person_profile`` endpoint),
 * not just the episodes currently merged into the loaded graph. This is the
 * corpus-wide "what X said across episodes" payoff of the #875/#876 diarization work.
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
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import { StaleGeneration } from '../../utils/staleGeneration'
import {
  countPersonEntityIncidentEdges,
  findRawNodeInArtifact,
  findRawNodeInArtifactByIdOrPrefixed,
  normalizeGiEdgeType,
  personEpisodeAppearances,
  personInsightsByTopic,
  personRoleFromNode,
  rankedPersonOrganizations,
  rankedPersonTopicMentions,
} from '../../utils/parsing'
import { logicalEpisodeIdFromGraphNodeId } from '../../utils/graphEpisodeMetadata'
import { buildSubjectMentionsTimeline } from '../../utils/subjectMentionsTimeline'
import PositionTrackerPanel from './PositionTrackerPanel.vue'
import SubjectTimelineChart from './SubjectTimelineChart.vue'

/** Positions list cap — Persons accumulate more attributed quotes than a
 *  Topic accumulates mentions, so the cap is higher than ``TopicEntityView``'s
 *  25; still bounded so the rail does not become a full-text wall. */
const PERSON_LANDING_POSITIONS_CAP = 50

const emit = defineEmits<{
  goGraph: []
  closeSubject: []
  prefillSemanticSearch: [{ query: string }]
}>()

const artifacts = useArtifactsStore()
const shell = useShellStore()
const subject = useSubjectStore()

// #1048 — tab vocabulary aligned with PRD-028 / PRD-029. ``profile`` is the
// aggregate Person Profile view (PRD-029); ``position_tracker`` is the
// per-topic drill-in (PRD-028, filled by follow-up #1049).
type PersonTab = 'profile' | 'position_tracker'
const activeTab = ref<PersonTab>('profile')

// RFC-088 chunk 6c: enrichment-layer signals for the focused person.
interface GroundingRateRow {
  person_id: string
  person_name?: string
  total_insights: number
  grounded_insights: number
  rate: number
}
interface CoappearanceRow {
  person_a_id: string
  person_b_id: string
  person_a_name?: string
  person_b_name?: string
  episode_count: number
}
interface CoGuestChip {
  person_id: string
  person_name?: string
  episode_count: number
}
const groundingRow = ref<GroundingRateRow | null>(null)
const coGuestChips = ref<CoGuestChip[]>([])
const enrichmentLoaded = ref(false)
const COGUEST_TOP_N = 6

async function loadPersonEnrichmentSignals(focusedPersonId: string): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!focusedPersonId || !root) return
  groundingRow.value = null
  coGuestChips.value = []
  enrichmentLoaded.value = false
  try {
    const [grounding, coapp] = await Promise.all([
      fetchCachedCorpusEnvelope<{ persons: GroundingRateRow[] }>(root, 'grounding_rate').catch(
        () => null,
      ),
      fetchCachedCorpusEnvelope<{ pairs: CoappearanceRow[] }>(
        root,
        'guest_coappearance',
      ).catch(() => null),
    ])
    enrichmentLoaded.value = true
    if (grounding?.data?.persons) {
      groundingRow.value =
        grounding.data.persons.find((p) => p.person_id === focusedPersonId) ?? null
    }
    if (coapp?.data?.pairs) {
      const chips: CoGuestChip[] = []
      for (const p of coapp.data.pairs) {
        if (p.person_a_id === focusedPersonId) {
          chips.push({
            person_id: p.person_b_id,
            person_name: p.person_b_name,
            episode_count: p.episode_count,
          })
        } else if (p.person_b_id === focusedPersonId) {
          chips.push({
            person_id: p.person_a_id,
            person_name: p.person_a_name,
            episode_count: p.episode_count,
          })
        }
      }
      chips.sort((a, b) => b.episode_count - a.episode_count)
      coGuestChips.value = chips.slice(0, COGUEST_TOP_N)
    }
  } catch {
    /* enrichment is best-effort; never break the rail */
  }
}

watch(
  () => subject.personId,
  (id) => {
    if (id) void loadPersonEnrichmentSignals(id)
    else {
      groundingRow.value = null
      coGuestChips.value = []
      enrichmentLoaded.value = false
    }
  },
  { immediate: true },
)

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
    const [t, c] = await Promise.all([fetchPersonTopics(root, id), fetchCoSpeakers(root, id)])
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

/**
 * #909 — corpus-wide quotes this person spoke across ALL episodes, from the CIL
 * ``person_profile`` endpoint (``GET /api/persons/{id}/brief``). Unlike ``positionRows``
 * below (which only sees the loaded/merged graph), this resolves the person across the
 * whole corpus (incl. #852 name variants). Async, skeleton-first, stale-gated; renders
 * only when an API corpus is connected (no-op in local-file mode).
 */
interface CorpusQuoteRow {
  id: string
  text: string
  episodeId: string | null
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
      rows.push({ id: qid, text, episodeId })
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

const edgeCounts = computed(() =>
  countPersonEntityIncidentEdges(artifacts.displayArtifact, personGraphNodeId.value),
)

const timeline = computed(() =>
  buildSubjectMentionsTimeline(artifacts.displayArtifact, personGraphNodeId.value),
)

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

interface PositionRow {
  id: string
  text: string
  episodeId: string | null
  episodeTitle: string | null
  publishDate: string | null
}

const positionRows = computed<PositionRow[]>(() => {
  const art = artifacts.displayArtifact
  const pid = personGraphNodeId.value
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

// #1049 — clicking a ranked-Topic row pivots the Position Tracker tab
// onto that (Person, Topic) pair and switches tabs. The same handler is
// the entry point #1050 will eventually use from Topic group headers.
function onPickTopicForPositionTracker(topicId: string): void {
  if (!topicId.trim()) return
  subject.selectTopicForPositionTracker(topicId)
  activeTab.value = 'position_tracker'
}
</script>

<template>
  <div
    class="mx-3 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
    role="region"
    aria-label="Person"
    data-testid="person-landing-view"
  >
    <div class="mt-1 shrink-0 border-b border-border pb-2">
      <div class="flex items-baseline gap-2">
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
        v-if="edgeCounts.spokenByQuotes > 0 || edgeCounts.spokeInEpisodes > 0"
        class="text-[10px] text-muted"
        data-testid="person-landing-edge-counts"
      >
        In this graph: {{ edgeCounts.spokenByQuotes }}
        attributed quote{{ edgeCounts.spokenByQuotes === 1 ? '' : 's' }} ·
        {{ edgeCounts.spokeInEpisodes }}
        episode link{{ edgeCounts.spokeInEpisodes === 1 ? '' : 's' }}.
      </p>
      <p
        v-else
        class="text-[10px] text-muted"
        data-testid="person-landing-edge-counts-empty"
      >
        No graph links for this person yet — load the corpus graph to populate.
      </p>
      <!-- RFC-088 chunk 6c: enrichment signals (grounding rate + co-guest chips). -->
      <section
        v-if="enrichmentLoaded && (groundingRow || coGuestChips.length)"
        class="w-full min-w-0 rounded border border-default bg-overlay/40 p-2"
        aria-label="Enrichment signals"
        data-testid="person-landing-enrichment-signals"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Enrichment signals
        </h3>
        <div
          v-if="groundingRow"
          class="mb-1 flex items-center gap-2 text-[10px]"
          data-testid="person-landing-grounding-rate"
        >
          <span class="text-muted">Grounded:</span>
          <span
            class="rounded px-2 py-0.5 font-mono"
            :class="groundingRow.rate >= 0.8 ? 'bg-emerald-700/30 text-emerald-300' : groundingRow.rate >= 0.5 ? 'bg-overlay text-muted' : 'bg-amber-700/30 text-amber-300'"
          >{{ (groundingRow.rate * 100).toFixed(0) }}%</span>
          <span class="text-muted">
            · {{ groundingRow.grounded_insights }} / {{ groundingRow.total_insights }} insights
          </span>
        </div>
        <div v-if="coGuestChips.length" data-testid="person-landing-coguests">
          <p class="mb-1 text-[10px] text-muted">Often appears with</p>
          <div class="flex flex-wrap gap-1">
            <button
              v-for="g in coGuestChips"
              :key="g.person_id"
              type="button"
              class="rounded border border-default bg-overlay px-2 py-0.5 text-[10px] hover:bg-overlay-2"
              :data-testid="`person-landing-coguest-${g.person_id}`"
              @click="subject.focusPerson(g.person_id)"
            >
              {{ g.person_name || g.person_id }}
              <span class="ml-1 text-muted">·{{ g.episode_count }}</span>
            </button>
          </div>
        </div>
      </section>
      <section aria-label="Mentions by month">
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Mentions by month
        </h3>
        <SubjectTimelineChart
          :timeline="timeline"
          aria-label="Mentions by month for this person"
        />
      </section>
      <!-- #1050 — UXS-010 "Topics discussed" — ranked by ABOUT∩MENTIONS_PERSON
           insight count. Each row is a button that pivots the Position
           Tracker tab to (this Person, that Topic) — #1049 entry point. -->
      <section
        v-if="rankedTopics.length"
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
            v-for="t in topicsRows"
            :key="t.id"
            class="rounded bg-overlay px-1.5 py-0.5 text-[10px] text-surface-foreground"
            data-testid="person-landing-topic-chip"
          >{{ t.text }}</span>
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
          <span
            v-for="p in coSpeakersRows"
            :key="p.id"
            class="rounded bg-overlay px-1.5 py-0.5 text-[10px] text-surface-foreground"
            data-testid="person-landing-co-speaker-chip"
          >{{ p.text }}</span>
        </div>
        <p v-else class="text-[10px] text-muted" data-testid="person-landing-co-speakers-empty">
          No co-speakers share a topic with this voice yet.
        </p>
      </section>
      <!-- #909 / #1048 — corpus-wide quotes this person spoke across ALL episodes (CIL person_profile). -->
      <section
        v-if="corpusLoading || corpusError || corpusQuotes.length"
        aria-label="Across the corpus"
        data-testid="person-landing-corpus"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Across the corpus<span v-if="corpusEpisodeCount">
            · {{ corpusEpisodeCount }} episode{{ corpusEpisodeCount === 1 ? '' : 's' }}</span>
        </h3>
        <p
          v-if="corpusLoading"
          data-testid="person-landing-corpus-loading"
          class="text-[11px] text-muted"
        >
          Loading…
        </p>
        <p
          v-else-if="corpusError"
          class="text-[11px] text-warning"
        >
          {{ corpusError }}
        </p>
        <ul
          v-else
          class="space-y-1.5"
          data-testid="person-landing-corpus-list"
        >
          <li
            v-for="row in corpusQuotes.slice(0, PERSON_LANDING_POSITIONS_CAP)"
            :key="row.id"
            data-testid="person-landing-corpus-row"
            class="rounded border border-border bg-elevated/40 px-2 py-1.5 text-[11px] leading-snug"
          >
            <blockquote class="border-l-2 border-primary/40 pl-2 text-surface-foreground">
              {{ row.text || row.id }}
            </blockquote>
            <p
              v-if="row.episodeId"
              class="mt-0.5 text-[10px] text-muted"
            >
              {{ row.episodeId }}
            </p>
          </li>
        </ul>
        <p
          v-if="corpusQuotes.length > PERSON_LANDING_POSITIONS_CAP"
          class="text-[10px] text-muted"
          data-testid="person-landing-corpus-overflow"
        >
          + {{ corpusQuotes.length - PERSON_LANDING_POSITIONS_CAP }} more
        </p>
      </section>

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

      <h3
        v-if="statedRows.length || statedLoading"
        class="mt-2 text-[10px] font-semibold uppercase tracking-wider text-muted"
      >
        Attributed quotes
      </h3>
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
          v-for="row in positionRows.slice(0, PERSON_LANDING_POSITIONS_CAP)"
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
        v-if="positionRows.length > PERSON_LANDING_POSITIONS_CAP"
        class="text-[10px] text-muted"
        data-testid="person-landing-positions-overflow"
      >
        + {{ positionRows.length - PERSON_LANDING_POSITIONS_CAP }} more
      </p>
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
    <!-- #1049 — Position Tracker per PRD-028 / RFC-072 §5A. -->
    <div
      v-show="activeTab === 'position_tracker'"
      id="person-landing-panel-position-tracker"
      role="tabpanel"
      aria-labelledby="person-landing-tab-position-tracker"
      data-testid="person-landing-panel-position-tracker"
      class="flex min-h-0 flex-1 flex-col"
    >
      <PositionTrackerPanel />
    </div>
  </div>
</template>
