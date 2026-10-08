<script setup lang="ts">
import { computed, ref, watch } from "vue"
import type { BridgeDocument } from "../../types/bridge"
import type { ParsedArtifact } from "../../types/artifact"
import { useGraphFilterStore } from "../../stores/graphFilters"
import { useGraphHandoffStore } from "../../stores/graphHandoff"
import { useGraphNavigationStore } from "../../stores/graphNavigation"
import { useGraphAnalyticsStore } from "../../stores/graphAnalytics"
import { useShellStore } from "../../stores/shell"
import { useSubjectStore } from "../../stores/subject"
import { graphNodeTypeChrome } from "../../utils/colors"
import { formatCalendarDateForDisplay, truncate } from "../../utils/formatting"
import { quoteAttributionDisplayFromId } from "../../utils/parsing"
import {
  fetchTopicTimeline,
  type CilArcEpisodeBlock,
  type CilTopicTimelineMergedResponse,
  type CilTopicTimelineResponse,
} from "../../api/cilApi"
import {
  findRawNodeInArtifact,
  findRawNodeInArtifactByIdOrPrefixed,
  fullPrimaryNodeLabel,
  insightProvenanceLine,
  insightRelatedTopicRows,
  insightSupportingQuoteRows,
  insightSupportingTranscriptAggregate,
  primaryTextFromLooseGiNode,
} from "../../utils/parsing"
import { formatInsightPositionHintLine } from "../../utils/insightPositionHint"
import { copyTextToClipboard } from "../../utils/clipboard"
import { graphTypeAvatarLetter } from "../../utils/graphTypeAvatar"
import { stripLayerPrefixesForCil } from "../../utils/mergeGiKg"
import HelpTip from "../shared/HelpTip.vue"
import {
  SEARCH_RESULT_COPY_TITLE_CHIP_CLASS,
  SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS,
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
  SEARCH_RESULT_GRAPH_BUTTON_CLASS,
} from "../../utils/searchResultActionStyles"
import { bridgeIdentityForGraphNodeId, crossLayerPresenceLabel } from "../../utils/bridgeDocument"
import { visualGroupForNode } from "../../utils/visualGroup"
import {
  corpusTextFileViewUrl,
  formatAudioTimingRange,
  formatTranscriptCharRange,
  GI_QUOTE_SPEAKER_UNAVAILABLE_HINT,
  resolveGiPathForTranscript,
  resolveTranscriptCorpusRelpath,
} from "../../utils/transcriptSourceDisplay"
import GraphConnectionsSection from "./GraphConnectionsSection.vue"
import NodeEnrichmentSection from "./NodeEnrichmentSection.vue"
import TopicEntityView from "../subject/TopicEntityView.vue"
import PersonLandingView from "../subject/PersonLandingView.vue"
import PodcastNodeView from "../subject/PodcastNodeView.vue"
import InsightNodeView from "../subject/InsightNodeView.vue"
import SubjectTimelineChart from "../subject/SubjectTimelineChart.vue"
import NodeTopicPerspectives from "./NodeTopicPerspectives.vue"
import {
  buildSubjectMentionsTimeline,
  type SubjectMentionsTimeline,
} from "../../utils/subjectMentionsTimeline"
import TranscriptViewerDialog from "../shared/TranscriptViewerDialog.vue"
import PodcastCover from "../shared/PodcastCover.vue"

const emit = defineEmits<{
  close: []
  "go-graph": []
  /** Off the Graph tab: open the Graph with this node focused (UXS-007 "View in graph"). */
  "view-in-graph": [string]
  "prefill-semantic-search": [{ query: string }]
  "open-search-topic-filter": [{ topic: string }]
  "open-search-speaker-filter": [{ speaker: string }]
  "open-search-insight-filters": [{ groundedOnly: boolean; minConfidence: number | null }]
  "open-library-episode": [{ metadata_relative_path: string }]
}>()

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  nodeId: string | null
  /** Embedded in App right rail: full width, no fixed 280px strip. */
  embedInRail?: boolean
  /** bridge.json (optional) for cross-layer diagnostics. */
  bridgeDocument?: BridgeDocument | null
  /** Offer "View in graph" on a topic — only meaningful when the Graph tab is not showing. */
  showViewInGraph?: boolean
}>()

const shell = useShellStore()
const graphNav = useGraphNavigationStore()
const graphAnalytics = useGraphAnalyticsStore()
const graphFilters = useGraphFilterStore()
const graphHandoff = useGraphHandoffStore()
const subject = useSubjectStore()

/** Merged GI/KG before per-type visibility filters (quotes/speakers/episodes off by default on canvas). */
const fullMergedArtifactForMetadata = computed(
  () => graphFilters.fullArtifact ?? props.viewArtifact
)

const transcriptViewerRef = ref<InstanceType<typeof TranscriptViewerDialog> | null>(null)

/** Shown in **Where this appears**; omit from the generic property list. */
const TRANSCRIPT_ANCHOR_PROP_KEYS = new Set([
  "transcript_ref",
  "char_start",
  "char_end",
  "timestamp_start_ms",
  "timestamp_end_ms",
  "speaker_id",
])

const HIDDEN_PROPS = new Set([
  "slug",
  "label",
  "name",
  "text",
  "title",
  "description",
  "entity_kind",
  "kind",
  // Person role is surfaced as the Host/Guest/Mentioned badge in the person
  // view; the raw "Role: mentioned" property row is redundant.
  "role",
])

/** Shown in insight meta strip; omit from generic property list. */
const INSIGHT_META_DL_KEYS = new Set(["grounded", "insight_type", "position_hint"])

const INSIGHT_SEMANTIC_SEARCH_MAX_CHARS = 240

const PERSON_ENTITY_SEMANTIC_SEARCH_MAX_CHARS = 240

const INSIGHT_SUPPORTING_QUOTES_COLLAPSE_AFTER = 5

/** Text-style HelpTip trigger under Full insight (not the round diagnostics chip). */
const INSIGHT_GROUNDING_TIP_TRIGGER_CLASS =
  "inline-flex max-w-full min-h-[1.25rem] items-center rounded px-0.5 py-0 text-left " +
  "text-[10px] font-semibold leading-snug text-muted underline decoration-muted/40 " +
  "underline-offset-2 hover:bg-overlay/50 hover:text-surface-foreground"

const graphFocusNeighborTooltip =
  "Show on graph — focus this node in the loaded merged graph (same as semantic search G)."

const node = computed(() => {
  const id = props.nodeId
  if (id == null) {
    return null
  }
  const slice = props.viewArtifact
  if (slice) {
    // Prefix-tolerant: a `quote`/`person` id from search arrives bare (``quote:…``)
    // but the merged artifact stores it GI/KG-prefixed (``g:quote:…``). Exact match
    // would miss it → empty "Node" rail on real (merged) corpora. (#967↔#974)
    const hit = findRawNodeInArtifactByIdOrPrefixed(slice, id)
    if (hit) {
      return hit
    }
  }
  const full = graphFilters.fullArtifact
  return full ? findRawNodeInArtifactByIdOrPrefixed(full, id) : null
})

// When the focused node isn't in the current graph slice — e.g. a co-speaker
// or related entity sourced from the full server-side relational graph — the
// artifact lookup above returns null. Infer the kind from the id so the rail
// still renders the person / entity / topic view (all of which load from server
// endpoints, independent of the viewer's graph slice) instead of an empty
// "Node" shell.
const inferredKindFromId = computed(
  (): "Person" | "Organization" | "Object" | "Topic" | "Podcast" | "Insight" | null => {
    if (node.value) return null
    const bare = stripLayerPrefixesForCil(props.nodeId ?? "")
    if (bare.startsWith("person:")) return "Person"
    if (bare.startsWith("org:")) return "Organization"
    // #2057: without this an `object:` id fell through to null and rendered the empty
    // "Node" shell instead of the entity view.
    if (bare.startsWith("object:")) return "Object"
    if (bare.startsWith("topic:")) return "Topic"
    if (bare.startsWith("podcast:")) return "Podcast"
    // An out-of-slice insight (e.g. a corpus-wide timeline-mention drill) renders
    // from the /relational/insight-detail endpoint via InsightNodeView.
    if (bare.startsWith("insight:")) return "Insight"
    return null
  }
)

const nodeType = computed(() => {
  const n = node.value
  if (!n) return inferredKindFromId.value ?? "?"
  return String(n.type ?? "?")
})

const isQuoteNode = computed(() => nodeType.value.trim().toLowerCase() === "quote")

const isTopicNode = computed(() => nodeType.value.trim().toLowerCase() === "topic")

const isInsightNode = computed(() => nodeType.value.trim().toLowerCase() === "insight")

/** GI ``Person`` / legacy ``Speaker`` / KG ``Entity`` (person, organization or object). */
const isPersonEntityRailNode = computed(() => {
  // RFC-097 v3.0: Organization is a first-class node type and shares the
  // Person/Entity rail UI — the rail handles both speaker-like (Person /
  // Speaker) and brand-like (Entity / Organization) nodes. The button copy
  // adapts via the ``['person', 'speaker'].includes(...)`` check at the
  // render site (Person profile vs Entity profile).
  //
  // KG schema 2.1 (#2057): Object is the third first-class entity kind, and it was missing here —
  // so selecting "Fahrenheit 451" or "Project Panama" in the graph opened nothing at all, while
  // the node beside it opened a full rail. It is brand-like rather than speaker-like (it has no
  // talk time and no co-speakers), so it takes the same branch as Organization; the sections that
  // are speaker-only are already gated on ``isPersonNode`` below and stay hidden.
  const t = nodeType.value.trim().toLowerCase()
  return (
    t === "person" ||
    t === "entity" ||
    t === "speaker" ||
    t === "organization" ||
    t === "object"
  )
})

/** Person / Speaker — keeps its own PersonLandingView profile (not yet folded). */
const isPersonNode = computed(() => {
  const t = nodeType.value.trim().toLowerCase()
  return t === "person" || t === "speaker"
})

/** Podcast / Show node — folds in the PodcastNodeView (basics + episode list). */
const isPodcastNode = computed(() => nodeType.value.trim().toLowerCase() === "podcast")

// Show cover art emitted up by PodcastNodeView so the rail header avatar shows
// the real cover for a podcast node instead of the generic "P" letter (P2).
const podcastCover = ref<{ imageUrl: string | null; imageLocalRelpath: string | null } | null>(null)

// Out-of-slice insight: InsightNodeView resolves the text from the server and
// emits it up so the header shows the claim (not the opaque insight: hash).
const insightHeaderText = ref("")

/**
 * Full quote/insight text in the rail — not ``nodeLabel`` (that caps at ~40 chars for on-canvas labels).
 */
const displayName = computed(() => {
  const n = node.value
  if (!n) {
    // Out-of-slice node (e.g. a co-speaker from the full relational graph):
    // derive a readable name from the id slug so the header isn't blank while
    // the person / topic view loads from server endpoints.
    const bare = stripLayerPrefixesForCil(props.nodeId ?? "")
    // An insight's "name" is its text (an opaque hash id makes a useless title);
    // InsightNodeView emits the resolved text up. Show a placeholder until then.
    if (bare.startsWith("insight:")) return insightHeaderText.value.trim() || "Insight"
    const slug = bare.replace(/^[a-z]+:/, "").trim()
    if (slug) {
      return slug.replace(/[-_]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
    }
    return ""
  }
  return fullPrimaryNodeLabel(n)
})

/** Full passage for copy/paste; header may still clamp visually. */
const quoteFullPassage = computed((): string | null => {
  if (!isQuoteNode.value) return null
  const full = displayName.value.trim()
  return full || null
})

/** Long primary label (Topic uses ``properties.label``; same as header source, uncapped). */
const fullRailTextContent = computed((): string | null => {
  if (isQuoteNode.value) return quoteFullPassage.value
  if (isTopicNode.value || isInsightNode.value || isPersonEntityRailNode.value) {
    const s = displayName.value.trim()
    return s || null
  }
  return null
})

/** Native tooltip on clamped header: omit when full text is shown in the scroll body below. */
const primaryTitleNativeTooltip = computed((): string | undefined => {
  if (
    isQuoteNode.value ||
    isTopicNode.value ||
    isInsightNode.value ||
    isPersonEntityRailNode.value
  ) {
    return undefined
  }
  const s = displayName.value.trim()
  return s.length > 0 ? s : undefined
})

/** Optional GI topic aliases (comma-separated). */
const topicAliasesLine = computed((): string | null => {
  if (!isTopicNode.value) return null
  const a = node.value?.properties?.aliases
  if (!Array.isArray(a) || a.length === 0) return null
  const parts = a
    .filter((x): x is string => typeof x === "string" && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(", ") : null
})

/** Optional Person / Entity aliases (UXS-004 line 176). Same shape as
 *  topicAliasesLine — reads `properties.aliases`, comma-joins. Null when
 *  absent so the row is `v-if`-hidden. */
const personEntityAliasesLine = computed((): string | null => {
  if (!isPersonEntityRailNode.value) return null
  const a = node.value?.properties?.aliases
  if (!Array.isArray(a) || a.length === 0) return null
  const parts = a
    .filter((x): x is string => typeof x === "string" && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(", ") : null
})

/** Optional Person / Entity role label (UXS-004 line 176 —
 *  Host/Guest/Mentioned). Reads `properties.role`; the SPOKEN_BY /
 *  SPOKE_IN edge-count breakdown UXS-004 mentions is deferred until the
 *  graph-analytics store exposes per-node edge counts. */
const personEntityRoleLabel = computed((): string | null => {
  if (!isPersonEntityRailNode.value) return null
  const r = node.value?.properties?.role
  if (typeof r !== "string" || !r.trim()) return null
  const v = r.trim().toLowerCase()
  if (v === "host") return "Host"
  if (v === "guest") return "Guest"
  if (v === "mentioned") return "Mentioned"
  // Preserve any future role value verbatim rather than dropping it.
  return r.trim()
})

/** Topic label for Search / Explore handoff (same primary string as full block). */
const topicGatewayQuery = computed((): string => {
  if (!isTopicNode.value) return ""
  return displayName.value.trim()
})

function emitTopicPrefillSearch(): void {
  const q = topicGatewayQuery.value
  if (!q) return
  emit("prefill-semantic-search", { query: q })
}

function emitTopicExploreFilter(): void {
  const t = topicGatewayQuery.value
  if (!t) return
  emit("open-search-topic-filter", { topic: t })
}

const personEntityGatewayQuery = computed((): string => {
  if (!isPersonEntityRailNode.value) return ""
  return displayName.value.trim()
})

const personEntitySemanticSearchQuery = computed((): string => {
  if (!isPersonEntityRailNode.value) return ""
  return truncate(displayName.value.trim(), PERSON_ENTITY_SEMANTIC_SEARCH_MAX_CHARS)
})

function emitPersonEntityPrefillSearch(): void {
  const q = personEntitySemanticSearchQuery.value.trim()
  if (!q) return
  emit("prefill-semantic-search", { query: q })
}

function emitPersonEntityExploreHandoff(): void {
  const q = personEntityGatewayQuery.value.trim()
  if (!q) return
  if (visualType.value !== "Entity_organization") {
    emit("open-search-speaker-filter", { speaker: q })
  } else {
    emit("open-search-topic-filter", { topic: q })
  }
}

const insightEpisodeDurationMs = computed((): number | null => {
  if (!isInsightNode.value || !node.value) {
    return null
  }
  const p = node.value.properties as Record<string, unknown> | undefined
  const direct = p?.episode_duration_ms ?? p?.duration_ms
  if (typeof direct === "number" && Number.isFinite(direct) && direct > 0) {
    return direct
  }
  const rawEp = p?.episode_id
  const epId =
    typeof rawEp === "string" && rawEp.trim()
      ? rawEp.trim()
      : typeof rawEp === "number" && Number.isFinite(rawEp)
        ? String(rawEp)
        : null
  if (!epId) {
    return null
  }
  const art = fullMergedArtifactForMetadata.value
  if (!art) {
    return null
  }
  const ep = findRawNodeInArtifact(art, epId)
  const qp = ep?.properties as Record<string, unknown> | undefined
  if (!qp) {
    return null
  }
  const d = qp.episode_duration_ms ?? qp.duration_ms
  return typeof d === "number" && Number.isFinite(d) && d > 0 ? d : null
})

/** Type / position / confidence only (grounding + provenance live in the details HelpTip). */
const insightSecondaryDetailLines = computed((): string[] => {
  if (!isInsightNode.value) return []
  const parts: string[] = []
  const p = node.value?.properties
  const it = p?.insight_type
  if (typeof it === "string" && it.trim()) {
    parts.push(`Type: ${it.trim()}`)
  }
  const ph = p?.position_hint
  if (typeof ph === "number" && Number.isFinite(ph)) {
    parts.push(formatInsightPositionHintLine(ph, insightEpisodeDurationMs.value))
  }
  const rawNode = node.value as Record<string, unknown> | null
  const cn = rawNode?.confidence
  if (typeof cn === "number" && Number.isFinite(cn)) {
    parts.push(`Confidence: ${cn.toFixed(2)}`)
  }
  return parts
})

const insightGroundedBoolean = computed((): boolean | null => {
  if (!isInsightNode.value) return null
  const g = node.value?.properties?.grounded
  return typeof g === "boolean" ? g : null
})

const insightSemanticSearchQuery = computed((): string => {
  if (!isInsightNode.value) return ""
  return truncate(displayName.value.trim(), INSIGHT_SEMANTIC_SEARCH_MAX_CHARS)
})

const insightSupportingQuotes = computed(() =>
  isInsightNode.value
    ? insightSupportingQuoteRows(fullMergedArtifactForMetadata.value, props.nodeId)
    : []
)

const insightSupportingTranscriptAgg = computed(() =>
  isInsightNode.value
    ? insightSupportingTranscriptAggregate(fullMergedArtifactForMetadata.value, props.nodeId)
    : null
)

const insightOpenAllSupportingQuotesReady = computed((): boolean => {
  const agg = insightSupportingTranscriptAgg.value
  if (!agg) {
    return false
  }
  const root = shell.corpusPath.trim()
  if (!shell.healthStatus || !shell.hasCorpusPath) {
    return false
  }
  const giPath = resolveGiPathForTranscript(fullMergedArtifactForMetadata.value, agg.episodeId)
  const resolvedRelpath = resolveTranscriptCorpusRelpath(agg.transcriptRef, giPath)
  return Boolean(resolvedRelpath && corpusTextFileViewUrl(root, resolvedRelpath))
})

const insightQuotesExpanded = ref(false)

watch(
  () => props.nodeId,
  () => {
    insightQuotesExpanded.value = false
  }
)

const insightSupportingQuotesVisible = computed(() => {
  const all = insightSupportingQuotes.value
  if (insightQuotesExpanded.value || all.length <= INSIGHT_SUPPORTING_QUOTES_COLLAPSE_AFTER) {
    return all
  }
  return all.slice(0, INSIGHT_SUPPORTING_QUOTES_COLLAPSE_AFTER)
})

function toggleInsightSupportingQuotesExpanded(): void {
  insightQuotesExpanded.value = !insightQuotesExpanded.value
}

const insightRelatedTopics = computed(() =>
  isInsightNode.value && props.nodeId
    ? insightRelatedTopicRows(props.viewArtifact, props.nodeId)
    : []
)

const insightProvenanceText = computed(() =>
  isInsightNode.value ? insightProvenanceLine(props.viewArtifact) : null
)

const hasInsightDetailsTipContent = computed((): boolean => {
  if (!isInsightNode.value) return false
  if (insightGroundedBoolean.value !== null) return true
  if (insightSecondaryDetailLines.value.length > 0) return true
  return Boolean(insightProvenanceText.value?.trim())
})

/** Visible trigger and accessible name (popover holds the full grounding explainer). */
const insightGroundingTriggerLabel = computed((): string => {
  if (insightGroundedBoolean.value === true) return "Grounded"
  if (insightGroundedBoolean.value === false) return "Not grounded"
  return "Extraction details"
})

const insightExploreFiltersPayload = computed(
  (): {
    groundedOnly: boolean
    minConfidence: number | null
  } | null => {
    if (!isInsightNode.value) return null
    const p = node.value?.properties
    const g = p?.grounded
    const groundedOnly = typeof g === "boolean" ? g : false
    const rawNode = node.value as Record<string, unknown> | null
    const c = rawNode?.confidence
    const minConfidence = typeof c === "number" && Number.isFinite(c) ? c : null
    return { groundedOnly, minConfidence }
  }
)

function emitInsightPrefillSearch(): void {
  const q = insightSemanticSearchQuery.value.trim()
  if (!q) return
  emit("prefill-semantic-search", { query: q })
}

function emitInsightOpenSearchFilters(): void {
  const p = insightExploreFiltersPayload.value
  if (!p) return
  emit("open-search-insight-filters", p)
}

function focusNeighborOnGraph(nbId: string, ev: MouseEvent): void {
  ev.stopPropagation()
  const id = nbId.trim()
  if (!id) return
  // F1.6 — NodeDetail "neighbour go-graph" is a graph-internal expansion (preserves layout) targeted at
  // a specific neighbour node.
  graphHandoff.expansionRequested({
    kind: "graph-node",
    cyId: id,
    source: "node-detail",
    loadSource: "graph-internal",
    camera: { kind: "center-on-target" },
  })
  graphNav.requestFocusNode(id)
  emit("go-graph")
}

/** Gate for the CIL timeline API. */
const cilTimelineApiUnavailable = computed(
  (): boolean =>
    !shell.healthStatus || shell.cilQueriesApiAvailable === false || !shell.hasCorpusPath
)

type InlineTimelineMode = "single" | null

const inlineTimelineMode = ref<InlineTimelineMode>(null)
const inlineTimelineTopicId = ref("")
const inlineTimelineLoading = ref(false)
const inlineTimelineError = ref<string | null>(null)
const inlineTimelinePayload = ref<CilTopicTimelineResponse | CilTopicTimelineMergedResponse | null>(
  null
)
const inlineTimelineSortOrder = ref<"asc" | "desc">("desc")

const showInlineTopicTimeline = computed(
  () => isTopicNode.value && Boolean(props.nodeId?.trim())
)

// Corpus-wide timeline data, de-duplicated to ONE row per episode. The API
// yields one row per bundle, so an episode processed in multiple runs shows up
// more than once; keep the richest bundle (most insights) per episode_id — the
// same "unique episode" contract the corpus catalog applies.
const dedupedEpisodes = computed((): CilArcEpisodeBlock[] => {
  const eps = inlineTimelinePayload.value?.episodes ?? []
  const byId = new Map<string, CilArcEpisodeBlock>()
  for (const ep of eps) {
    const id = String(ep.episode_id ?? "")
    const prev = byId.get(id)
    if (!prev || (ep.insights?.length ?? 0) > (prev.insights?.length ?? 0)) {
      byId.set(id, ep)
    }
  }
  return [...byId.values()]
})

// MENTIONS: the deduped episodes flattened to one row per insight — corpus-wide,
// or scoped to a single episode when the user drills in from the Episodes list.
const mentionsSortOrder = ref<"asc" | "desc">("desc")
const mentionsEpisodeFilter = ref<string | null>(null)
interface FlatMentionRow {
  key: string
  text: string
  episodeTitle: string
  publishDate: string | null
  /** Insight node id, so a mention can drill into the insight's node view. */
  insightId: string | null
}
const flatMentions = computed<FlatMentionRow[]>(() => {
  const filter = mentionsEpisodeFilter.value
  const rows: FlatMentionRow[] = []
  dedupedEpisodes.value.forEach((ep, ei) => {
    if (filter && String(ep.episode_id ?? "") !== filter) return
    const title = episodePrimaryHeading(ep)
    const date = ep.publish_date
    const insights = ep.insights ?? []
    for (let ii = 0; ii < insights.length; ii++) {
      const ins = insights[ii]
      const insId =
        ins && typeof ins === "object" && (ins as Record<string, unknown>).id != null
          ? String((ins as Record<string, unknown>).id)
          : null
      rows.push({
        key: `${ep.episode_id ?? ei}-${ii}`,
        text: insightLine(ins),
        episodeTitle: title,
        publishDate: date,
        insightId: insId,
      })
    }
  })
  rows.sort((a, b) => {
    const av = a.publishDate ?? ""
    const bv = b.publishDate ?? ""
    if (av === bv) return 0
    const cmp = av < bv ? -1 : 1
    return mentionsSortOrder.value === "asc" ? cmp : -cmp
  })
  return rows
})
const filteredMentionEpisode = computed((): CilArcEpisodeBlock | null => {
  const f = mentionsEpisodeFilter.value
  if (!f) return null
  return dedupedEpisodes.value.find((e) => String(e.episode_id ?? "") === f) ?? null
})

function insightLine(ins: Record<string, unknown>): string {
  return primaryTextFromLooseGiNode(ins).trim() || "(no text)"
}

function formatEpisodeDate(raw: string | null | undefined): string {
  if (!raw?.trim()) return ""
  return formatCalendarDateForDisplay(raw)
}

function episodePrimaryHeading(ep: CilArcEpisodeBlock): string {
  const t = ep.episode_title?.trim()
  if (t) return t
  const n = ep.episode_number
  if (n != null && Number.isFinite(Number(n))) return `Episode ${n}`
  const f = ep.feed_title?.trim()
  return f || "Unnamed episode"
}

function episodeContextLine(ep: CilArcEpisodeBlock): string | null {
  const t = ep.episode_title?.trim()
  const f = ep.feed_title?.trim()
  const n = ep.episode_number
  const hasNum = n != null && Number.isFinite(Number(n))
  if (t && f) return `Podcast: ${f}`
  if (t && !f && hasNum) return `Episode ${n}`
  if (!t && f && hasNum) return `Podcast: ${f}`
  if (!t && f && !hasNum) return "No episode title in corpus metadata"
  if (!t && !f && !hasNum) return "Corpus episode"
  return null
}

const inlineTimelineEpisodeCount = computed(() => dedupedEpisodes.value.length)

// N8 — a compact dot time-series above the Episodes/Mentions toggle. Buckets the
// same CIL timeline data by YYYY-MM, following the active view: Episodes counts
// distinct episodes per month, Mentions counts atomic mentions per month. Shaped
// as a SubjectMentionsTimeline so it feeds the shared chart (dots variant).
function timelineMonthOf(raw: string | null | undefined): string | null {
  const m = (raw ?? "").trim().match(/^(\d{4})-(\d{2})/)
  return m ? `${m[1]}-${m[2]}` : null
}
const timelineChartData = computed<SubjectMentionsTimeline>(() => {
  const counts = new Map<string, number>()
  let undated = 0
  if (timelineView.value === "mentions") {
    for (const r of flatMentions.value) {
      const ym = timelineMonthOf(r.publishDate)
      if (!ym) {
        undated += 1
        continue
      }
      counts.set(ym, (counts.get(ym) ?? 0) + 1)
    }
  } else {
    for (const ep of dedupedEpisodes.value) {
      const ym = timelineMonthOf(ep.publish_date)
      if (!ym) {
        undated += 1
        continue
      }
      counts.set(ym, (counts.get(ym) ?? 0) + 1)
    }
  }
  const months = [...counts.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([ymd, count]) => ({ ymd, count }))
  return {
    months,
    total: months.reduce((s, m) => s + m.count, 0),
    undated,
    episodeCount: dedupedEpisodes.value.length,
    insightIds: [],
    quoteIds: [],
  }
})

const inlineTimelineTopicIdsLabel = computed((): string => inlineTimelineTopicId.value)

const corpusPathForCovers = computed(() =>
  (shell.resolvedCorpusPath ?? shell.corpusPath ?? "").trim()
)

const inlineTimelineSortedEpisodes = computed((): CilArcEpisodeBlock[] => {
  const arr = [...dedupedEpisodes.value]
  arr.sort((a, b) => {
    const da = (a.publish_date ?? "").trim()
    const db = (b.publish_date ?? "").trim()
    const cmp = da.localeCompare(db) || String(a.episode_id).localeCompare(String(b.episode_id))
    return inlineTimelineSortOrder.value === "asc" ? cmp : -cmp
  })
  return arr
})

// Episodes render compact (title + our summary_title); clicking a row drills
// into just that episode's mentions — switches the Timeline view to Mentions,
// filtered to it — the way to read one episode's insights + summary.
function episodeKey(ep: CilArcEpisodeBlock): string {
  return String(ep.episode_id ?? "")
}
function openEpisodeMentions(ep: CilArcEpisodeBlock): void {
  mentionsEpisodeFilter.value = String(ep.episode_id ?? "")
  mentionsPage.value = 1
  timelineView.value = "mentions"
}

// FB15 — drill from a timeline mention into the insight's own node view (with
// Back). Resolves for insights in the loaded graph slice; corpus-wide CIL
// mentions outside the slice fall back to the empty-node state (insight
// out-of-slice rendering is a separate follow-up, like the person/topic gate).
function openMentionInsight(insightId: string | null): void {
  const id = insightId?.trim()
  if (id) subject.focusGraphNode(id)
}
function episodeSummaryTitle(ep: CilArcEpisodeBlock): string {
  return ep.summary_title?.trim() ?? ""
}
function episodeSummaryText(ep: CilArcEpisodeBlock): string {
  return ep.summary_text?.trim() ?? ""
}

// Pagination — a topic can span hundreds of episodes; render one page at a time
// so the DOM stays small on huge corpora (a single scroll list would choke).
const EPISODES_PAGE_SIZE = 8
const MENTIONS_PAGE_SIZE = 12
const episodesPage = ref(1)
const mentionsPage = ref(1)
const episodesTotalPages = computed(() =>
  Math.max(1, Math.ceil(inlineTimelineSortedEpisodes.value.length / EPISODES_PAGE_SIZE))
)
const mentionsTotalPages = computed(() =>
  Math.max(1, Math.ceil(flatMentions.value.length / MENTIONS_PAGE_SIZE))
)
const pagedEpisodes = computed((): CilArcEpisodeBlock[] => {
  const start = (episodesPage.value - 1) * EPISODES_PAGE_SIZE
  return inlineTimelineSortedEpisodes.value.slice(start, start + EPISODES_PAGE_SIZE)
})
const pagedMentions = computed((): FlatMentionRow[] => {
  const start = (mentionsPage.value - 1) * MENTIONS_PAGE_SIZE
  return flatMentions.value.slice(start, start + MENTIONS_PAGE_SIZE)
})
// Reset paging whenever the data, sort, or mentions episode filter changes.
watch([inlineTimelineSortOrder, () => inlineTimelinePayload.value], () => {
  episodesPage.value = 1
})
watch([mentionsSortOrder, mentionsEpisodeFilter, () => inlineTimelinePayload.value], () => {
  mentionsPage.value = 1
})

async function loadInlineTimeline(): Promise<void> {
  const path = (shell.resolvedCorpusPath ?? shell.corpusPath).trim()
  if (!path || cilTimelineApiUnavailable.value) {
    inlineTimelineError.value = "Set corpus path and ensure CIL API is available."
    inlineTimelinePayload.value = null
    return
  }
  inlineTimelineLoading.value = true
  inlineTimelineError.value = null
  inlineTimelinePayload.value = null
  try {
    if (!inlineTimelineTopicId.value) {
      inlineTimelineError.value = "Missing topic id."
      return
    }
    inlineTimelinePayload.value = await fetchTopicTimeline(path, inlineTimelineTopicId.value)
  } catch (e) {
    inlineTimelineError.value = e instanceof Error ? e.message : String(e)
  } finally {
    inlineTimelineLoading.value = false
  }
}

type RailFullTextCopyUi = "idle" | "copied" | "failed"

const railFullTextCopyUi = ref<RailFullTextCopyUi>("idle")
let railFullTextCopyResetTimer: ReturnType<typeof setTimeout> | null = null

function resetRailFullTextCopyUi(): void {
  if (railFullTextCopyResetTimer !== null) {
    clearTimeout(railFullTextCopyResetTimer)
    railFullTextCopyResetTimer = null
  }
  railFullTextCopyUi.value = "idle"
}

watch(
  () => [props.nodeId, fullRailTextContent.value] as const,
  () => {
    resetRailFullTextCopyUi()
  }
)

async function copyFullRailText(): Promise<void> {
  const text = fullRailTextContent.value
  if (!text) return
  resetRailFullTextCopyUi()
  const ok = await copyTextToClipboard(text)
  railFullTextCopyUi.value = ok ? "copied" : "failed"
  railFullTextCopyResetTimer = setTimeout(() => {
    railFullTextCopyUi.value = "idle"
    railFullTextCopyResetTimer = null
  }, 2000)
}

const railFullTextCopyAriaLabel = computed((): string => {
  if (railFullTextCopyUi.value === "copied") return "Copied to clipboard"
  if (railFullTextCopyUi.value === "failed") return "Copy failed; try again"
  return "Copy title"
})

/** Native tooltip on the **C** chip (matches ``aria-label``). */
const railFullTextCopyTitleTooltip = computed((): string => railFullTextCopyAriaLabel.value)

const railFullTextSectionTestId = computed((): string => {
  if (isQuoteNode.value) return "node-detail-full-quote"
  if (isTopicNode.value) return "node-detail-full-topic"
  if (isInsightNode.value) return "node-detail-full-insight"
  if (isPersonEntityRailNode.value) return "node-detail-full-person-entity"
  return "node-detail-full-primary"
})

const railFullTextCopyTestId = computed((): string => {
  if (isQuoteNode.value) return "node-detail-full-quote-copy"
  if (isTopicNode.value) return "node-detail-full-topic-copy"
  if (isInsightNode.value) return "node-detail-full-insight-copy"
  if (isPersonEntityRailNode.value) return "node-detail-full-person-entity-copy"
  return "node-detail-full-primary-copy"
})

const visualType = computed(() => {
  const inferred = inferredKindFromId.value
  if (!node.value && inferred) {
    if (inferred === "Person") return "Entity_person"
    if (inferred === "Organization") return "Entity_organization"
    if (inferred === "Podcast") return "Podcast"
    if (inferred === "Insight") return "Insight"
    return "Topic"
  }
  return visualGroupForNode(node.value)
})

const avatarVisualGroup = computed(() => visualType.value)

const personEntityExploreUsesSpeakerFilter = computed(
  () => visualType.value !== "Entity_organization"
)

const personEntityExploreButtonLabel = computed((): string =>
  personEntityExploreUsesSpeakerFilter.value ? "Speaker filter" : "Topic filter"
)

const personEntityExploreHandoffAriaLabel = computed((): string =>
  personEntityExploreUsesSpeakerFilter.value
    ? "Open Explore with Speaker contains filled from this name; topic filter cleared"
    : "Open Explore with Topic contains filled from this organization name; speaker filter cleared"
)

const nodeTypeAvatarStyle = computed((): Record<string, string> => {
  const c = graphNodeTypeChrome(avatarVisualGroup.value)
  return {
    backgroundColor: c.background,
    border: `3px solid ${c.border}`,
    color: c.labelColor,
  }
})

const avatarLetter = computed(() => graphTypeAvatarLetter(avatarVisualGroup.value))

const graphNodeIdTooltip = computed((): string => {
  const id = props.nodeId
  if (!id) return ""
  return (
    `Graph node id (Cytoscape / merged graph): ${id}. ` +
    "Same id as in artifact edges; use for support and tooling."
  )
})

const entityKind = computed(() => {
  const p = node.value?.properties
  if (!p) return null
  const kd = typeof p.kind === "string" ? p.kind.trim().toLowerCase() : ""
  if (kd === "org") return "organization"
  if (kd === "person") return "person"
  if (kd === "object") return "object"
  const ek = p.entity_kind
  if (typeof ek === "string" && ek.trim()) return ek.trim()
  // #2057: a TYPED node (schema 2.x) carries its kind in `type`, not in `properties.kind` — so an
  // Object node reached here with nothing to report and the rail showed no kind at all, while a
  // legacy `Entity` node beside it showed one. Fall back to the node's own type.
  const t = typeof node.value?.type === "string" ? node.value.type.trim().toLowerCase() : ""
  if (t === "object" || t === "organization" || t === "person") return t
  return null
})

const bodyText = computed(() => {
  const n = node.value
  if (!n) return null
  const p = n.properties
  if (!p) return null
  const explicit = p.text ?? p.description
  if (typeof explicit === "string" && explicit.trim()) return explicit.trim()
  const label = p.label ?? p.name ?? p.title
  if (typeof label === "string" && label.length > displayName.value.length) {
    return label.trim()
  }
  return null
})

/** Omit body paragraph when it only repeats the full title (Quote / Topic / Insight use full-text block). */
const bodyTextForTemplate = computed(() => {
  const b = bodyText.value
  if (!b) return null
  const full = displayName.value.trim()
  if (full && b.trim() === full) return null
  return b
})

/**
 * ``entity_kind`` is often ``episode`` on grounded insights; do not show that as a second title under **Insight**.
 */
const showEntityKindSubtitle = computed(() => {
  const ek = entityKind.value
  if (!ek?.trim()) return false
  const nt = nodeType.value
  if (nt !== "Episode" && ek.trim().toLowerCase() === "episode") {
    return false
  }
  return true
})

// Graph type + entity kind. When embedded in the node rail, the rail header
// already names the kind (Person / Topic / Podcast …), so the "PERSON · person"
// kicker is pure redundancy — only show it standalone.
const showNodeKindRowInDetails = computed(
  () => !props.embedInRail && (showEntityKindSubtitle.value || Boolean(nodeType.value?.trim()))
)

const showNodeTypeChipInDetails = computed(
  () => !props.embedInRail && Boolean(nodeType.value?.trim())
)

const transcriptSourceSection = computed(() => {
  const p = node.value?.properties
  if (!p || typeof p !== "object") {
    return null
  }
  const ref = typeof p.transcript_ref === "string" ? p.transcript_ref.trim() : ""
  const charLine = formatTranscriptCharRange(p.char_start, p.char_end)
  const timeLine = formatAudioTimingRange(p.timestamp_start_ms, p.timestamp_end_ms)
  const sp = p.speaker_id
  let speakerLine = ""
  if (typeof sp === "string" && sp.trim()) {
    // Humanize the raw speaker id so the chip shows "Pushkin", not "person:pushkin" (#1011).
    speakerLine = quoteAttributionDisplayFromId(sp.trim())
  } else if (typeof sp === "number" && Number.isFinite(sp)) {
    speakerLine = String(sp)
  }
  if (!ref && !charLine && !timeLine && !speakerLine) {
    return null
  }
  // Slim rail: char range and audio timing live in the transcript dialog; need a file ref or speaker to show.
  if (!ref && !speakerLine) {
    return null
  }
  const rawEp = p.episode_id
  const quoteEpId =
    typeof rawEp === "string" && rawEp.trim()
      ? rawEp.trim()
      : typeof rawEp === "number" && Number.isFinite(rawEp)
        ? String(rawEp)
        : null
  const giPath = resolveGiPathForTranscript(fullMergedArtifactForMetadata.value, quoteEpId)
  const root = shell.corpusPath.trim()
  const resolvedRelpath = ref ? resolveTranscriptCorpusRelpath(ref, giPath) : ""
  const href =
    shell.healthStatus && root && resolvedRelpath
      ? corpusTextFileViewUrl(root, resolvedRelpath)
      : null
  const rawNt = node.value?.type
  const isQuoteKind = typeof rawNt === "string" && rawNt.trim().toLowerCase() === "quote"
  const showMissingSpeakerHint = isQuoteKind && Boolean(ref) && !speakerLine
  return { ref, charLine, timeLine, speakerLine, href, resolvedRelpath, showMissingSpeakerHint }
})

function openTranscriptViewer(): void {
  const sec = transcriptSourceSection.value
  const root = shell.corpusPath.trim()
  if (!sec?.href || !root || !sec.resolvedRelpath) {
    return
  }
  const p = node.value?.properties
  transcriptViewerRef.value?.open({
    corpusRoot: root,
    transcriptRelpath: sec.resolvedRelpath,
    rawTabUrl: sec.href,
    charStart: p?.char_start,
    charEnd: p?.char_end,
    audioTimingLabel: sec.timeLine,
    charPositionLabel: sec.charLine,
    subtitle: sec.ref || null,
    audioSeekStartMs: p?.timestamp_start_ms,
  })
}

function openInsightAllSupportingQuotesTranscript(): void {
  const agg = insightSupportingTranscriptAgg.value
  const root = shell.corpusPath.trim()
  if (!agg || !root) {
    return
  }
  const giPath = resolveGiPathForTranscript(fullMergedArtifactForMetadata.value, agg.episodeId)
  const resolvedRelpath = resolveTranscriptCorpusRelpath(agg.transcriptRef, giPath)
  const href = corpusTextFileViewUrl(root, resolvedRelpath)
  if (!href || !resolvedRelpath) {
    return
  }
  const n = agg.charRanges.length
  const first = agg.charRanges[0]
  const charPositionLabel =
    n === 1
      ? formatTranscriptCharRange(first.charStart, first.charEnd)
      : `${n} character spans (supporting quotes)`
  transcriptViewerRef.value?.open({
    corpusRoot: root,
    transcriptRelpath: resolvedRelpath,
    rawTabUrl: href,
    charRanges: agg.charRanges,
    charPositionLabel,
    subtitle: agg.transcriptRef || null,
    audioTimingLabel: null,
  })
}

function humanizePropertyKey(k: string): string {
  return k.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase())
}

const extraProps = computed(() => {
  const p = node.value?.properties
  if (!p || typeof p !== "object") return []
  const out: { k: string; v: string }[] = []
  for (const k of Object.keys(p).sort()) {
    if (HIDDEN_PROPS.has(k)) continue
    if (TRANSCRIPT_ANCHOR_PROP_KEYS.has(k)) continue
    if (isTopicNode.value && k === "aliases") continue
    if (isPersonEntityRailNode.value && k === "aliases") continue
    if (isInsightNode.value && INSIGHT_META_DL_KEYS.has(k)) continue
    // Corpus episode id is redundant with the **E** header chip on non-Episode nodes (Insight, Quote, …).
    if (nodeType.value !== "Episode" && k === "episode_id") continue
    let v: unknown = p[k]
    if (v === null || v === undefined || v === "") continue
    if (typeof v === "boolean") {
      v = v ? "yes" : "no"
    } else if (typeof v === "object") {
      try {
        v = JSON.stringify(v)
      } catch {
        v = String(v)
      }
    } else {
      v = String(v)
    }
    out.push({ k, v: truncate(String(v), 400) })
  }
  return out
})

const nodeDiagnosticsEntries = computed((): { label: string; value: string }[] => {
  const id = props.nodeId
  const n = node.value
  if (!id || !n) return []
  const rows: { label: string; value: string }[] = [
    { label: "Graph node id", value: id },
    { label: "Type", value: nodeType.value },
    { label: "Visual group", value: visualType.value },
  ]
  if (entityKind.value) {
    rows.push({ label: "Entity kind", value: entityKind.value })
  }
  return rows
})

const crossLayerBridgeLine = computed(() => {
  const row = bridgeIdentityForGraphNodeId(props.bridgeDocument ?? null, props.nodeId)
  if (!row) return ""
  return crossLayerPresenceLabel(row.sources)
})

type GraphRailDetailTab =
  "details" | "timeline" | "position_tracker" | "enrichment" | "neighbourhood" | "perspectives"

const graphRailDetailTab = ref<GraphRailDetailTab>("details")

// The Timeline tab (corpus-wide Episodes + Mentions) exists only for topic /
// nodes that actually have a timeline; a segmented toggle inside it
// switches between the two views. If it empties on a node switch, fall back.
const timelineView = ref<"episodes" | "mentions">("episodes")
const hasTimelineTab = computed(() => showInlineTopicTimeline.value)
watch(hasTimelineTab, (has) => {
  if (!has && graphRailDetailTab.value === "timeline") {
    graphRailDetailTab.value = "details"
  }
})

// Person nodes get a top-level Position Tracker tab (the PLV Position Tracker,
// promoted from a nested sub-tab). Fall back to Details if it empties.
const hasPositionTrackerTab = computed(() => isPersonNode.value)
watch(hasPositionTrackerTab, (has) => {
  if (!has && graphRailDetailTab.value === "position_tracker") {
    graphRailDetailTab.value = "details"
  }
})

// Topic nodes get a Perspectives tab (#1146) — each speaker's grounded insights on the
// topic, grouped by speaker. Shown for every topic node; the panel handles the rare
// empty case. Fall back to Details when the node is no longer a topic.
const hasPerspectivesTab = computed(() => isTopicNode.value)
watch(hasPerspectivesTab, (has) => {
  if (!has && graphRailDetailTab.value === "perspectives") {
    graphRailDetailTab.value = "details"
  }
})
const perspectivesCorpusPath = computed(() =>
  (shell.resolvedCorpusPath ?? shell.corpusPath ?? "").trim()
)
const perspectivesTopicId = computed(() => stripLayerPrefixesForCil(props.nodeId ?? ""))
// Picking a topic from the profile's ranked-topics list (or Insights-voiced)
// sets ``positionTrackerTopicId`` but the arc renders in the Positions tab — a
// different rail tab than the profile. Auto-switch so the pick has a visible
// effect instead of silently setting state behind a hidden tab.
watch(
  () => subject.positionTrackerTopicId,
  (topicId) => {
    if (topicId && hasPositionTrackerTab.value) {
      graphRailDetailTab.value = "position_tracker"
    }
  }
)

// The Enrichment tab only exists when the focused node actually has signals;
// NodeEnrichmentSection reports this after it loads. Empty is the common case
// for graph nodes, so we hide the tab rather than surface a dead panel. If the
// user is on the tab when it empties (node switch), fall back to Details.
const enrichmentHasContent = ref(false)
// Mentions-by-month timeline for the focused node — lives in the Signals tab.
const signalsTimeline = computed(() =>
  buildSubjectMentionsTimeline(fullMergedArtifactForMetadata.value, props.nodeId ?? "")
)
const hasSignalsTimeline = computed(
  () => signalsTimeline.value.total > 0 || signalsTimeline.value.undated > 0
)
// The Signals tab appears when EITHER the enrichers reported content OR there's
// a mentions timeline for this node.
const signalsTabHasContent = computed(() => enrichmentHasContent.value || hasSignalsTimeline.value)
watch(signalsTabHasContent, (has) => {
  if (!has && graphRailDetailTab.value === "enrichment") {
    graphRailDetailTab.value = "details"
  }
})

watch(
  () => props.nodeId,
  (newId) => {
    graphRailDetailTab.value = "details"
    timelineView.value = "episodes"
    mentionsEpisodeFilter.value = null
    // #6 L0 — in the graph rail only, record each navigated-to node on the breadcrumb trail so the
    // graph loads + shows it. addToTrail skips the pinned ego origin; a new origin resets the trail.
    const t = newId?.trim()
    if (props.embedInRail && t) {
      graphNav.addToTrail(t)
      graphAnalytics.track("graph_rail_nav", {
        to_id: t,
        to_kind: t.replace(/^g:/, "").split(":")[0] || "unknown",
        trail_size: graphNav.trailNodeIds.length,
      })
    }
  }
)

watch(
  () =>
    [
      props.nodeId,
      shell.corpusPath,
      shell.resolvedCorpusPath,
      shell.healthStatus,
      shell.cilQueriesApiAvailable,
    ] as const,
  () => {
    inlineTimelineSortOrder.value = "desc"
    inlineTimelineError.value = null
    inlineTimelinePayload.value = null
    if (showInlineTopicTimeline.value) {
      inlineTimelineMode.value = "single"
      inlineTimelineTopicId.value = props.nodeId?.trim() ?? ""
      void loadInlineTimeline()
      return
    }
    inlineTimelineMode.value = null
    inlineTimelineTopicId.value = ""
  },
  { immediate: true }
)

/** Same gate as ``GraphConnectionsSection`` ``centerInView`` (minimap + list hidden when false). */
const graphConnectionsCenterInView = computed((): boolean => {
  const a = props.viewArtifact
  const id = props.nodeId?.trim()
  return Boolean(a && id && findRawNodeInArtifact(a, id))
})
</script>

<template>
  <aside
    v-if="nodeId && (node || inferredKindFromId)"
    class="relative z-20 text-surface-foreground"
    :class="
      props.embedInRail
        ? 'flex min-h-0 w-full min-w-0 flex-1 flex-col border-0 bg-surface shadow-none'
        : 'border-l border-border bg-elevated text-elevated-foreground shadow-lg'
    "
    :style="props.embedInRail ? undefined : { width: '280px' }"
  >
    <div
      class="flex items-start justify-between gap-2"
      :class="
        props.embedInRail ? 'border-b border-border px-2 py-2' : 'border-b border-border px-3 py-2'
      "
    >
      <div class="flex min-w-0 flex-1 gap-3">
        <div
          class="flex h-[4.5rem] w-[4.5rem] shrink-0 items-center justify-center overflow-hidden rounded-2xl text-2xl font-black leading-none shadow-md ring-1 ring-black/15 dark:ring-white/15"
          :style="
            isPodcastNode && (podcastCover?.imageUrl || podcastCover?.imageLocalRelpath)
              ? undefined
              : nodeTypeAvatarStyle
          "
          aria-hidden="true"
        >
          <!-- P2 — podcast nodes show the real show cover instead of the "P" letter. -->
          <PodcastCover
            v-if="isPodcastNode && (podcastCover?.imageUrl || podcastCover?.imageLocalRelpath)"
            :corpus-path="corpusPathForCovers"
            :feed-image-url="podcastCover?.imageUrl ?? null"
            :feed-image-local-relpath="podcastCover?.imageLocalRelpath ?? null"
            alt=""
            size-class="h-full w-full"
          />
          <template v-else>{{ avatarLetter }}</template>
        </div>
        <div class="min-h-0 min-w-0 flex-1 basis-0">
          <div class="flex min-w-0 items-start justify-between gap-1">
            <h3 class="min-w-0 flex-1 basis-0">
              <span
                class="node-detail-primary-title select-text block w-full min-w-0 text-base font-semibold leading-snug text-surface-foreground"
                :title="primaryTitleNativeTooltip"
                :data-testid="fullRailTextContent ? railFullTextSectionTestId : undefined"
                tabindex="0"
                >{{ displayName }}</span
              >
            </h3>
            <div class="ml-1 flex shrink-0 flex-col items-end gap-0.5 self-start pt-0.5">
              <button
                v-if="nodeId"
                type="button"
                :class="SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS"
                :aria-label="graphNodeIdTooltip"
                :title="graphNodeIdTooltip"
                @click.stop.prevent
              >
                E
              </button>
              <HelpTip
                v-if="nodeId"
                :pref-width="300"
                :button-class="SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS"
                button-aria-label="Graph node diagnostics"
              >
                <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
                  Troubleshooting
                </p>
                <p class="mb-2 font-sans text-[10px] text-muted">
                  Technical ids if you need help or file a bug — same node the graph and files use.
                </p>
                <dl class="space-y-1.5 font-mono text-[10px] leading-snug">
                  <template v-for="(row, di) in nodeDiagnosticsEntries" :key="di">
                    <dt class="font-sans font-medium text-muted">
                      {{ row.label }}
                    </dt>
                    <dd class="break-words text-elevated-foreground">
                      {{ row.value }}
                    </dd>
                  </template>
                </dl>
              </HelpTip>
              <button
                v-if="fullRailTextContent"
                type="button"
                :class="SEARCH_RESULT_COPY_TITLE_CHIP_CLASS"
                :aria-label="railFullTextCopyAriaLabel"
                :title="railFullTextCopyTitleTooltip"
                :data-testid="railFullTextCopyTestId"
                @click="copyFullRailText"
              >
                C
              </button>
              <button
                v-if="!props.embedInRail"
                type="button"
                class="shrink-0 text-xs text-muted hover:text-canvas-foreground"
                aria-label="Close detail"
                @click="emit('close')"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <nav
      v-if="props.embedInRail"
      class="flex shrink-0 gap-1 border-b border-border bg-elevated/50 px-2 py-1.5"
      role="tablist"
      aria-label="Graph node detail sections"
    >
      <button
        id="node-detail-rail-tab-details"
        type="button"
        role="tab"
        class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
        :class="
          graphRailDetailTab === 'details'
            ? 'bg-primary text-primary-foreground'
            : 'text-elevated-foreground hover:bg-overlay'
        "
        :aria-selected="graphRailDetailTab === 'details'"
        aria-controls="node-detail-rail-panel-details"
        data-testid="node-detail-rail-tab-details"
        :tabindex="graphRailDetailTab === 'details' ? 0 : -1"
        @click="graphRailDetailTab = 'details'"
      >
        Details
      </button>
      <button
        v-if="hasTimelineTab"
        id="node-detail-rail-tab-timeline"
        type="button"
        role="tab"
        class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
        :class="
          graphRailDetailTab === 'timeline'
            ? 'bg-primary text-primary-foreground'
            : 'text-elevated-foreground hover:bg-overlay'
        "
        :aria-selected="graphRailDetailTab === 'timeline'"
        aria-controls="node-detail-rail-panel-details"
        data-testid="node-detail-rail-tab-timeline"
        :tabindex="graphRailDetailTab === 'timeline' ? 0 : -1"
        @click="graphRailDetailTab = 'timeline'"
      >
        Timeline
      </button>
      <button
        v-if="hasPerspectivesTab"
        id="node-detail-rail-tab-perspectives"
        type="button"
        role="tab"
        class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
        :class="
          graphRailDetailTab === 'perspectives'
            ? 'bg-primary text-primary-foreground'
            : 'text-elevated-foreground hover:bg-overlay'
        "
        :aria-selected="graphRailDetailTab === 'perspectives'"
        aria-controls="node-detail-rail-panel-perspectives"
        data-testid="node-detail-rail-tab-perspectives"
        :tabindex="graphRailDetailTab === 'perspectives' ? 0 : -1"
        @click="graphRailDetailTab = 'perspectives'"
      >
        Perspectives
      </button>
      <button
        v-if="hasPositionTrackerTab"
        id="node-detail-rail-tab-position-tracker"
        type="button"
        role="tab"
        class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
        :class="
          graphRailDetailTab === 'position_tracker'
            ? 'bg-primary text-primary-foreground'
            : 'text-elevated-foreground hover:bg-overlay'
        "
        :aria-selected="graphRailDetailTab === 'position_tracker'"
        aria-controls="node-detail-rail-panel-position-tracker"
        data-testid="node-detail-rail-tab-position-tracker"
        :tabindex="graphRailDetailTab === 'position_tracker' ? 0 : -1"
        @click="graphRailDetailTab = 'position_tracker'"
      >
        Positions
      </button>
      <button
        v-if="signalsTabHasContent"
        id="node-detail-rail-tab-enrichment"
        type="button"
        role="tab"
        class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
        :class="
          graphRailDetailTab === 'enrichment'
            ? 'bg-primary text-primary-foreground'
            : 'text-elevated-foreground hover:bg-overlay'
        "
        :aria-selected="graphRailDetailTab === 'enrichment'"
        aria-controls="node-detail-rail-panel-enrichment"
        data-testid="node-detail-rail-tab-enrichment"
        :tabindex="graphRailDetailTab === 'enrichment' ? 0 : -1"
        @click="graphRailDetailTab = 'enrichment'"
      >
        Signals
      </button>
      <button
        id="node-detail-rail-tab-neighbourhood"
        type="button"
        role="tab"
        class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
        :class="
          graphRailDetailTab === 'neighbourhood'
            ? 'bg-primary text-primary-foreground'
            : 'text-elevated-foreground hover:bg-overlay'
        "
        :aria-selected="graphRailDetailTab === 'neighbourhood'"
        aria-controls="node-detail-rail-panel-neighbourhood"
        data-testid="node-detail-rail-tab-neighbourhood"
        :tabindex="graphRailDetailTab === 'neighbourhood' ? 0 : -1"
        @click="graphRailDetailTab = 'neighbourhood'"
      >
        Neighbourhood
      </button>
    </nav>

    <div
      class="py-2"
      :class="
        props.embedInRail
          ? 'min-h-0 w-full min-w-0 flex-1 overflow-y-auto px-2'
          : 'overflow-y-auto px-3'
      "
      :style="props.embedInRail ? undefined : { maxHeight: 'calc(100vh - 12rem)' }"
    >
      <div
        v-show="
          !props.embedInRail ||
          graphRailDetailTab === 'details' ||
          graphRailDetailTab === 'timeline'
        "
        id="node-detail-rail-panel-details"
        class="min-h-0"
        :role="props.embedInRail ? 'tabpanel' : undefined"
        :aria-labelledby="
          props.embedInRail
            ? graphRailDetailTab === 'timeline'
              ? 'node-detail-rail-tab-timeline'
              : 'node-detail-rail-tab-details'
            : undefined
        "
        :tabindex="props.embedInRail ? -1 : undefined"
      >
        <div v-show="!props.embedInRail || graphRailDetailTab === 'details'" class="min-h-0">
          <p
            v-if="showNodeKindRowInDetails"
            class="mb-2 text-xs text-muted"
            data-testid="node-detail-kind-row"
          >
            <span
              v-if="showNodeTypeChipInDetails"
              class="inline-block rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary"
              >{{ nodeType }}</span
            >
            <template v-if="showEntityKindSubtitle">
              <span v-if="showNodeTypeChipInDetails" class="mx-1 text-muted">·</span>
              <span>{{ entityKind }}</span>
            </template>
          </p>
          <!-- Actions first: Search / Explore shortcuts + help lead the topic
           Details tab, above every other section (incl. Member topics). -->
          <div
            v-if="isTopicNode && topicGatewayQuery"
            class="mb-3 flex items-center gap-2"
            role="group"
            aria-label="Topic shortcuts to Search and Explore"
          >
            <button
              type="button"
              class="min-w-0 flex-1 rounded bg-primary px-2 py-1.5 text-center text-xs font-medium leading-snug text-primary-foreground disabled:opacity-40"
              data-testid="node-detail-topic-prefill-search"
              :disabled="!shell.healthStatus"
              @click="emitTopicPrefillSearch"
            >
              Prefill semantic search
            </button>
            <button
              type="button"
              class="min-w-0 flex-1 rounded bg-gi px-2 py-1.5 text-center text-xs font-medium leading-snug text-gi-foreground disabled:opacity-40"
              data-testid="node-detail-topic-explore-filter"
              :disabled="!shell.healthStatus"
              @click="emitTopicExploreFilter"
            >
              Set Search topic filter
            </button>
            <button
              v-if="showViewInGraph && nodeId?.trim()"
              type="button"
              class="min-w-0 flex-1 rounded border border-border px-2 py-1.5 text-center text-xs font-medium leading-snug text-surface-foreground hover:bg-overlay"
              data-testid="node-detail-topic-view-in-graph"
              @click="emit('view-in-graph', nodeId!.trim())"
            >
              View in graph
            </button>
            <HelpTip
              class="shrink-0 self-center"
              :pref-width="280"
              button-aria-label="About topic Search and Explore shortcuts"
            >
              <p class="font-sans text-[10px] leading-snug text-muted">
                <strong class="font-medium text-surface-foreground">Prefill semantic search</strong>
                switches to Search with this topic label as the query (clears feed filter). Run
                Search to hit the vector index.
                <strong class="font-medium text-surface-foreground">Set Search topic filter</strong>
                fills the Search chip bar's
                <strong class="font-medium text-surface-foreground/90">Topic contains</strong>
                chip. (Both labels said "Explore" until #1600 — that panel was folded into Search,
                so they named a surface that no longer exists.)
              </p>
            </HelpTip>
          </div>
          <template v-if="isPersonEntityRailNode">
            <p
              v-if="personEntityRoleLabel"
              class="mb-1 text-[11px] leading-tight text-muted"
              data-testid="node-detail-person-entity-role"
            >
              Role: <span class="text-surface-foreground">{{ personEntityRoleLabel }}</span>
            </p>
            <p
              v-if="personEntityAliasesLine"
              class="mb-2 text-[11px] leading-tight text-muted"
              data-testid="node-detail-person-entity-aliases"
            >
              Aliases: <span class="text-surface-foreground">{{ personEntityAliasesLine }}</span>
            </p>
            <div
              v-if="personEntityGatewayQuery"
              class="mb-3 flex items-center gap-2"
              role="group"
              aria-label="Person and entity shortcuts to Search and Explore"
            >
              <button
                type="button"
                class="min-w-0 flex-1 rounded bg-primary px-2 py-1.5 text-center text-xs font-medium leading-snug text-primary-foreground disabled:opacity-40"
                data-testid="node-detail-person-entity-prefill-search"
                :disabled="!shell.healthStatus"
                @click="emitPersonEntityPrefillSearch"
              >
                Prefill semantic search
              </button>
              <button
                type="button"
                class="min-w-0 flex-1 rounded bg-gi px-2 py-1.5 text-center text-xs font-medium leading-snug text-gi-foreground disabled:opacity-40"
                data-testid="node-detail-person-entity-explore-filter"
                :aria-label="personEntityExploreHandoffAriaLabel"
                :disabled="!shell.healthStatus"
                @click="emitPersonEntityExploreHandoff"
              >
                {{ personEntityExploreButtonLabel }}
              </button>
              <HelpTip
                class="shrink-0 self-center"
                :pref-width="300"
                button-aria-label="About Person, Entity, and Explore shortcuts"
              >
                <p class="font-sans text-[10px] leading-snug text-muted">
                  <strong class="font-medium text-surface-foreground"
                    >Prefill semantic search</strong
                  >
                  uses this primary name as the query against the vector index. For
                  <strong class="font-medium text-surface-foreground/90">people</strong> (GI
                  <span class="font-mono">Person</span> / legacy
                  <span class="font-mono">Speaker</span>, or KG
                  <span class="font-mono">Entity</span> marked as a person),
                  <strong class="font-medium text-surface-foreground">Speaker filter</strong> fills
                  <strong class="font-medium text-surface-foreground/90">Speaker contains</strong>
                  and clears the topic filter. For
                  <strong class="font-medium text-surface-foreground/90">organizations</strong> (KG
                  <span class="font-mono">Entity</span> with organization-style kind), the second
                  button
                  <strong class="font-medium text-surface-foreground">Topic filter</strong> fills
                  <strong class="font-medium text-surface-foreground/90">Topic contains</strong>
                  (same control as the topic graph row) and clears the speaker filter. You still run
                  <strong class="font-medium text-surface-foreground/90">Explore</strong>.
                </p>
              </HelpTip>
            </div>
          </template>

          <div
            v-if="isInsightNode && hasInsightDetailsTipContent"
            class="mb-2"
            data-testid="node-detail-insight-details-tip"
          >
            <HelpTip
              :key="String(props.nodeId ?? '')"
              :pref-width="320"
              :button-text="insightGroundingTriggerLabel"
              :button-aria-label="insightGroundingTriggerLabel"
              :button-class="INSIGHT_GROUNDING_TIP_TRIGGER_CLASS"
            >
              <div
                data-testid="node-detail-insight-details-tooltip-body"
                class="space-y-2 font-sans text-[10px] leading-snug text-muted"
              >
                <div>
                  <p class="mb-1 font-semibold text-surface-foreground">Grounding</p>
                  <p>
                    <strong class="text-surface-foreground/90">Grounded</strong> means this insight
                    has at least one supporting quote in this episode graph (linked via SUPPORTED_BY
                    in the GI file).
                    <strong class="text-surface-foreground/90">Not grounded</strong> means the
                    artifact marks grounded as false (no supporting quotes recorded for this
                    insight).
                  </p>
                </div>
                <div v-if="insightSecondaryDetailLines.length">
                  <p class="mb-1 font-semibold text-surface-foreground">Other fields</p>
                  <ul class="list-inside list-disc space-y-0.5">
                    <li v-for="(line, li) in insightSecondaryDetailLines" :key="li">
                      {{ line }}
                    </li>
                  </ul>
                </div>
                <div v-if="insightProvenanceText">
                  <p class="mb-1 font-semibold text-surface-foreground">Lineage</p>
                  <p class="break-words font-mono text-[10px] text-elevated-foreground">
                    {{ insightProvenanceText }}
                  </p>
                </div>
              </div>
            </HelpTip>
          </div>

          <div
            v-if="isInsightNode && insightSemanticSearchQuery"
            class="mb-3 flex items-center gap-2"
            role="group"
            aria-label="Insight shortcuts to Search and Explore"
          >
            <button
              type="button"
              class="min-w-0 flex-1 rounded bg-primary px-2 py-1.5 text-center text-xs font-medium leading-snug text-primary-foreground disabled:opacity-40"
              data-testid="node-detail-insight-prefill-search"
              :disabled="!shell.healthStatus"
              @click="emitInsightPrefillSearch"
            >
              Prefill semantic search
            </button>
            <button
              type="button"
              class="min-w-0 flex-1 rounded bg-gi px-2 py-1.5 text-center text-xs font-medium leading-snug text-gi-foreground disabled:opacity-40"
              data-testid="node-detail-insight-explore-filters"
              aria-label="Open Explore with grounded and min confidence from this insight; topic and speaker filters cleared"
              :disabled="!shell.healthStatus"
              @click="emitInsightOpenSearchFilters"
            >
              Set Search filters
            </button>
            <HelpTip
              class="shrink-0 self-center"
              :pref-width="280"
              button-aria-label="About insight Search and Explore shortcuts"
            >
              <p class="font-sans text-[10px] leading-snug text-muted">
                <strong class="font-medium text-surface-foreground">Prefill semantic search</strong>
                uses the first part of this insight as the query against the vector index.
                <strong class="font-medium text-surface-foreground">Set Search filters</strong>
                clears the topic/speaker chips and sets
                <strong class="font-medium text-surface-foreground/90">Grounded only</strong> plus
                an optional
                <strong class="font-medium text-surface-foreground/90">Min confidence</strong> from
                this node, then you run the search. (This said "Explore" until #1600 — that panel
                was folded into Search.)
              </p>
            </HelpTip>
          </div>

          <section
            v-if="isInsightNode && insightRelatedTopics.length"
            class="mb-3 mt-1 border-t border-border pt-2"
            role="region"
            aria-label="Related topics"
            data-testid="node-detail-insight-related-topics"
          >
            <h4 class="text-xs font-semibold text-surface-foreground">Related topics</h4>
            <div class="mt-1 min-w-0" data-testid="node-detail-insight-related-topics-list">
              <ul class="space-y-1 text-xs">
                <li v-for="row in insightRelatedTopics" :key="row.topicId">
                  <button
                    type="button"
                    class="flex w-full rounded px-1 py-0.5 text-left hover:bg-overlay"
                    data-testid="node-detail-insight-related-topic-row"
                    :aria-label="`${row.label} — show on graph`"
                    :title="graphFocusNeighborTooltip"
                    @click="focusNeighborOnGraph(row.topicId, $event)"
                  >
                    <span class="min-w-0 flex-1 truncate font-medium text-surface-foreground">{{
                      row.label
                    }}</span>
                  </button>
                </li>
              </ul>
            </div>
          </section>

          <section
            v-if="isInsightNode && insightSupportingQuotes.length"
            class="mb-3 rounded border border-border bg-elevated/40 p-2"
            data-testid="node-detail-insight-supporting-quotes"
          >
            <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h4 class="text-[10px] font-semibold uppercase tracking-wide text-muted">
                Supporting quotes
              </h4>
              <button
                v-if="insightOpenAllSupportingQuotesReady"
                type="button"
                class="shrink-0 rounded border border-border bg-canvas px-2 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-elevated"
                data-testid="node-detail-insight-view-transcript-all-quotes"
                aria-label="View transcript with all supporting quotes highlighted"
                @click="openInsightAllSupportingQuotesTranscript"
              >
                Transcript (all quotes)
              </button>
            </div>
            <ul class="space-y-1.5">
              <li
                v-for="row in insightSupportingQuotesVisible"
                :key="row.id"
                class="flex items-start gap-1.5 rounded px-0.5 py-0.5 hover:bg-overlay/60"
              >
                <span class="min-w-0 flex-1 text-xs leading-snug text-muted" :title="row.preview">{{
                  row.preview
                }}</span>
                <button
                  type="button"
                  :class="SEARCH_RESULT_GRAPH_BUTTON_CLASS"
                  :aria-label="graphFocusNeighborTooltip"
                  :title="graphFocusNeighborTooltip"
                  @click="focusNeighborOnGraph(row.id, $event)"
                >
                  G
                </button>
              </li>
            </ul>
            <button
              v-if="insightSupportingQuotes.length > INSIGHT_SUPPORTING_QUOTES_COLLAPSE_AFTER"
              type="button"
              class="mt-2 w-full rounded border border-border bg-canvas px-2 py-1 text-[10px] font-medium text-muted hover:bg-elevated hover:text-surface-foreground"
              data-testid="node-detail-insight-supporting-quotes-toggle-expand"
              @click="toggleInsightSupportingQuotesExpanded"
            >
              {{
                insightQuotesExpanded
                  ? "Show fewer quotes"
                  : `Show all ${insightSupportingQuotes.length}`
              }}
            </button>
          </section>

          <p
            v-if="topicAliasesLine"
            class="mb-3 text-[10px] leading-snug text-muted"
            data-testid="node-detail-topic-aliases"
          >
            <span class="font-medium text-surface-foreground/80">Aliases:</span>
            {{ topicAliasesLine }}
          </p>

          <!-- Topic overview folds in directly (no "open full profile" hop) — the
           subject overview that used to live in the separate TopicEntityView rail
           now renders inline in this Details tab. (Non-person entities keep the
           generic node view — TopicEntityView is topic-oriented and renders empty
           for organizations, so there's nothing to fold.) -->
          <TopicEntityView
            v-if="isTopicNode"
            embedded
            :subject-id-override="nodeId ?? ''"
            class="mb-1"
          />

          <!-- Person overview folds in directly too — PersonLandingView (profile +
           Position Tracker) renders inline instead of a separate rail. -->
          <PersonLandingView
            v-if="isPersonNode"
            embedded
            view="profile"
            :subject-id-override="nodeId ?? ''"
            class="mb-1"
          />

          <!-- FB14 — Podcast/Show node: basics + episode list, from the corpus feed. -->
          <PodcastNodeView
            v-if="isPodcastNode"
            :subject-id-override="nodeId ?? ''"
            class="mb-1"
            @cover="podcastCover = $event"
          />

          <!-- Out-of-slice insight (e.g. a timeline-mention drill): render its own
           content from /relational/insight-detail; in-slice insights use the
           node-backed sections below. -->
          <InsightNodeView
            v-if="isInsightNode && !node"
            :subject-id-override="nodeId ?? ''"
            class="mb-1"
            @resolved="insightHeaderText = $event"
          />
        </div>

        <!-- TIMELINE tab: corpus-wide Episodes (grouped) + Mentions (atomic),
           two views of one API load, switched by a segmented control. -->
        <div
          v-show="!props.embedInRail || graphRailDetailTab === 'timeline'"
          class="min-h-0"
          data-testid="node-detail-timeline-panel"
        >
          <!-- N8 — compact dot time-series over the same CIL data, above the
             segmented control; follows the active view (episodes vs mentions). -->
          <SubjectTimelineChart
            v-if="timelineChartData.months.length > 1"
            :timeline="timelineChartData"
            variant="dots"
            :value-label="timelineView === 'mentions' ? 'Mentions' : 'Episodes'"
            :aria-label="timelineView === 'mentions' ? 'Mentions over time' : 'Episodes over time'"
            class="mb-2"
            data-testid="node-detail-timeline-chart"
          />
          <div class="mb-2 flex items-center justify-between gap-2">
            <div
              role="tablist"
              aria-label="Timeline view"
              class="inline-flex shrink-0 rounded-md border border-border p-0.5"
            >
              <button
                type="button"
                role="tab"
                class="rounded px-2 py-0.5 text-[10px] font-medium transition-colors"
                :class="
                  timelineView === 'episodes'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted hover:bg-overlay'
                "
                :aria-selected="timelineView === 'episodes'"
                data-testid="node-detail-timeline-view-episodes"
                @click="
                  timelineView = 'episodes';
                  mentionsEpisodeFilter = null;
                "
              >
                Episodes
              </button>
              <button
                type="button"
                role="tab"
                class="rounded px-2 py-0.5 text-[10px] font-medium transition-colors"
                :class="
                  timelineView === 'mentions'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted hover:bg-overlay'
                "
                :aria-selected="timelineView === 'mentions'"
                data-testid="node-detail-timeline-view-mentions"
                @click="
                  timelineView = 'mentions';
                  mentionsEpisodeFilter = null;
                "
              >
                Mentions
              </button>
            </div>
            <div class="flex shrink-0 items-center gap-1.5">
              <HelpTip
                class="shrink-0"
                :pref-width="360"
                button-aria-label="About the corpus-wide timeline, its two views, and topic ids"
              >
                <p class="font-sans text-[10px] leading-snug text-muted">
                  <strong class="font-medium text-surface-foreground">Corpus-wide</strong> — every
                  episode and insight about this
                  topic, from CIL + bridge + GI.
                  <strong class="font-medium text-surface-foreground">Episodes</strong> groups by
                  episode (with our summary);
                  <strong class="font-medium text-surface-foreground">Mentions</strong> lists every
                  individual insight / quote.
                </p>
                <p
                  v-if="inlineTimelineTopicIdsLabel"
                  class="mt-2 border-t border-border pt-2 text-[10px] leading-snug text-muted"
                >
                  <span class="mb-1 block font-medium text-surface-foreground">
                    Topic id
                  </span>
                  <span class="block break-all font-mono text-[10px] text-surface-foreground">
                    {{ inlineTimelineTopicIdsLabel }}
                  </span>
                </p>
              </HelpTip>
              <button
                type="button"
                class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
:disabled="inlineTimelineLoading"
                @click="loadInlineTimeline"
              >
                Refresh
              </button>
            </div>
          </div>

          <section
            v-show="timelineView === 'episodes'"
            v-if="showInlineTopicTimeline"
            class="min-w-0 w-full overflow-x-clip overflow-y-visible"
            data-testid="node-detail-inline-timeline"
          >
            <p
              v-if="inlineTimelineLoading"
              class="text-[10px] text-muted"
              data-testid="node-detail-inline-timeline-loading"
            >
              Loading timeline...
            </p>
            <p
              v-else-if="inlineTimelineError"
              class="text-[10px] text-destructive"
              data-testid="node-detail-inline-timeline-error"
            >
              {{ inlineTimelineError }}
            </p>
            <p
              v-else-if="
                inlineTimelinePayload && (inlineTimelinePayload.episodes?.length ?? 0) === 0
              "
              class="text-[10px] text-muted"
              data-testid="node-detail-inline-timeline-empty"
            >
              No matching episodes found in the current corpus path.
            </p>
            <div
              v-else-if="inlineTimelinePayload"
              class="space-y-1.5"
              data-testid="node-detail-inline-timeline-results"
            >
              <p class="text-[10px] leading-snug text-muted">
                {{ inlineTimelineEpisodeCount }}
                {{ inlineTimelineEpisodeCount === 1 ? "episode" : "episodes" }}
                with insights about
                this topic.
              </p>
              <div class="mb-1 flex items-center gap-1.5">
                <span class="text-[10px] text-muted">Date</span>
                <button
                  type="button"
                  class="rounded border px-1.5 py-0.5 text-[10px]"
                  :class="
                    inlineTimelineSortOrder === 'asc'
                      ? 'border-gi/60 bg-gi/15'
                      : 'border-border text-muted'
                  "
                  @click="inlineTimelineSortOrder = 'asc'"
                >
                  Oldest
                </button>
                <button
                  type="button"
                  class="rounded border px-1.5 py-0.5 text-[10px]"
                  :class="
                    inlineTimelineSortOrder === 'desc'
                      ? 'border-gi/60 bg-gi/15'
                      : 'border-border text-muted'
                  "
                  @click="inlineTimelineSortOrder = 'desc'"
                >
                  Newest
                </button>
              </div>
              <ul class="w-full min-w-0 space-y-2 overflow-x-clip pr-0.5">
                <li
                  v-for="ep in pagedEpisodes"
                  :key="episodeKey(ep)"
                  class="min-w-0 border-b border-border/40 pb-2 last:border-b-0"
                >
                  <button
                    type="button"
                    class="flex w-full min-w-0 items-start gap-2 rounded text-left hover:bg-overlay/60"
                    data-testid="node-detail-episode-open"
                    :title="`Show mentions in ${episodePrimaryHeading(ep)}`"
                    @click="openEpisodeMentions(ep)"
                  >
                    <div class="shrink-0 self-start">
                      <PodcastCover
                        :corpus-path="corpusPathForCovers"
                        :episode-image-local-relpath="ep.episode_image_local_relpath"
                        :feed-image-local-relpath="ep.feed_image_local_relpath"
                        :episode-image-url="ep.episode_image_url"
                        :feed-image-url="ep.feed_image_url"
                        :alt="`Cover for ${episodePrimaryHeading(ep)}`"
                        size-class="h-10 w-10"
                      />
                    </div>
                    <div class="min-w-0 flex-1">
                      <p class="text-[10px] font-medium text-gi/90">
                        {{ formatEpisodeDate(ep.publish_date) || "Date unknown" }}
                      </p>
                      <p
                        class="break-words text-[11px] font-semibold leading-snug text-surface-foreground"
                      >
                        {{ episodePrimaryHeading(ep) }}
                      </p>
                      <p
                        v-if="episodeSummaryTitle(ep)"
                        class="break-words text-[10px] leading-snug text-muted"
                      >
                        {{ episodeSummaryTitle(ep) }}
                      </p>
                      <p
                        v-else-if="episodeContextLine(ep)"
                        class="break-words text-[10px] leading-snug text-muted"
                      >
                        {{ episodeContextLine(ep) }}
                      </p>
                    </div>
                    <span
                      class="shrink-0 self-center text-xs leading-none text-muted"
                      aria-hidden="true"
                      >›</span
                    >
                  </button>
                </li>
              </ul>
              <nav
                v-if="episodesTotalPages > 1"
                class="mt-2 flex items-center justify-center gap-1"
                aria-label="Episode pages"
                data-testid="node-detail-episodes-pager"
              >
                <button
                  type="button"
                  class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                  :disabled="episodesPage === 1"
                  aria-label="First page"
                  @click="episodesPage = 1"
                >
                  «
                </button>
                <button
                  type="button"
                  class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                  :disabled="episodesPage === 1"
                  aria-label="Previous page"
                  @click="episodesPage = Math.max(1, episodesPage - 1)"
                >
                  ‹
                </button>
                <span class="px-1.5 text-[10px] tabular-nums text-muted">
                  {{ episodesPage }} / {{ episodesTotalPages }}
                </span>
                <button
                  type="button"
                  class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                  :disabled="episodesPage >= episodesTotalPages"
                  aria-label="Next page"
                  @click="episodesPage = Math.min(episodesTotalPages, episodesPage + 1)"
                >
                  ›
                </button>
                <button
                  type="button"
                  class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                  :disabled="episodesPage >= episodesTotalPages"
                  aria-label="Last page"
                  @click="episodesPage = episodesTotalPages"
                >
                  »
                </button>
              </nav>
            </div>
          </section>

          <!-- MENTIONS: corpus-wide, one row per insight (flattened from the same
           timeline API as Episodes above) — mentions + episodes, two views, one source. -->
          <section
            v-show="timelineView === 'mentions'"
            v-if="showInlineTopicTimeline && flatMentions.length"
            class="min-w-0 w-full"
            data-testid="node-detail-mentions"
            aria-label="Mentions"
          >
            <!-- Drilled in from an episode: show its summary for context + a way back. -->
            <div
              v-if="filteredMentionEpisode"
              class="mb-2 rounded border border-border bg-surface/40 p-2"
              data-testid="node-detail-mentions-episode-filter"
            >
              <div class="flex items-start justify-between gap-2">
                <div class="min-w-0">
                  <p class="text-[10px] font-medium text-gi/90">
                    {{ formatEpisodeDate(filteredMentionEpisode.publish_date) || "Date unknown" }}
                  </p>
                  <p
                    class="break-words text-[11px] font-semibold leading-snug text-surface-foreground"
                  >
                    {{ episodePrimaryHeading(filteredMentionEpisode) }}
                  </p>
                  <p
                    v-if="episodeSummaryText(filteredMentionEpisode)"
                    class="mt-0.5 whitespace-pre-line break-words text-[10px] leading-snug text-muted"
                  >
                    {{ episodeSummaryText(filteredMentionEpisode) }}
                  </p>
                </div>
                <button
                  type="button"
                  class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay"
                  data-testid="node-detail-mentions-clear-filter"
                  @click="mentionsEpisodeFilter = null"
                >
                  ‹ All
                </button>
              </div>
            </div>
            <p class="mb-1 text-[10px] leading-snug text-muted">
              {{ flatMentions.length }} mention{{ flatMentions.length === 1 ? "" : "s"
              }}{{ filteredMentionEpisode ? " in this episode" : "" }}.
            </p>
            <div class="mb-1 flex items-center gap-1.5">
              <span class="text-[10px] text-muted">Date</span>
              <button
                type="button"
                class="rounded border px-1.5 py-0.5 text-[10px]"
                :class="
                  mentionsSortOrder === 'asc' ? 'border-gi/60 bg-gi/15' : 'border-border text-muted'
                "
                data-testid="node-detail-mentions-sort-oldest"
                @click="mentionsSortOrder = 'asc'"
              >
                Oldest
              </button>
              <button
                type="button"
                class="rounded border px-1.5 py-0.5 text-[10px]"
                :class="
                  mentionsSortOrder === 'desc'
                    ? 'border-gi/60 bg-gi/15'
                    : 'border-border text-muted'
                "
                data-testid="node-detail-mentions-sort-newest"
                @click="mentionsSortOrder = 'desc'"
              >
                Newest
              </button>
            </div>
            <ul
              class="w-full min-w-0 space-y-1.5 overflow-x-clip pr-0.5"
              data-testid="node-detail-mentions-list"
            >
              <li
                v-for="m in pagedMentions"
                :key="m.key"
                class="min-w-0 border-b border-border/40 pb-1.5 last:border-b-0"
              >
                <button
                  v-if="m.insightId"
                  type="button"
                  class="w-full min-w-0 rounded text-left hover:bg-overlay/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                  data-testid="node-detail-mention-open"
                  title="Open this insight"
                  @click="openMentionInsight(m.insightId)"
                >
                  <p class="break-words text-[11px] leading-snug text-surface-foreground">
                    {{ m.text }}
                  </p>
                  <p
                    v-if="m.episodeTitle || m.publishDate"
                    class="mt-0.5 break-words text-[10px] leading-snug text-muted"
                  >
                    <span v-if="m.episodeTitle">{{ m.episodeTitle }}</span>
                    <span v-if="m.episodeTitle && m.publishDate"> · </span>
                    <span v-if="m.publishDate">{{ formatEpisodeDate(m.publishDate) }}</span>
                  </p>
                </button>
                <template v-else>
                  <p class="break-words text-[11px] leading-snug text-surface-foreground">
                    {{ m.text }}
                  </p>
                  <p
                    v-if="m.episodeTitle || m.publishDate"
                    class="mt-0.5 break-words text-[10px] leading-snug text-muted"
                  >
                    <span v-if="m.episodeTitle">{{ m.episodeTitle }}</span>
                    <span v-if="m.episodeTitle && m.publishDate"> · </span>
                    <span v-if="m.publishDate">{{ formatEpisodeDate(m.publishDate) }}</span>
                  </p>
                </template>
              </li>
            </ul>
            <nav
              v-if="mentionsTotalPages > 1"
              class="mt-2 flex items-center justify-center gap-1"
              aria-label="Mention pages"
              data-testid="node-detail-mentions-pager"
            >
              <button
                type="button"
                class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                :disabled="mentionsPage === 1"
                aria-label="First page"
                @click="mentionsPage = 1"
              >
                «
              </button>
              <button
                type="button"
                class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                :disabled="mentionsPage === 1"
                aria-label="Previous page"
                @click="mentionsPage = Math.max(1, mentionsPage - 1)"
              >
                ‹
              </button>
              <span class="px-1.5 text-[10px] tabular-nums text-muted">
                {{ mentionsPage }} / {{ mentionsTotalPages }}
              </span>
              <button
                type="button"
                class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                :disabled="mentionsPage >= mentionsTotalPages"
                aria-label="Next page"
                @click="mentionsPage = Math.min(mentionsTotalPages, mentionsPage + 1)"
              >
                ›
              </button>
              <button
                type="button"
                class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
                :disabled="mentionsPage >= mentionsTotalPages"
                aria-label="Last page"
                @click="mentionsPage = mentionsTotalPages"
              >
                »
              </button>
            </nav>
          </section>
        </div>

        <div v-show="!props.embedInRail || graphRailDetailTab === 'details'" class="min-h-0">
          <p v-if="bodyTextForTemplate" class="mb-3 text-xs leading-relaxed text-muted">
            {{ truncate(bodyTextForTemplate, 600) }}
          </p>

          <p v-if="crossLayerBridgeLine" class="mb-2 text-[11px] leading-snug text-primary">
            <span class="font-medium text-muted">Also in other graph layers:</span>
            {{ crossLayerBridgeLine }}
          </p>

          <section v-if="transcriptSourceSection" class="mb-2 flex flex-col gap-1.5">
            <div v-if="transcriptSourceSection.href" class="flex flex-wrap items-center gap-2">
              <button
                type="button"
                class="rounded border border-border bg-canvas px-2 py-0.5 text-[11px] font-medium text-surface-foreground hover:bg-elevated"
                aria-label="View transcript"
                data-testid="node-detail-view-transcript"
                @click="openTranscriptViewer"
              >
                View transcript
              </button>
              <span
                v-if="transcriptSourceSection.showMissingSpeakerHint"
                class="text-[10px] leading-snug text-muted"
                data-testid="node-detail-quote-speaker-unavailable"
              >
                {{ GI_QUOTE_SPEAKER_UNAVAILABLE_HINT }}
              </span>
            </div>
            <div v-else-if="transcriptSourceSection.ref" class="text-[10px] leading-snug">
              <span class="font-mono text-[11px] text-surface-foreground">{{
                transcriptSourceSection.ref
              }}</span>
              <p class="mt-0.5 text-muted">
                Connect to the API and set
                <strong class="font-medium text-surface-foreground/90">Corpus path</strong>
                in the left panel to open this file in your browser.
              </p>
            </div>
            <p
              v-if="transcriptSourceSection.speakerLine"
              class="text-[10px] leading-snug text-muted"
            >
              <span class="font-medium text-surface-foreground/80">Speaker:</span>
              {{ transcriptSourceSection.speakerLine }}
            </p>
            <p
              v-else-if="
                transcriptSourceSection.showMissingSpeakerHint && !transcriptSourceSection.href
              "
              class="text-[10px] leading-snug text-muted"
              data-testid="node-detail-quote-speaker-unavailable"
            >
              {{ GI_QUOTE_SPEAKER_UNAVAILABLE_HINT }}
            </p>
          </section>

          <dl v-if="extraProps.length" class="mb-2 space-y-1.5">
            <template v-for="(r, i) in extraProps" :key="i">
              <dt class="text-[10px] font-medium text-muted">
                {{ humanizePropertyKey(r.k) }}
              </dt>
              <dd class="break-words text-xs text-surface-foreground">
                {{ r.v }}
              </dd>
            </template>
          </dl>
        </div>
      </div>

      <!-- Position Tracker (person nodes) — the PLV Position Tracker promoted to a
           peer node-view tab; the pivot topic is set from the embedded profile. -->
      <div
        v-if="props.embedInRail && hasPositionTrackerTab"
        v-show="graphRailDetailTab === 'position_tracker'"
        id="node-detail-rail-panel-position-tracker"
        class="min-h-0"
        role="tabpanel"
        aria-labelledby="node-detail-rail-tab-position-tracker"
        :tabindex="-1"
      >
        <PersonLandingView
          v-if="isPersonNode"
          embedded
          view="positions"
          :subject-id-override="nodeId ?? ''"
        />
      </div>

      <!-- Mounted whenever the rail is shown (not just when the tab is active) so
           NodeEnrichmentSection can report has-content up-front; the tab button
           only appears once it does. v-show keeps it hidden until selected. -->
      <div
        v-if="props.embedInRail"
        v-show="graphRailDetailTab === 'enrichment'"
        id="node-detail-rail-panel-enrichment"
        class="min-h-0"
        role="tabpanel"
        aria-labelledby="node-detail-rail-tab-enrichment"
        :tabindex="-1"
      >
        <section
          v-if="hasSignalsTimeline"
          class="mb-3 w-full min-w-0"
          aria-label="Mentions by month"
          data-testid="node-detail-signals-timeline"
        >
          <h4 class="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">
            Mentions by month
          </h4>
          <SubjectTimelineChart
            :timeline="signalsTimeline"
            aria-label="Mentions by month for this subject"
          />
        </section>
        <NodeEnrichmentSection
          :node-id="props.nodeId ?? ''"
          :node-type="nodeType"
          @has-content="enrichmentHasContent = $event"
        />
      </div>

      <div
        v-if="props.embedInRail && hasPerspectivesTab"
        v-show="graphRailDetailTab === 'perspectives'"
        id="node-detail-rail-panel-perspectives"
        class="min-h-0"
        role="tabpanel"
        aria-labelledby="node-detail-rail-tab-perspectives"
        :tabindex="-1"
      >
        <NodeTopicPerspectives
          :corpus-path="perspectivesCorpusPath"
          :topic-id="perspectivesTopicId"
        />
      </div>

      <div
        v-if="!props.embedInRail || graphRailDetailTab === 'neighbourhood'"
        id="node-detail-rail-panel-neighbourhood"
        class="min-h-0"
        :role="props.embedInRail ? 'tabpanel' : undefined"
        :aria-labelledby="props.embedInRail ? 'node-detail-rail-tab-neighbourhood' : undefined"
        :tabindex="props.embedInRail ? -1 : undefined"
      >
        <p
          v-if="props.embedInRail && !graphConnectionsCenterInView"
          class="mb-2 text-[11px] leading-snug text-muted"
          data-testid="node-detail-rail-neighbourhood-unavailable"
        >
          This node is not in the current merged graph slice, so the Neighbourhood preview and
          connection list are hidden. Try another ego focus, clear filters, or load a graph that
          includes this node.
        </p>
        <GraphConnectionsSection
          class="mt-2"
          :view-artifact="props.viewArtifact"
          :node-id="props.nodeId"
          :dense-neighbor-list="!props.embedInRail"
          @go-graph="emit('go-graph')"
          @open-library-episode="emit('open-library-episode', $event)"
          @prefill-semantic-search="emit('prefill-semantic-search', $event)"
        />
      </div>
    </div>

    <TranscriptViewerDialog ref="transcriptViewerRef" />
  </aside>
</template>

<style scoped>
/* Full primary label in the header: grow vertically (no line clamp); separate panel removed. */
.node-detail-primary-title {
  overflow-wrap: anywhere;
  word-break: break-word;
  white-space: pre-wrap;
}
</style>
