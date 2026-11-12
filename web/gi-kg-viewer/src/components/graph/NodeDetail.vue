<script setup lang="ts">
import { computed, ref, watch } from "vue"
import type { ParsedArtifact } from "../../types/artifact"
import { useGraphFilterStore } from "../../stores/graphFilters"
import { useGraphHandoffStore } from "../../stores/graphHandoff"
import { useGraphNavigationStore } from "../../stores/graphNavigation"
import { useGraphAnalyticsStore } from "../../stores/graphAnalytics"
import { useShellStore } from "../../stores/shell"
import { graphNodeTypeChrome } from "../../utils/colors"
import { truncate } from "../../utils/formatting"
import { quoteAttributionDisplayFromId } from "../../utils/parsing"
import { nodeDetailExtensions, type NodeDetailContext } from "../../extensions"
import {
  findRawNodeInArtifact,
  findRawNodeInArtifactByIdOrPrefixed,
  fullPrimaryNodeLabel,
} from "../../utils/parsing"
import { copyTextToClipboard } from "../../utils/clipboard"
import { graphTypeAvatarLetter } from "../../utils/graphTypeAvatar"
import { stripLayerPrefixesForCil } from "../../utils/mergeGiKg"
import HelpTip from "../shared/HelpTip.vue"
import {
  SEARCH_RESULT_COPY_TITLE_CHIP_CLASS,
  SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS,
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
} from "../../utils/searchResultActionStyles"
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
import SubjectTimelineChart from "../subject/SubjectTimelineChart.vue"
import { buildSubjectMentionsTimeline } from "../../utils/subjectMentionsTimeline"
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
  /** Offer "View in graph" on a topic — only meaningful when the Graph tab is not showing. */
  showViewInGraph?: boolean
}>()

const shell = useShellStore()
const graphNav = useGraphNavigationStore()
const graphAnalytics = useGraphAnalyticsStore()
const graphFilters = useGraphFilterStore()
const graphHandoff = useGraphHandoffStore()

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

const PERSON_ENTITY_SEMANTIC_SEARCH_MAX_CHARS = 240

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
  (): string | null => {
    if (node.value) return null
    const bare = stripLayerPrefixesForCil(props.nodeId ?? "")
    if (bare.startsWith("person:")) return "Person"
    if (bare.startsWith("org:")) return "Organization"
    // #2057: without this an `object:` id fell through to null and rendered the empty
    // "Node" shell instead of the entity view.
    if (bare.startsWith("object:")) return "Object"
    if (bare.startsWith("topic:")) return "Topic"
    if (bare.startsWith("podcast:")) return "Podcast"
    for (const e of nodeDetailExtensions.list()) {
      const k = e.kindOf?.(props.nodeId ?? "")
      if (k) return k
    }
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

/** This node as extensions see it (extension point `nodeDetailExtensions`). */
const extensionContext = computed<NodeDetailContext>(() => ({
  nodeId: props.nodeId?.trim() ?? "",
  nodeType: nodeType.value,
  isTopicNode: isTopicNode.value,
  viewArtifact: props.viewArtifact,
  embedInRail: Boolean(props.embedInRail),
  node: node.value,
  fullArtifact: fullMergedArtifactForMetadata.value,
  goGraph: () => emit("go-graph"),
  focusOnGraph: focusNodeOnGraph,
  prefillSearch: (query: string) => {
    const q = query.trim()
    if (q) emit("prefill-semantic-search", { query: q })
  },
  openSearchFilters: (filters) => emit("open-search-insight-filters", filters),
  openTranscript: (payload) => transcriptViewerRef.value?.open(payload),
}))
const extensionKnowsNode = computed(() => {
  const id = props.nodeId?.trim()
  return Boolean(id && nodeDetailExtensions.list().some((e) => e.knowsNode?.(id)))
})
function extensionSections(region: "header" | "overview" | "body" | "after-aliases") {
  return nodeDetailExtensions.list().flatMap((e) => (e.sections ?? []).filter((x) => x.region === region))
}
/** An extension's neighbour list for this node (with the minimap neighbourhood). */
const extensionConnections = computed(() => {
  for (const e of nodeDetailExtensions.list()) {
    const c = e.connections?.(extensionContext.value)
    if (c) return c
  }
  return null
})

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

/**
 * Full quote/insight text in the rail — not ``nodeLabel`` (that caps at ~40 chars for on-canvas labels).
 */
const displayName = computed(() => {
  for (const e of nodeDetailExtensions.list()) {
    const t = e.title?.(extensionContext.value)
    if (t) return t
  }
  const n = node.value
  if (!n) {
    // Out-of-slice node (e.g. a co-speaker from the full relational graph):
    // derive a readable name from the id slug so the header isn't blank while
    // the person / topic view loads from server endpoints.
    const bare = stripLayerPrefixesForCil(props.nodeId ?? "")
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

/** A node handed to the graph from this panel (an extension's neighbour rows use it through `ctx`). */
function focusNodeOnGraph(nodeId: string): void {
  const id = nodeId.trim()
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

const corpusPathForCovers = computed(() =>
  (shell.resolvedCorpusPath ?? shell.corpusPath ?? "").trim()
)

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
    if (inferred === "Topic" || inferred === "Object") return "Topic"
    // A kind an extension named (`kindOf`) is its own visual group.
    return inferred
  }
  return visualGroupForNode(node.value)
})

const extensionAvatarGroup = computed((): string | null => {
  for (const e of nodeDetailExtensions.list()) {
    const g = e.avatarGroup?.(extensionContext.value)
    if (g) return g
  }
  return null
})
const avatarVisualGroup = computed(() => extensionAvatarGroup.value ?? visualType.value)

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

function humanizePropertyKey(k: string): string {
  return k.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase())
}

/** Keys an extension's section shows itself (extension point `nodeDetailExtensions`). */
const extensionHiddenProperties = computed(
  () => new Set(nodeDetailExtensions.list().flatMap((e) => e.hiddenProperties?.(extensionContext.value) ?? [])),
)

const extraProps = computed(() => {
  const p = node.value?.properties
  if (!p || typeof p !== "object") return []
  const out: { k: string; v: string }[] = []
  for (const k of Object.keys(p).sort()) {
    if (HIDDEN_PROPS.has(k)) continue
    if (TRANSCRIPT_ANCHOR_PROP_KEYS.has(k)) continue
    if (isTopicNode.value && k === "aliases") continue
    if (isPersonEntityRailNode.value && k === "aliases") continue
    if (extensionHiddenProperties.value.has(k)) continue
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

type GraphRailDetailTab =
  | "details"
  | "enrichment"
  | "neighbourhood"
  | `ext:${string}`

const graphRailDetailTab = ref<GraphRailDetailTab>("details")

// Tabs extensions add (`NodeDetailExtension.tabs`); the id is prefixed so it never collides
// with the panel's own. Fall back to Details when the current one stops showing.
const extensionTabs = computed(() =>
  nodeDetailExtensions
    .list()
    .flatMap((e) => e.tabs ?? [])
    .filter((t) => t.shows(extensionContext.value))
    .map((t) => ({ ...t, key: `ext:${t.id}` as const }))
)
watch(extensionTabs, (tabs) => {
  const current = graphRailDetailTab.value
  if (current.startsWith("ext:") && !tabs.some((t) => t.key === current)) {
    graphRailDetailTab.value = "details"
  }
})

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

/** Same gate as ``GraphConnectionsSection`` ``centerInView`` (minimap + list hidden when false). */
const graphConnectionsCenterInView = computed((): boolean => {
  const a = props.viewArtifact
  const id = props.nodeId?.trim()
  return Boolean(a && id && findRawNodeInArtifact(a, id))
})
</script>

<template>
  <aside
    v-if="nodeId && (node || extensionKnowsNode || inferredKindFromId)"
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
          class="flex h-[4.5rem] w-[4.5rem] shrink-0 items-center justify-center overflow-hidden rounded-2xl font-black leading-none shadow-md ring-1 ring-black/15 dark:ring-white/15"
          :class="extensionAvatarGroup ? 'text-xl tracking-tight' : 'text-2xl'"
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
          <component
            :is="section.component"
            v-for="(section, i) in extensionSections('header')"
            :key="`header-${i}`"
            :ctx="extensionContext"
          />
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
        v-for="tab in extensionTabs"
        :id="`node-detail-rail-tab-${tab.key}`"
        :key="tab.key"
        type="button"
        role="tab"
        class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
        :class="
          graphRailDetailTab === tab.key
            ? 'bg-primary text-primary-foreground'
            : 'text-elevated-foreground hover:bg-overlay'
        "
        :aria-selected="graphRailDetailTab === tab.key"
        :aria-controls="`node-detail-rail-panel-${tab.key}`"
        :data-testid="`node-detail-rail-tab-${tab.key}`"
        :tabindex="graphRailDetailTab === tab.key ? 0 : -1"
        @click="graphRailDetailTab = tab.key"
      >
        {{ tab.label }}
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
        v-show="!props.embedInRail || graphRailDetailTab === 'details'"
        id="node-detail-rail-panel-details"
        class="min-h-0"
        :role="props.embedInRail ? 'tabpanel' : undefined"
        :aria-labelledby="props.embedInRail ? 'node-detail-rail-tab-details' : undefined"
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
          <component
            :is="section.component"
            v-for="(section, i) in extensionSections('overview')"
            :key="`overview-${i}`"
            :ctx="extensionContext"
          />
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

          <component
            :is="section.component"
            v-for="(section, i) in extensionSections('body')"
            :key="`body-${i}`"
            :ctx="extensionContext"
          />

          <p
            v-if="topicAliasesLine"
            class="mb-3 text-[10px] leading-snug text-muted"
            data-testid="node-detail-topic-aliases"
          >
            <span class="font-medium text-surface-foreground/80">Aliases:</span>
            {{ topicAliasesLine }}
          </p>
          <component
            :is="section.component"
            v-for="(section, i) in extensionSections('after-aliases')"
            :key="`after-aliases-${i}`"
            :ctx="extensionContext"
          />

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

          <!-- Person overview folds in directly too — PersonLandingView's profile
           renders inline instead of a separate rail. -->
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

        </div>

        <div v-show="!props.embedInRail || graphRailDetailTab === 'details'" class="min-h-0">
          <p v-if="bodyTextForTemplate" class="mb-3 text-xs leading-relaxed text-muted">
            {{ truncate(bodyTextForTemplate, 600) }}
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
        v-for="tab in extensionTabs"
        v-show="graphRailDetailTab === tab.key"
        :id="`node-detail-rail-panel-${tab.key}`"
        :key="tab.key"
        class="min-h-0"
        role="tabpanel"
        :aria-labelledby="`node-detail-rail-tab-${tab.key}`"
        :tabindex="-1"
      >
        <component :is="tab.component" :ctx="extensionContext" />
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
          :aggregated-neighbor-rows="extensionConnections?.rows"
          :connections-empty-hint="extensionConnections?.emptyHint"
          :compound-neighborhood="extensionConnections?.compound ?? null"
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
