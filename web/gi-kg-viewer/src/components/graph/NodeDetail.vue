<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { BridgeDocument } from '../../types/bridge'
import type { ParsedArtifact } from '../../types/artifact'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useShellStore } from '../../stores/shell'
import { graphNodeTypeChrome } from '../../utils/colors'
import { truncate } from '../../utils/formatting'
import {
  countPersonEntityIncidentEdges,
  findRawNodeInArtifact,
  fullPrimaryNodeLabel,
  insightProvenanceLine,
  insightRelatedTopicRows,
  insightSupportingQuoteRows,
  insightSupportingTranscriptAggregate,
} from '../../utils/parsing'
import { graphTypeAvatarLetter } from '../../utils/graphTypeAvatar'
import HelpTip from '../shared/HelpTip.vue'
import {
  SEARCH_RESULT_COPY_TITLE_CHIP_CLASS,
  SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS,
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
  SEARCH_RESULT_GRAPH_BUTTON_CLASS,
} from '../../utils/searchResultActionStyles'
import {
  bridgeIdentityForGraphNodeId,
  crossLayerPresenceLabel,
} from '../../utils/bridgeDocument'
import { visualGroupForNode } from '../../utils/visualGroup'
import {
  corpusTextFileViewUrl,
  formatAudioTimingRange,
  formatTranscriptCharRange,
  GI_QUOTE_SPEAKER_UNAVAILABLE_HINT,
  resolveGiPathForTranscript,
  resolveTranscriptCorpusRelpath,
} from '../../utils/transcriptSourceDisplay'
import type { TopicClustersCluster } from '../../api/corpusTopicClustersApi'
import { fetchResolveEpisodeArtifacts } from '../../api/corpusLibraryApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { graphNeighborsForMemberGraphIds } from '../../utils/graphNeighbors'
import {
  findClusterByCompoundId,
  findTopicClusterContextForGraphNode,
  clusterTimelineCilTopicIdsForCluster,
  topicClusterMemberRowsForDetail,
} from '../../utils/topicClustersOverlay'
import GraphConnectionsSection from './GraphConnectionsSection.vue'
import TopicTimelineDialog from '../shared/TopicTimelineDialog.vue'
import TranscriptViewerDialog from '../shared/TranscriptViewerDialog.vue'
import {
  artifactRelPathsForResolvedRow,
  clusterSiblingEpisodeCap,
  episodeIdsForClusterMember,
  episodeIdsFromParsedArtifacts,
  sortResolvedArtifactsNewestFirst,
} from '../../utils/clusterSiblingMerge'

const emit = defineEmits<{
  close: []
  'go-graph': []
  'prefill-semantic-search': [{ query: string }]
  'open-explore-topic-filter': [{ topic: string }]
  'open-explore-speaker-filter': [{ speaker: string }]
  'open-explore-insight-filters': [{ groundedOnly: boolean; minConfidence: number | null }]
  'open-library-episode': [{ metadata_relative_path: string }]
}>()

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  nodeId: string | null
  /** Embedded in App right rail: full width, no fixed 280px strip. */
  embedInRail?: boolean
  /** RFC-072 bridge.json (optional) for cross-layer diagnostics. */
  bridgeDocument?: BridgeDocument | null
}>()

const shell = useShellStore()
const graphNav = useGraphNavigationStore()
const artifacts = useArtifactsStore()

const transcriptViewerRef = ref<InstanceType<typeof TranscriptViewerDialog> | null>(null)
const topicTimelineRef = ref<InstanceType<typeof TopicTimelineDialog> | null>(null)
/** Per-row catalog load for cluster members missing from the merge. */
const clusterMemberLoadBusyTopicId = ref<string | null>(null)
const clusterMemberLoadMessage = ref<string | null>(null)

/** Shown in **Where this appears**; omit from the generic property list. */
const TRANSCRIPT_ANCHOR_PROP_KEYS = new Set([
  'transcript_ref',
  'char_start',
  'char_end',
  'timestamp_start_ms',
  'timestamp_end_ms',
  'speaker_id',
])

const HIDDEN_PROPS = new Set([
  'slug',
  'label',
  'name',
  'text',
  'title',
  'description',
  'entity_kind',
  'kind',
])

/** Shown in insight meta strip; omit from generic property list. */
const INSIGHT_META_DL_KEYS = new Set(['grounded', 'insight_type', 'position_hint'])

const INSIGHT_SEMANTIC_SEARCH_MAX_CHARS = 240

const PERSON_ENTITY_SEMANTIC_SEARCH_MAX_CHARS = 240

const INSIGHT_SUPPORTING_QUOTES_COLLAPSE_AFTER = 5

/** Text-style HelpTip trigger under Full insight (not the round diagnostics chip). */
const INSIGHT_GROUNDING_TIP_TRIGGER_CLASS =
  'inline-flex max-w-full min-h-[1.25rem] items-center rounded px-0.5 py-0 text-left ' +
  'text-[10px] font-semibold leading-snug text-muted underline decoration-muted/40 ' +
  'underline-offset-2 hover:bg-overlay/50 hover:text-surface-foreground'

const graphFocusNeighborTooltip =
  'Show on graph — focus this node in the loaded merged graph (same as semantic search G).'

const node = computed(() => {
  const art = props.viewArtifact
  const id = props.nodeId
  if (!art || id == null) return null
  return findRawNodeInArtifact(art, id)
})

const nodeType = computed(() => {
  const n = node.value
  if (!n) return '?'
  return String(n.type ?? '?')
})

const isQuoteNode = computed(() => nodeType.value.trim().toLowerCase() === 'quote')

const isTopicNode = computed(() => nodeType.value.trim().toLowerCase() === 'topic')

const isInsightNode = computed(() => nodeType.value.trim().toLowerCase() === 'insight')

/** GI ``Person`` / legacy ``Speaker`` / KG ``Entity`` (person or organization). */
const isPersonEntityRailNode = computed(() => {
  const t = nodeType.value.trim().toLowerCase()
  return t === 'person' || t === 'entity' || t === 'speaker'
})

const isTopicClusterNode = computed(
  () => nodeType.value.trim().toLowerCase() === 'topiccluster',
)

/**
 * Cluster row from ``topic_clusters.json`` whether the user selected the **compound** or a
 * **member Topic** (same panel: TC chrome, members, merged connections).
 */
const topicClusterDocEntry = computed((): TopicClustersCluster | null => {
  const doc = artifacts.topicClustersDoc
  const id = props.nodeId?.trim()
  if (!doc || !id) {
    return null
  }
  const direct = findClusterByCompoundId(doc, id)
  if (direct) {
    return direct
  }
  const ctx = findTopicClusterContextForGraphNode(id, doc)
  const parent = ctx?.compoundParentId?.trim()
  if (!parent) {
    return null
  }
  return findClusterByCompoundId(doc, parent)
})

/** Corpus TopicCluster compound id (``tc:…``) for collapse/minimap; null if not in a cluster. */
const topicClusterCompoundId = computed((): string | null => {
  const id = props.nodeId?.trim()
  if (!id) {
    return null
  }
  if (isTopicClusterNode.value) {
    return id
  }
  return findTopicClusterContextForGraphNode(id, artifacts.topicClustersDoc)?.compoundParentId?.trim() ?? null
})

const hasTopicClusterJson = computed(() => topicClusterDocEntry.value != null)

/**
 * Full merged graph (all loaded GI/KG) for topic-cluster corpus alignment. The ego slice in
 * ``viewArtifact`` can omit Topic children under a ``tc:…`` compound (parent is not an edge), so
 * member resolution and cluster timelines use the display artifact when available.
 */
const artifactForTopicClusterCorpusMatch = computed(
  () => artifacts.displayArtifact ?? props.viewArtifact,
)

const topicClusterMemberRows = computed(() =>
  topicClusterMemberRowsForDetail(artifactForTopicClusterCorpusMatch.value, topicClusterDocEntry.value),
)

/** CIL topic ids for merged cluster timeline (prefer Topic children under compound, then members). */
const clusterTimelineTopicIds = computed((): string[] =>
  clusterTimelineCilTopicIdsForCluster(
    artifactForTopicClusterCorpusMatch.value,
    topicClusterCompoundId.value,
    topicClusterMemberRows.value,
  ),
)

const clusterTimelineUnavailable = computed(
  () =>
    (hasTopicClusterJson.value || isTopicClusterNode.value) &&
    clusterTimelineTopicIds.value.length === 0,
)

const topicClusterMembersMissingFromLoadedGraph = computed(() =>
  topicClusterMemberRows.value.some((r) => !r.graphNodeId),
)

const topicClusterMemberGraphIds = computed((): string[] =>
  topicClusterMemberRows.value
    .map((r) => r.graphNodeId)
    .filter((x): x is string => x != null),
)

const topicClusterAggregatedNeighbors = computed(() =>
  graphNeighborsForMemberGraphIds(props.viewArtifact, topicClusterMemberGraphIds.value),
)

/** Merged edges + cluster minimap when JSON lists members with graph ids in this view. */
const useTopicClusterAggregatedConnections = computed(
  () =>
    (hasTopicClusterJson.value || isTopicClusterNode.value) &&
    topicClusterMemberGraphIds.value.length > 0,
)

const topicClusterNeighborhoodForMap = computed((): {
  compoundId: string
  memberIds: string[]
} | undefined => {
  if (!useTopicClusterAggregatedConnections.value) {
    return undefined
  }
  const cid = topicClusterCompoundId.value?.trim()
  if (!cid) {
    return undefined
  }
  return { compoundId: cid, memberIds: topicClusterMemberGraphIds.value }
})

/** Always the corpus compound id when collapsing member topics on canvas (RFC-075). */
const topicClusterCollapseCyId = computed((): string => {
  const c = topicClusterCompoundId.value?.trim()
  if (c) {
    return c
  }
  return props.nodeId?.trim() ?? ''
})

const TOPIC_CLUSTER_CONNECTIONS_EMPTY =
  'No edges from member topics in this graph view (adjust filters or ego view).'

/**
 * Full quote/insight text in the rail — not ``nodeLabel`` (that caps at ~40 chars for on-canvas labels).
 * For RFC-075 clusters with JSON, the header matches the **cluster** (same for compound and member).
 */
const displayName = computed(() => {
  const cl = topicClusterDocEntry.value
  const art = props.viewArtifact
  if (cl && art) {
    const raw = cl.canonical_label
    if (typeof raw === 'string' && raw.trim()) {
      return raw.trim()
    }
    const cid = topicClusterCompoundId.value?.trim()
    if (cid) {
      const pn = findRawNodeInArtifact(art, cid)
      if (pn) {
        return fullPrimaryNodeLabel(pn)
      }
    }
  }
  const n = node.value
  if (!n) {
    return ''
  }
  return fullPrimaryNodeLabel(n)
})

function focusTopicClusterMember(graphNodeId: string): void {
  const id = graphNodeId.trim()
  if (!id) {
    return
  }
  graphNav.requestFocusNode(id)
  emit('go-graph')
}

const canLoadClusterMembersFromCatalog = computed(
  () =>
    Boolean(shell.healthStatus) &&
    shell.corpusLibraryApiAvailable !== false &&
    shell.hasCorpusPath,
)

function clusterMemberEpisodeIdsListed(topicId: string): string[] {
  return episodeIdsForClusterMember(topicClusterDocEntry.value, topicId)
}

async function loadClusterMemberEpisodes(topicId: string): Promise<void> {
  const tid = topicId.trim()
  clusterMemberLoadMessage.value = null
  const root = (shell.resolvedCorpusPath ?? shell.corpusPath).trim()
  if (!root) {
    clusterMemberLoadMessage.value = 'Set a corpus path first.'
    return
  }
  if (!shell.healthStatus || shell.corpusLibraryApiAvailable === false) {
    clusterMemberLoadMessage.value = 'Corpus catalog API is not available.'
    return
  }
  const cl = topicClusterDocEntry.value
  if (!cl) {
    return
  }
  const fromJson = episodeIdsForClusterMember(cl, tid)
  if (fromJson.length === 0) {
    clusterMemberLoadMessage.value =
      'No episode_ids for this member in topic_clusters.json — add them to load from the catalog.'
    return
  }
  const loaded = episodeIdsFromParsedArtifacts(artifacts.parsedList)
  const candidateIds = fromJson.filter((id) => !loaded.has(id))
  if (candidateIds.length === 0) {
    clusterMemberLoadMessage.value = 'Those episodes are already in the graph selection.'
    return
  }

  clusterMemberLoadBusyTopicId.value = tid
  try {
    const res = await fetchResolveEpisodeArtifacts(root, candidateIds)
    const sorted = sortResolvedArtifactsNewestFirst(res.resolved)
    const cap = clusterSiblingEpisodeCap()
    const selected = new Set(artifacts.selectedRelPaths.map((p) => p.replace(/\\/g, '/')))
    const pathsToAdd: string[] = []
    let addedEpisodes = 0
    for (const row of sorted) {
      if (addedEpisodes >= cap) {
        break
      }
      const rels = artifactRelPathsForResolvedRow(row)
      if (rels.length === 0) {
        continue
      }
      let anyNew = false
      for (const rel of rels) {
        const norm = rel.replace(/\\/g, '/')
        if (!selected.has(norm)) {
          anyNew = true
          break
        }
      }
      if (!anyNew) {
        continue
      }
      for (const rel of rels) {
        const norm = rel.replace(/\\/g, '/')
        if (!selected.has(norm)) {
          selected.add(norm)
          pathsToAdd.push(rel)
        }
      }
      addedEpisodes += 1
    }
    const z = res.missing_episode_ids.length
    if (pathsToAdd.length === 0) {
      clusterMemberLoadMessage.value =
        z > 0
          ? `Catalog had no GI paths for ${z} episode id(s).`
          : 'No new artifact paths to add.'
      return
    }
    await artifacts.appendRelativeArtifacts(pathsToAdd)
    clusterMemberLoadMessage.value =
      `Loaded ${addedEpisodes} episode(s) (cap ${cap}).` + (z > 0 ? ` ${z} not in catalog.` : '')
  } catch (e) {
    clusterMemberLoadMessage.value = e instanceof Error ? e.message : String(e)
  } finally {
    clusterMemberLoadBusyTopicId.value = null
  }
}

/** Full passage for copy/paste; header may still clamp visually. */
const quoteFullPassage = computed((): string | null => {
  if (!isQuoteNode.value) return null
  const full = displayName.value.trim()
  return full || null
})

/** Long primary label (Topic uses ``properties.label``; same as header source, uncapped). */
const fullRailTextContent = computed((): string | null => {
  if (isQuoteNode.value) return quoteFullPassage.value
  if (
    isTopicNode.value ||
    isInsightNode.value ||
    isPersonEntityRailNode.value
  ) {
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
    .filter((x): x is string => typeof x === 'string' && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(', ') : null
})

/** Topic label for Search / Explore handoff (same primary string as full block). */
const topicGatewayQuery = computed((): string => {
  if (!isTopicNode.value) return ''
  return displayName.value.trim()
})

function emitTopicPrefillSearch(): void {
  const q = topicGatewayQuery.value
  if (!q) return
  emit('prefill-semantic-search', { query: q })
}

function emitTopicExploreFilter(): void {
  const t = topicGatewayQuery.value
  if (!t) return
  emit('open-explore-topic-filter', { topic: t })
}

const personEntityGatewayQuery = computed((): string => {
  if (!isPersonEntityRailNode.value) return ''
  return displayName.value.trim()
})

const personEntitySemanticSearchQuery = computed((): string => {
  if (!isPersonEntityRailNode.value) return ''
  return truncate(
    displayName.value.trim(),
    PERSON_ENTITY_SEMANTIC_SEARCH_MAX_CHARS,
  )
})

const personEntityAliasesLine = computed((): string | null => {
  if (!isPersonEntityRailNode.value) return null
  const a = node.value?.properties?.aliases
  if (!Array.isArray(a) || a.length === 0) return null
  const parts = a
    .filter((x): x is string => typeof x === 'string' && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(', ') : null
})

const personEntityEdgeCounts = computed(() =>
  countPersonEntityIncidentEdges(props.viewArtifact, props.nodeId),
)

const personEntityRoleSummaryLine = computed((): string | null => {
  if (!isPersonEntityRailNode.value) return null
  const { spokenByQuotes, spokeInEpisodes } = personEntityEdgeCounts.value
  const parts: string[] = []
  if (spokenByQuotes > 0) {
    parts.push(
      `${spokenByQuotes} attributed quote${spokenByQuotes === 1 ? '' : 's'}`,
    )
  }
  if (spokeInEpisodes > 0) {
    parts.push(
      `${spokeInEpisodes} episode link${spokeInEpisodes === 1 ? '' : 's'}`,
    )
  }
  if (parts.length === 0) return null
  return `In this graph: ${parts.join(' · ')}.`
})

function emitPersonEntityPrefillSearch(): void {
  const q = personEntitySemanticSearchQuery.value.trim()
  if (!q) return
  emit('prefill-semantic-search', { query: q })
}

function emitPersonEntityExploreHandoff(): void {
  const q = personEntityGatewayQuery.value.trim()
  if (!q) return
  if (visualType.value !== 'Entity_organization') {
    emit('open-explore-speaker-filter', { speaker: q })
  } else {
    emit('open-explore-topic-filter', { topic: q })
  }
}

/** Type / position / confidence only (grounding + provenance live in the details HelpTip). */
const insightSecondaryDetailLines = computed((): string[] => {
  if (!isInsightNode.value) return []
  const parts: string[] = []
  const p = node.value?.properties
  const it = p?.insight_type
  if (typeof it === 'string' && it.trim()) {
    parts.push(`Type: ${it.trim()}`)
  }
  const ph = p?.position_hint
  if (typeof ph === 'number' && Number.isFinite(ph)) {
    const pct = Math.round(Math.max(0, Math.min(1, ph)) * 100)
    parts.push(`Position in episode: ~${pct}%`)
  }
  const rawNode = node.value as Record<string, unknown> | null
  const cn = rawNode?.confidence
  if (typeof cn === 'number' && Number.isFinite(cn)) {
    parts.push(`Confidence: ${cn.toFixed(2)}`)
  }
  return parts
})

const insightGroundedBoolean = computed((): boolean | null => {
  if (!isInsightNode.value) return null
  const g = node.value?.properties?.grounded
  return typeof g === 'boolean' ? g : null
})

const insightSemanticSearchQuery = computed((): string => {
  if (!isInsightNode.value) return ''
  return truncate(displayName.value.trim(), INSIGHT_SEMANTIC_SEARCH_MAX_CHARS)
})

const insightSupportingQuotes = computed(() =>
  isInsightNode.value ? insightSupportingQuoteRows(props.viewArtifact, props.nodeId) : [],
)

const insightSupportingTranscriptAgg = computed(() =>
  isInsightNode.value
    ? insightSupportingTranscriptAggregate(props.viewArtifact, props.nodeId)
    : null,
)

const insightOpenAllSupportingQuotesReady = computed((): boolean => {
  const agg = insightSupportingTranscriptAgg.value
  if (!agg) {
    return false
  }
  const root = shell.corpusPath.trim()
  if (!shell.healthStatus || !root) {
    return false
  }
  const giPath = resolveGiPathForTranscript(props.viewArtifact, agg.episodeId)
  const resolvedRelpath = resolveTranscriptCorpusRelpath(agg.transcriptRef, giPath)
  return Boolean(
    resolvedRelpath && corpusTextFileViewUrl(root, resolvedRelpath),
  )
})

const insightQuotesExpanded = ref(false)

watch(
  () => props.nodeId,
  () => {
    insightQuotesExpanded.value = false
    clusterMemberLoadMessage.value = null
  },
)

const insightSupportingQuotesVisible = computed(() => {
  const all = insightSupportingQuotes.value
  if (
    insightQuotesExpanded.value ||
    all.length <= INSIGHT_SUPPORTING_QUOTES_COLLAPSE_AFTER
  ) {
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
    : [],
)

const insightProvenanceText = computed(() =>
  isInsightNode.value ? insightProvenanceLine(props.viewArtifact) : null,
)

const hasInsightDetailsTipContent = computed((): boolean => {
  if (!isInsightNode.value) return false
  if (insightGroundedBoolean.value !== null) return true
  if (insightSecondaryDetailLines.value.length > 0) return true
  return Boolean(insightProvenanceText.value?.trim())
})

/** Visible trigger and accessible name (popover holds the full grounding explainer). */
const insightGroundingTriggerLabel = computed((): string => {
  if (insightGroundedBoolean.value === true) return 'Grounded'
  if (insightGroundedBoolean.value === false) return 'Not grounded'
  return 'Extraction details'
})

const insightExploreFiltersPayload = computed((): {
  groundedOnly: boolean
  minConfidence: number | null
} | null => {
  if (!isInsightNode.value) return null
  const p = node.value?.properties
  const g = p?.grounded
  const groundedOnly = typeof g === 'boolean' ? g : false
  const rawNode = node.value as Record<string, unknown> | null
  const c = rawNode?.confidence
  const minConfidence = typeof c === 'number' && Number.isFinite(c) ? c : null
  return { groundedOnly, minConfidence }
})

function emitInsightPrefillSearch(): void {
  const q = insightSemanticSearchQuery.value.trim()
  if (!q) return
  emit('prefill-semantic-search', { query: q })
}

function emitInsightOpenExploreFilters(): void {
  const p = insightExploreFiltersPayload.value
  if (!p) return
  emit('open-explore-insight-filters', p)
}

function focusNeighborOnGraph(nbId: string, ev: MouseEvent): void {
  ev.stopPropagation()
  const id = nbId.trim()
  if (!id) return
  graphNav.requestFocusNode(id)
  emit('go-graph')
}

/** Shared gate for CIL timeline API (single-topic GET or merged cluster POST). */
const cilTimelineApiUnavailable = computed(
  (): boolean =>
    !shell.healthStatus ||
    shell.cilQueriesApiAvailable === false ||
    !shell.hasCorpusPath,
)

const topicTimelineDisabled = computed((): boolean => {
  if (!isTopicNode.value) return true
  const id = props.nodeId
  if (id == null || !String(id).trim()) return true
  return cilTimelineApiUnavailable.value
})

/** Merged CIL timeline for TopicCluster (member topic ids from JSON). */
const clusterTimelineDisabled = computed((): boolean => {
  if (clusterTimelineTopicIds.value.length === 0) return true
  return cilTimelineApiUnavailable.value
})

/** Single-topic rail button hidden when JSON-backed cluster uses merged timeline in-cluster. */
const showSingleTopicTimelineButton = computed(
  () => isTopicNode.value && props.nodeId && !hasTopicClusterJson.value,
)

function openTopicTimeline(): void {
  const id = props.nodeId
  if (id == null) return
  void topicTimelineRef.value?.open(String(id), {
    variant: 'entity',
    entityLabel: nodeType.value,
  })
}

function openClusterTimeline(): void {
  void topicTimelineRef.value?.openCluster(clusterTimelineTopicIds.value)
}

/** One member row: single-topic CIL timeline for that topic id (not merged cluster). */
function openClusterMemberTopicTimeline(topicId: string): void {
  const id = String(topicId).trim()
  if (!id) return
  void topicTimelineRef.value?.open(id, {
    variant: 'entity',
    entityLabel: 'Topic',
  })
}

type RailFullTextCopyUi = 'idle' | 'copied' | 'failed'

const railFullTextCopyUi = ref<RailFullTextCopyUi>('idle')
let railFullTextCopyResetTimer: ReturnType<typeof setTimeout> | null = null

function resetRailFullTextCopyUi(): void {
  if (railFullTextCopyResetTimer !== null) {
    clearTimeout(railFullTextCopyResetTimer)
    railFullTextCopyResetTimer = null
  }
  railFullTextCopyUi.value = 'idle'
}

watch(
  () => [props.nodeId, fullRailTextContent.value] as const,
  () => {
    resetRailFullTextCopyUi()
  },
)

async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    try {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      ta.style.left = '-9999px'
      document.body.appendChild(ta)
      ta.focus()
      ta.select()
      const ok = document.execCommand('copy')
      document.body.removeChild(ta)
      return ok
    } catch {
      return false
    }
  }
}

async function copyFullRailText(): Promise<void> {
  const text = fullRailTextContent.value
  if (!text) return
  resetRailFullTextCopyUi()
  const ok = await copyTextToClipboard(text)
  railFullTextCopyUi.value = ok ? 'copied' : 'failed'
  railFullTextCopyResetTimer = setTimeout(() => {
    railFullTextCopyUi.value = 'idle'
    railFullTextCopyResetTimer = null
  }, 2000)
}

const railFullTextCopyAriaLabel = computed((): string => {
  if (railFullTextCopyUi.value === 'copied') return 'Copied to clipboard'
  if (railFullTextCopyUi.value === 'failed') return 'Copy failed; try again'
  return 'Copy title'
})

/** Native tooltip on the **C** chip (matches ``aria-label``). */
const railFullTextCopyTitleTooltip = computed((): string => railFullTextCopyAriaLabel.value)

const railFullTextSectionTestId = computed((): string => {
  if (isQuoteNode.value) return 'node-detail-full-quote'
  if (isTopicNode.value) return 'node-detail-full-topic'
  if (isInsightNode.value) return 'node-detail-full-insight'
  if (isPersonEntityRailNode.value) return 'node-detail-full-person-entity'
  return 'node-detail-full-primary'
})

const railFullTextCopyTestId = computed((): string => {
  if (isQuoteNode.value) return 'node-detail-full-quote-copy'
  if (isTopicNode.value) return 'node-detail-full-topic-copy'
  if (isInsightNode.value) return 'node-detail-full-insight-copy'
  if (isPersonEntityRailNode.value) return 'node-detail-full-person-entity-copy'
  return 'node-detail-full-primary-copy'
})

const visualType = computed(() => visualGroupForNode(node.value))

/** TC glyph when the row is a TopicCluster node or a corpus cluster member (JSON-backed). */
const avatarVisualGroup = computed(() => {
  if (hasTopicClusterJson.value || isTopicClusterNode.value) {
    return 'TopicCluster'
  }
  return visualType.value
})

const personEntityExploreUsesSpeakerFilter = computed(
  () => visualType.value !== 'Entity_organization',
)

const personEntityExploreButtonLabel = computed((): string =>
  personEntityExploreUsesSpeakerFilter.value ? 'Speaker filter' : 'Topic filter',
)

const personEntityExploreHandoffAriaLabel = computed((): string =>
  personEntityExploreUsesSpeakerFilter.value
    ? 'Open Explore with Speaker contains filled from this name; topic filter cleared'
    : 'Open Explore with Topic contains filled from this organization name; speaker filter cleared',
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
  if (!id) return ''
  return (
    `Graph node id (Cytoscape / merged graph): ${id}. ` +
    'Same id as in artifact edges; use for support and tooling.'
  )
})

const entityKind = computed(() => {
  const p = node.value?.properties
  if (!p) return null
  const kd = typeof p.kind === 'string' ? p.kind.trim().toLowerCase() : ''
  if (kd === 'org') return 'organization'
  if (kd === 'person') return 'person'
  const ek = p.entity_kind
  return typeof ek === 'string' && ek.trim() ? ek.trim() : null
})

const bodyText = computed(() => {
  const n = node.value
  if (!n) return null
  const p = n.properties
  if (!p) return null
  const explicit = p.text ?? p.description
  if (typeof explicit === 'string' && explicit.trim()) return explicit.trim()
  const label = p.label ?? p.name ?? p.title
  if (typeof label === 'string' && label.length > displayName.value.length) {
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
  if (nt !== 'Episode' && ek.trim().toLowerCase() === 'episode') {
    return false
  }
  return true
})

/** Graph type + entity kind — top of **Details** body, not under the rail title. */
const showNodeKindRowInDetails = computed(
  () =>
    showEntityKindSubtitle.value ||
    (!props.embedInRail && Boolean(nodeType.value?.trim())) ||
    (props.embedInRail &&
      isPersonEntityRailNode.value &&
      Boolean(nodeType.value?.trim())),
)

const showNodeTypeChipInDetails = computed(
  () =>
    (!props.embedInRail && Boolean(nodeType.value?.trim())) ||
    (props.embedInRail && isPersonEntityRailNode.value && Boolean(nodeType.value?.trim())),
)

const transcriptSourceSection = computed(() => {
  const p = node.value?.properties
  if (!p || typeof p !== 'object') {
    return null
  }
  const ref = typeof p.transcript_ref === 'string' ? p.transcript_ref.trim() : ''
  const charLine = formatTranscriptCharRange(p.char_start, p.char_end)
  const timeLine = formatAudioTimingRange(p.timestamp_start_ms, p.timestamp_end_ms)
  const sp = p.speaker_id
  let speakerLine = ''
  if (typeof sp === 'string' && sp.trim()) {
    speakerLine = sp.trim()
  } else if (typeof sp === 'number' && Number.isFinite(sp)) {
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
    typeof rawEp === 'string' && rawEp.trim()
      ? rawEp.trim()
      : typeof rawEp === 'number' && Number.isFinite(rawEp)
        ? String(rawEp)
        : null
  const giPath = resolveGiPathForTranscript(props.viewArtifact, quoteEpId)
  const root = shell.corpusPath.trim()
  const resolvedRelpath = ref ? resolveTranscriptCorpusRelpath(ref, giPath) : ''
  const href =
    shell.healthStatus && root && resolvedRelpath
      ? corpusTextFileViewUrl(root, resolvedRelpath)
      : null
  const rawNt = node.value?.type
  const isQuoteKind =
    typeof rawNt === 'string' && rawNt.trim().toLowerCase() === 'quote'
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
  })
}

function openInsightAllSupportingQuotesTranscript(): void {
  const agg = insightSupportingTranscriptAgg.value
  const root = shell.corpusPath.trim()
  if (!agg || !root) {
    return
  }
  const giPath = resolveGiPathForTranscript(props.viewArtifact, agg.episodeId)
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
  return k
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase())
}

const extraProps = computed(() => {
  const p = node.value?.properties
  if (!p || typeof p !== 'object') return []
  const out: { k: string; v: string }[] = []
  for (const k of Object.keys(p).sort()) {
    if (HIDDEN_PROPS.has(k)) continue
    if (TRANSCRIPT_ANCHOR_PROP_KEYS.has(k)) continue
    if (isTopicNode.value && k === 'aliases') continue
    if (isPersonEntityRailNode.value && k === 'aliases') continue
    if (isInsightNode.value && INSIGHT_META_DL_KEYS.has(k)) continue
    // Corpus episode id is redundant with the **E** header chip on non-Episode nodes (Insight, Quote, …).
    if (nodeType.value !== 'Episode' && k === 'episode_id') continue
    let v: unknown = p[k]
    if (v === null || v === undefined || v === '') continue
    if (typeof v === 'boolean') {
      v = v ? 'yes' : 'no'
    } else if (typeof v === 'object') {
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
    { label: 'Graph node id', value: id },
    { label: 'Type', value: nodeType.value },
    { label: 'Visual group', value: visualType.value },
  ]
  if (entityKind.value) {
    rows.push({ label: 'Entity kind', value: entityKind.value })
  }
  return rows
})

const crossLayerBridgeLine = computed(() => {
  const row = bridgeIdentityForGraphNodeId(props.bridgeDocument ?? null, props.nodeId)
  if (!row) return ''
  return crossLayerPresenceLabel(row.sources)
})

/** RFC-075: corpus topic cluster label when this Topic is a member (API-loaded clusters only). */
const topicClusterContext = computed(() => {
  if (!isTopicNode.value || !props.nodeId) {
    return null
  }
  return findTopicClusterContextForGraphNode(props.nodeId, artifacts.topicClustersDoc)
})

type GraphRailDetailTab = 'details' | 'neighbourhood'

const graphRailDetailTab = ref<GraphRailDetailTab>('details')

watch(
  () => props.nodeId,
  () => {
    graphRailDetailTab.value = 'details'
  },
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
    v-if="nodeId && node"
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
      :class="props.embedInRail ? 'border-b border-border px-2 py-2' : 'border-b border-border px-3 py-2'"
    >
      <div class="flex min-w-0 flex-1 gap-3">
        <div
          class="flex h-[4.5rem] w-[4.5rem] shrink-0 items-center justify-center rounded-2xl font-black leading-none shadow-md ring-1 ring-black/15 dark:ring-white/15"
          :class="hasTopicClusterJson || isTopicClusterNode ? 'text-xl tracking-tight' : 'text-2xl'"
          :style="nodeTypeAvatarStyle"
          aria-hidden="true"
        >
          {{ avatarLetter }}
        </div>
        <div class="min-h-0 min-w-0 flex-1 basis-0">
          <div class="flex min-w-0 items-start justify-between gap-1">
            <h3 class="min-w-0 flex-1 basis-0">
              <span
                class="node-detail-primary-title select-text block w-full min-w-0 text-base font-semibold leading-snug text-surface-foreground"
                :title="primaryTitleNativeTooltip"
                :data-testid="fullRailTextContent ? railFullTextSectionTestId : undefined"
                tabindex="0"
              >{{ displayName }}</span>
            </h3>
            <div
              class="ml-1 flex shrink-0 flex-col items-end gap-0.5 self-start pt-0.5"
            >
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
          <p
            v-if="(hasTopicClusterJson || isTopicClusterNode) && !props.embedInRail"
            class="mt-1.5 text-[10px] leading-snug text-muted"
          >
            <span
              class="inline-block rounded bg-primary/12 px-1.5 py-0.5 font-semibold uppercase tracking-wide text-primary"
            >Topic cluster</span>
            <span class="ml-1.5">Corpus grouping from <span class="font-mono text-[9px]">topic_clusters.json</span>; members share one compound on the graph.</span>
          </p>
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
      class="px-2 py-2"
      :class="props.embedInRail ? 'min-h-0 flex-1 overflow-y-auto' : 'overflow-y-auto px-3'"
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
        <p
          v-if="showNodeKindRowInDetails"
          class="mb-2 text-xs text-muted"
          data-testid="node-detail-kind-row"
        >
          <span
            v-if="showNodeTypeChipInDetails"
            class="inline-block rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary"
          >{{ nodeType }}</span>
          <template v-if="showEntityKindSubtitle">
            <span v-if="showNodeTypeChipInDetails" class="mx-1 text-muted">·</span>
            <span>{{ entityKind }}</span>
          </template>
        </p>
      <template v-if="hasTopicClusterJson || isTopicClusterNode">
        <section
          class="mb-3 rounded border border-border bg-surface/40 p-2 text-[10px]"
          data-testid="node-detail-topic-cluster-members"
          aria-label="Topic cluster members"
        >
          <div class="flex flex-wrap items-center justify-between gap-2">
            <span class="font-semibold text-surface-foreground">Member topics</span>
            <div class="flex flex-wrap items-center gap-1.5">
              <button
                v-if="clusterTimelineTopicIds.length"
                type="button"
                class="rounded border border-border bg-canvas px-2 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-elevated disabled:opacity-40"
                data-testid="node-detail-cluster-timeline"
                :disabled="clusterTimelineDisabled"
                @click="openClusterTimeline"
              >
                Cluster timeline
              </button>
              <button
                v-if="topicClusterCollapseCyId"
                type="button"
                class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
                :aria-pressed="graphNav.isTopicClusterCanvasCollapsed(topicClusterCollapseCyId)"
                @click="graphNav.toggleTopicClusterCanvasCollapsed(topicClusterCollapseCyId)"
              >
                {{
                  graphNav.isTopicClusterCanvasCollapsed(topicClusterCollapseCyId)
                    ? 'Show topics on graph'
                    : 'Hide topics on graph'
                }}
              </button>
            </div>
          </div>
          <p class="mt-1 leading-snug text-muted">
            Connections below merge edges from every member topic in this cluster.
          </p>
          <p
            v-if="clusterTimelineUnavailable"
            class="mt-1.5 text-[10px] leading-snug text-warning"
            data-testid="node-detail-cluster-timeline-unavailable"
          >
            Cluster timeline is unavailable: no member topic ids were found. Add
            <span class="font-mono">members</span> in
            <span class="font-mono">topic_clusters.json</span>, or load a merged graph where Topic
            nodes are parented to this compound.
          </p>
          <p
            v-if="topicClusterMembersMissingFromLoadedGraph"
            class="mt-1.5 leading-snug text-warning"
          >
            Some members are not in the loaded graph: use per-row <strong>Load</strong> (when
            <span class="font-mono">episode_ids</span> are listed in
            <span class="font-mono">topic_clusters.json</span>), <strong>Load selected</strong> in the
            corpus list, or a broader corpus so those topic nodes exist in the merge.
          </p>
          <p
            v-if="clusterMemberLoadMessage"
            class="mt-1.5 text-[9px] leading-snug text-muted"
            data-testid="node-detail-cluster-member-load-message"
          >
            {{ clusterMemberLoadMessage }}
          </p>
          <ul
            v-if="topicClusterMemberRows.length"
            class="mt-2 max-h-40 space-y-1.5 overflow-y-auto"
          >
            <li
              v-for="(row, ri) in topicClusterMemberRows"
              :key="`${row.topicId}-${ri}`"
              class="flex items-start justify-between gap-2 rounded px-1 py-0.5 hover:bg-overlay/50"
            >
              <div class="min-w-0 flex-1">
                <span class="block font-medium leading-snug text-surface-foreground">{{
                  row.label
                }}</span>
                <code class="mt-0.5 block font-mono text-[9px] text-muted">{{ row.topicId }}</code>
                <span
                  v-if="!row.graphNodeId"
                  class="mt-0.5 block text-[9px] text-warning"
                >
                  Not in this graph view
                </span>
              </div>
              <div class="flex shrink-0 flex-col items-end gap-1 sm:flex-row sm:items-center">
                <button
                  v-if="row.topicId.trim()"
                  type="button"
                  class="rounded border border-border bg-canvas px-1.5 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-elevated disabled:opacity-40"
                  data-testid="node-detail-cluster-member-timeline"
                  :disabled="cilTimelineApiUnavailable"
                  @click="openClusterMemberTopicTimeline(row.topicId)"
                >
                  Timeline
                </button>
                <button
                  v-if="!row.graphNodeId && hasTopicClusterJson"
                  type="button"
                  class="rounded border border-border bg-gi/15 px-1.5 py-0.5 text-[10px] font-medium text-gi-foreground hover:bg-gi/25 disabled:opacity-40"
                  data-testid="node-detail-cluster-member-load"
                  :disabled="
                    !canLoadClusterMembersFromCatalog ||
                    artifacts.loading ||
                    clusterMemberLoadBusyTopicId === row.topicId ||
                    clusterMemberEpisodeIdsListed(row.topicId).length === 0
                  "
                  :title="
                    clusterMemberEpisodeIdsListed(row.topicId).length === 0
                      ? 'Add episode_ids for this member in topic_clusters.json'
                      : `Load up to ${clusterSiblingEpisodeCap()} episodes from the catalog (newest first)`
                  "
                  @click="loadClusterMemberEpisodes(row.topicId)"
                >
                  {{
                    clusterMemberLoadBusyTopicId === row.topicId ? 'Loading…' : 'Load'
                  }}
                </button>
                <button
                  v-if="row.graphNodeId"
                  type="button"
                  class="rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-medium text-primary"
                  @click="focusTopicClusterMember(row.graphNodeId)"
                >
                  Focus
                </button>
              </div>
            </li>
          </ul>
          <p
            v-else
            class="mt-2 text-[10px] text-muted"
          >
            No members listed in topic_clusters.json for this cluster.
          </p>
        </section>
      </template>

      <template v-if="isPersonEntityRailNode">
        <p
          v-if="personEntityRoleSummaryLine"
          class="mb-2 text-[10px] leading-snug text-muted"
          data-testid="node-detail-person-entity-role"
        >
          {{ personEntityRoleSummaryLine }}
        </p>
        <p
          v-if="personEntityAliasesLine"
          class="mb-3 text-[10px] leading-snug text-muted"
          data-testid="node-detail-person-entity-aliases"
        >
          <span class="font-medium text-surface-foreground/80">Aliases:</span>
          {{ personEntityAliasesLine }}
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
              <strong class="font-medium text-surface-foreground">Prefill semantic search</strong> uses this
              primary name as the query against the vector index.
              For <strong class="font-medium text-surface-foreground/90">people</strong> (GI
              <span class="font-mono">Person</span> / legacy <span class="font-mono">Speaker</span>, or KG
              <span class="font-mono">Entity</span> marked as a person),
              <strong class="font-medium text-surface-foreground">Speaker filter</strong> fills
              <strong class="font-medium text-surface-foreground/90">Speaker contains</strong> and clears the topic
              filter. For <strong class="font-medium text-surface-foreground/90">organizations</strong> (KG
              <span class="font-mono">Entity</span> with organization-style kind), the second button
              <strong class="font-medium text-surface-foreground">Topic filter</strong> fills
              <strong class="font-medium text-surface-foreground/90">Topic contains</strong> (same control as the
              topic graph row) and clears the speaker filter. You still run
              <strong class="font-medium text-surface-foreground/90">Run explore</strong>.
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
              <p class="mb-1 font-semibold text-surface-foreground">
                Grounding
              </p>
              <p>
                <strong class="text-surface-foreground/90">Grounded</strong> means this insight has at least one
                supporting quote in this episode graph (linked via SUPPORTED_BY in the GI file).
                <strong class="text-surface-foreground/90">Not grounded</strong> means the artifact marks
                grounded as false (no supporting quotes recorded for this insight).
              </p>
            </div>
            <div v-if="insightSecondaryDetailLines.length">
              <p class="mb-1 font-semibold text-surface-foreground">
                Other fields
              </p>
              <ul class="list-inside list-disc space-y-0.5">
                <li v-for="(line, li) in insightSecondaryDetailLines" :key="li">
                  {{ line }}
                </li>
              </ul>
            </div>
            <div v-if="insightProvenanceText">
              <p class="mb-1 font-semibold text-surface-foreground">
                Lineage
              </p>
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
          @click="emitInsightOpenExploreFilters"
        >
          Set Explore filters
        </button>
        <HelpTip
          class="shrink-0 self-center"
          :pref-width="280"
          button-aria-label="About insight Search and Explore shortcuts"
        >
          <p class="font-sans text-[10px] leading-snug text-muted">
            <strong class="font-medium text-surface-foreground">Prefill semantic search</strong> uses the
            first part of this insight as the query against the vector index (not GI-only Explore).
            <strong class="font-medium text-surface-foreground">Set Explore filters</strong> switches
            to Explore, clears topic/speaker filters, sets <strong class="font-medium text-surface-foreground/90">Grounded only</strong>
            and optional <strong class="font-medium text-surface-foreground/90">Min confidence</strong> from this node,
            then you run <strong class="font-medium text-surface-foreground/90">Run explore</strong>.
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
        <h4 class="text-xs font-semibold text-surface-foreground">
          Related topics
        </h4>
        <div
          class="mt-1 min-w-0"
          data-testid="node-detail-insight-related-topics-list"
        >
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
            <span
              class="min-w-0 flex-1 text-xs leading-snug text-muted"
              :title="row.preview"
            >{{ row.preview }}</span>
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
              ? 'Show fewer quotes'
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

      <p
        v-if="isTopicNode && topicClusterContext && !hasTopicClusterJson"
        class="mb-3 text-[10px] leading-snug text-muted"
        data-testid="node-detail-topic-cluster-context"
      >
        <span class="font-medium text-surface-foreground/80">Topic cluster:</span>
        {{ topicClusterContext.canonicalLabel }}
        <HelpTip
          class="ml-1 inline-flex align-middle"
          :pref-width="280"
          button-aria-label="About topic clusters"
        >
          <p class="font-sans text-[10px] leading-snug text-muted">
            This topic is grouped with similar
            <span class="font-mono">topic:…</span> ids in corpus clustering
            (<strong class="font-medium text-surface-foreground">search/topic_clusters.json</strong>). The graph
            can show a <strong class="font-medium text-surface-foreground">TopicCluster</strong> compound parent
            when that file is loaded. Detail and selection stay on this topic node.
          </p>
        </HelpTip>
      </p>

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
          Set Explore topic filter
        </button>
        <HelpTip
          class="shrink-0 self-center"
          :pref-width="280"
          button-aria-label="About topic Search and Explore shortcuts"
        >
          <p class="font-sans text-[10px] leading-snug text-muted">
            <strong class="font-medium text-surface-foreground">Prefill semantic search</strong> switches
            to Search with this topic label as the query (clears feed filter). Run Search to hit the vector index.
            <strong class="font-medium text-surface-foreground">Set Explore topic filter</strong> switches
            to Explore and fills <strong class="font-medium text-surface-foreground/90">Topic contains</strong>;
            press <strong class="font-medium text-surface-foreground/90">Run explore</strong> to load insights.
          </p>
        </HelpTip>
      </div>

      <div
        v-if="isTopicNode && props.nodeId && showSingleTopicTimelineButton"
        class="mb-3 flex items-center gap-2"
        role="group"
        aria-label="Topic timeline — corpus-wide episode list"
      >
        <button
          type="button"
          class="min-w-0 flex-1 rounded border border-border bg-canvas px-2 py-1.5 text-center text-xs font-medium leading-snug text-surface-foreground hover:bg-elevated disabled:opacity-40"
          data-testid="node-detail-topic-timeline"
          :disabled="topicTimelineDisabled"
          @click="openTopicTimeline"
        >
          Topic timeline
        </button>
        <HelpTip
          class="shrink-0 self-center"
          :pref-width="280"
          button-aria-label="About topic timeline (CIL)"
        >
          <p class="font-sans text-[10px] leading-snug text-muted">
            Opens a <strong class="font-medium text-surface-foreground">corpus-wide</strong> list:
            every episode under your corpus path with RFC-072 bridge + GI that has insights about this
            topic. You may see <strong class="font-medium text-surface-foreground">no rows, one episode,
            or several</strong> — that is how many matched, not a limit of the graph view. Uses the node
            <strong class="font-medium text-surface-foreground/90">id</strong> (e.g.
            <span class="font-mono">topic:…</span>).
          </p>
        </HelpTip>
      </div>

      <p
        v-if="bodyTextForTemplate"
        class="mb-3 text-xs leading-relaxed text-muted"
      >
        {{ truncate(bodyTextForTemplate, 600) }}
      </p>

      <p
        v-if="crossLayerBridgeLine"
        class="mb-2 text-[11px] leading-snug text-primary"
      >
        <span class="font-medium text-muted">Also in other graph layers:</span>
        {{ crossLayerBridgeLine }}
      </p>

      <section
        v-if="transcriptSourceSection"
        class="mb-2 flex flex-col gap-1.5"
      >
        <div
          v-if="transcriptSourceSection.href"
          class="flex flex-wrap items-center gap-2"
        >
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
        <div
          v-else-if="transcriptSourceSection.ref"
          class="text-[10px] leading-snug"
        >
          <span class="font-mono text-[11px] text-surface-foreground">{{ transcriptSourceSection.ref }}</span>
          <p class="mt-0.5 text-muted">
            Connect to the API and set <strong class="font-medium text-surface-foreground/90">Corpus path</strong>
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
          v-else-if="transcriptSourceSection.showMissingSpeakerHint && !transcriptSourceSection.href"
          class="text-[10px] leading-snug text-muted"
          data-testid="node-detail-quote-speaker-unavailable"
        >
          {{ GI_QUOTE_SPEAKER_UNAVAILABLE_HINT }}
        </p>
      </section>

      <dl
        v-if="extraProps.length"
        class="mb-2 space-y-1.5"
      >
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

      <div
        v-if="!props.embedInRail || graphRailDetailTab === 'neighbourhood'"
        id="node-detail-rail-panel-neighbourhood"
        class="min-h-0"
        :role="props.embedInRail ? 'tabpanel' : undefined"
        :aria-labelledby="
          props.embedInRail ? 'node-detail-rail-tab-neighbourhood' : undefined
        "
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
          :aggregated-neighbor-rows="
            useTopicClusterAggregatedConnections ? topicClusterAggregatedNeighbors : undefined
          "
          :topic-cluster-neighborhood="topicClusterNeighborhoodForMap"
          :connections-empty-hint="
            useTopicClusterAggregatedConnections ? TOPIC_CLUSTER_CONNECTIONS_EMPTY : undefined
          "
          :dense-neighbor-list="!props.embedInRail"
          @go-graph="emit('go-graph')"
          @open-library-episode="emit('open-library-episode', $event)"
          @prefill-semantic-search="emit('prefill-semantic-search', $event)"
        />
      </div>
    </div>

    <TranscriptViewerDialog ref="transcriptViewerRef" />
    <TopicTimelineDialog ref="topicTimelineRef" />
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
