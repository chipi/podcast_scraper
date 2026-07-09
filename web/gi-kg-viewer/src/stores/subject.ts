import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { useGraphNavigationStore } from './graphNavigation'

import { e2eHooksEnabled } from '../utils/e2eHooks'

export type SubjectKind = 'episode' | 'topic' | 'person' | 'graph-node' | 'show' | null

/** Full subject state captured for the rail's Back history. */
interface SubjectSnapshot {
  kind: SubjectKind
  episodeMetadataPath: string | null
  episodeUiLabel: string | null
  episodeId: string | null
  graphNodeCyId: string | null
  graphConnectionsCyId: string | null
  topicId: string | null
  personId: string | null
  positionTrackerTopicId: string | null
  feedId: string | null
  feedUiLabel: string | null
}

/**
 * Right-hand **subject** rail: one focused entity (episode, graph node, …).
 * Search / Explore live in the left query panel and do not share this column.
 */
function truncateUiLabel(s: string, max = 72): string {
  const t = s.trim()
  if (t.length <= max) return t
  return `${t.slice(0, max - 1)}…`
}

export const useSubjectStore = defineStore('subject', () => {
  const kind = ref<SubjectKind>(null)
  const episodeMetadataPath = ref<string | null>(null)
  /** Optional strip / graph territory label (e.g. episode title). */
  const episodeUiLabel = ref<string | null>(null)
  /**
   * Episode UUID (logical id) for the focused episode when known. Used
   * to resolve the graph Cytoscape node id when Episode rows in the
   * artifact only carry the UUID in their node id (``__unified_ep__:UUID``)
   * and do **not** expose ``metadata_relative_path`` as a property — the
   * common case for the unified-merge graph. Without this, camera
   * centering after Library / Search / Dashboard handoff cannot resolve
   * the node and never animates.
   */
  const episodeId = ref<string | null>(null)
  const graphNodeCyId = ref<string | null>(null)
  const graphConnectionsCyId = ref<string | null>(null)
  const topicId = ref<string | null>(null)
  const personId = ref<string | null>(null)
  /**
   * #1049 — Topic drill-in for the Position Tracker tab on Person Landing
   * (PRD-028 / RFC-072 §5A `position_arc`). When non-null AND `kind` is
   * `'person'`, the Position Tracker panel renders insights for the pair
   * (``personId``, ``positionTrackerTopicId``) ordered by publish_date +
   * position_hint. Cleared whenever ``focusPerson`` switches to a different
   * person OR the subject is cleared, so a stale Topic from a prior Person
   * never bleeds into a new one.
   */
  const positionTrackerTopicId = ref<string | null>(null)
  /**
   * Focused Show (feed) for the Show rail (UXS-015 / RFC-104). ``feedId`` is the
   * corpus feed id; ``feedUiLabel`` is the show title, kept for the Back
   * affordance's label. The Show rail (``ShowRailPanel``) re-fetches the feed +
   * its episodes from ``feedId`` so a Back-restore needs only these two fields.
   */
  const feedId = ref<string | null>(null)
  const feedUiLabel = ref<string | null>(null)

  // Back history for the subject rail. Node→node navigations (topic → entity →
  // person → co-speaker) push the prior subject so the rail can offer a Back
  // affordance; without it, drilling into a related entity is a dead end.
  const history = ref<SubjectSnapshot[]>([])
  const canGoBack = computed(() => history.value.length > 0)

  function currentSnapshot(): SubjectSnapshot {
    return {
      kind: kind.value,
      episodeMetadataPath: episodeMetadataPath.value,
      episodeUiLabel: episodeUiLabel.value,
      episodeId: episodeId.value,
      graphNodeCyId: graphNodeCyId.value,
      graphConnectionsCyId: graphConnectionsCyId.value,
      topicId: topicId.value,
      personId: personId.value,
      positionTrackerTopicId: positionTrackerTopicId.value,
      feedId: feedId.value,
      feedUiLabel: feedUiLabel.value,
    }
  }

  function pushHistory(): void {
    if (kind.value == null) return
    history.value.push(currentSnapshot())
    // Bound the stack so a long browse session can't grow it without limit.
    if (history.value.length > 50) history.value.shift()
  }

  function restoreSnapshot(s: SubjectSnapshot): void {
    kind.value = s.kind
    episodeMetadataPath.value = s.episodeMetadataPath
    episodeUiLabel.value = s.episodeUiLabel
    episodeId.value = s.episodeId
    graphNodeCyId.value = s.graphNodeCyId
    graphConnectionsCyId.value = s.graphConnectionsCyId
    topicId.value = s.topicId
    personId.value = s.personId
    positionTrackerTopicId.value = s.positionTrackerTopicId
    feedId.value = s.feedId
    feedUiLabel.value = s.feedUiLabel
  }

  /** Pop the previous subject off the history and restore it (rail Back). */
  function back(): void {
    const s = history.value.pop()
    if (s) restoreSnapshot(s)
  }

  function clearFields(): void {
    episodeMetadataPath.value = null
    episodeUiLabel.value = null
    episodeId.value = null
    graphNodeCyId.value = null
    graphConnectionsCyId.value = null
    topicId.value = null
    personId.value = null
    positionTrackerTopicId.value = null
    feedId.value = null
    feedUiLabel.value = null
  }

  function focusEpisode(
    metadataPath: string,
    opts?: {
      graphConnectionsCyId?: string | null
      uiTitle?: string | null
      episodeId?: string | null
    },
  ): void {
    const t = metadataPath.trim()
    if (!t) {
      clearFields()
      kind.value = null
      return
    }
    /** Re-open same episode (e.g. **Open in graph**): avoid nulling ``episodeMetadataPath`` — Library ``loadEpisodes`` treats null as “no subject” and auto-selects the first row. */
    const sameEpisode =
      kind.value === 'episode' && episodeMetadataPath.value?.trim() === t
    if (!sameEpisode) {
      // Record the prior graph-node OR show subject for Back — e.g. show → episode
      // should let you return to the show (the episode rail carries a Back).
      if (kind.value === 'graph-node' || kind.value === 'show') pushHistory()
      clearFields()
    } else {
      graphNodeCyId.value = null
      topicId.value = null
      personId.value = null
      // #1049 — orphan-state leak: re-opening the same episode after a
      // Person was focused must not carry the prior Position Tracker
      // topic across (the Person rail is gone but the store state survives
      // and would leak back into the next Person focus).
      positionTrackerTopicId.value = null
    }
    kind.value = 'episode'
    episodeMetadataPath.value = t
    const cy = opts?.graphConnectionsCyId?.trim()
    graphConnectionsCyId.value = cy || null
    const lab = opts?.uiTitle?.trim()
    episodeUiLabel.value = lab ? truncateUiLabel(lab) : null
    const eid = opts?.episodeId?.trim()
    episodeId.value = eid ? eid : null
  }

  /**
   * Focus a Show (feed) in the right rail (UXS-015 / RFC-104). The show detail
   * opens as its own rail subject — **not** inside the Library surface. Pushes the
   * prior subject onto Back history (no-op when the rail was empty) so a show
   * reached from another subject can return. ``ShowRailPanel`` reads ``feedId``
   * and re-fetches the feed + its episodes.
   */
  function focusShow(id: string, opts?: { uiTitle?: string | null }): void {
    const t = id.trim()
    if (!t) {
      clearFields()
      kind.value = null
      return
    }
    const sameShow = kind.value === 'show' && feedId.value === t
    if (!sameShow) {
      pushHistory()
      clearFields()
    }
    kind.value = 'show'
    feedId.value = t
    const lab = opts?.uiTitle?.trim()
    feedUiLabel.value = lab ? truncateUiLabel(lab) : null
  }

  function setEpisodeId(v: string | null): void {
    const t = v?.trim()
    episodeId.value = t ? t : null
  }

  function focusGraphNode(cyNodeId: string, opts?: { syncGraph?: boolean }): void {
    const t = cyNodeId.trim()
    if (!t) {
      clearFields()
      kind.value = null
      return
    }
    // Record the prior subject for Back — node→node chains (not a no-op re-focus of
    // the same node), and show → node (a topic/person opened from a Show rail should
    // let you return to the show, mirroring focusEpisode's show → episode handling).
    if ((kind.value === 'graph-node' && graphNodeCyId.value !== t) || kind.value === 'show') {
      pushHistory()
    }
    clearFields()
    kind.value = 'graph-node'
    graphNodeCyId.value = t
    // #6 two-way sync: when the navigation comes from the detail rail (not a graph click, which
    // already has the node selected), ask the graph to select + centre this node so it reflects
    // where the details went. In-slice nodes apply immediately; a node the graph is loading via a
    // handoff is picked up the moment it appears (GraphCanvas' pendingFocusNodeId watcher).
    if (opts?.syncGraph) {
      useGraphNavigationStore().requestFocusNode(t)
    }
  }

  /**
   * Focus a Topic OR a non-Person Entity in the unified node view. Both open the
   * generic {@link focusGraphNode} → ``NodeDetail`` rail (the standalone
   * ``TopicEntityView`` panel is retired; its overview is folded into NodeDetail's
   * Details tab). The id may be a corpus id (``topic:…`` / ``entity:…``) or a graph
   * cy id (``g:…`` / ``tc:…``) — NodeDetail's lookups resolve either form.
   */
  function focusTopic(id: string): void {
    focusGraphNode(id, { syncGraph: true })
  }

  /** Alias of {@link focusTopic} for non-Person Entity subjects. Same node view. */
  function focusEntity(id: string): void {
    focusGraphNode(id, { syncGraph: true })
  }

  /**
   * Focus a Person in the unified node view — opens the generic
   * {@link focusGraphNode} → ``NodeDetail`` rail (with PersonLandingView folded
   * into its Details tab); the standalone Person rail is retired.
   */
  function focusPerson(id: string): void {
    focusGraphNode(id, { syncGraph: true })
  }

  /**
   * #1049 — Pivot the Person Landing rail's Position Tracker tab onto a
   * specific Topic. No-op when no Person is currently focused (the
   * Position Tracker only renders when ``kind === 'person'``); silent
   * clear on empty / whitespace input.
   */
  function selectTopicForPositionTracker(topicGraphId: string): void {
    const t = topicGraphId.trim()
    // Persons now open through the unified node view (kind 'graph-node'); the
    // Position Tracker only renders inside the embedded PersonLandingView, which
    // is shown for person nodes only, so a graph-node subject is the valid state.
    if (kind.value !== 'graph-node') return
    positionTrackerTopicId.value = t || null
  }

  function clearPositionTrackerTopic(): void {
    positionTrackerTopicId.value = null
  }

  function clearSubject(): void {
    kind.value = null
    clearFields()
    // Closing the rail ends the navigation session — drop the Back history.
    history.value = []
  }

  /** Update graph strip label without re-running ``focusEpisode`` (e.g. **Open in graph**). */
  function setEpisodeUiLabel(v: string | null): void {
    const t = v?.trim()
    episodeUiLabel.value = t ? truncateUiLabel(t) : null
  }

  // -------------------------------------------------------------------------
  // Dev hook for E2E specs (L5 subject-store correctness assertion).
  //
  // After every handoff the subject store must reflect the envelope target —
  // ``kind`` matches the envelope's ``kind``, and the id field for that kind
  // (``episodeId`` / ``graphNodeCyId`` / ``topicId`` / ``personId``) carries
  // the resolved value. ``assertHandoffApplied`` reads this to pin the
  // user-visible "the rail shows the right thing" contract — selection
  // alone isn't enough because rail panels read this store, not cy directly.
  // -------------------------------------------------------------------------
  if (typeof window !== 'undefined' && e2eHooksEnabled) {
    ;(window as unknown as { __GIKG_SUBJECT__?: object }).__GIKG_SUBJECT__ = {
      get kind() {
        return kind.value
      },
      get episodeMetadataPath() {
        return episodeMetadataPath.value
      },
      get episodeId() {
        return episodeId.value
      },
      get graphNodeCyId() {
        return graphNodeCyId.value
      },
      get topicId() {
        return topicId.value
      },
      get personId() {
        return personId.value
      },
      get positionTrackerTopicId() {
        return positionTrackerTopicId.value
      },
      get feedId() {
        return feedId.value
      },
      get canGoBack() {
        return canGoBack.value
      },
      // E2E-only mutators. The TEV contract Playwright spec drives the
      // panel via ``focusTopic`` after the V2 architectural change removed
      // the digest topic-band-title click affordance; other handoff specs
      // use the FSM store dev hook. Mutators stay DEV-gated.
      focusTopic,
      focusEntity,
      focusPerson,
      focusShow,
      clearSubject,
      selectTopicForPositionTracker,
      back,
    }
  }

  return {
    kind,
    episodeMetadataPath,
    episodeUiLabel,
    episodeId,
    graphNodeCyId,
    graphConnectionsCyId,
    topicId,
    personId,
    positionTrackerTopicId,
    feedId,
    feedUiLabel,
    focusEpisode,
    focusShow,
    focusGraphNode,
    focusTopic,
    focusEntity,
    focusPerson,
    clearSubject,
    setEpisodeUiLabel,
    setEpisodeId,
    selectTopicForPositionTracker,
    clearPositionTrackerTopic,
    canGoBack,
    back,
  }
})
