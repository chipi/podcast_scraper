import { defineStore } from 'pinia'
import { ref } from 'vue'

import { e2eHooksEnabled } from '../utils/e2eHooks'

export type SubjectKind = 'episode' | 'topic' | 'person' | 'graph-node' | null

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

  function clearFields(): void {
    episodeMetadataPath.value = null
    episodeUiLabel.value = null
    episodeId.value = null
    graphNodeCyId.value = null
    graphConnectionsCyId.value = null
    topicId.value = null
    personId.value = null
    positionTrackerTopicId.value = null
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

  function setEpisodeId(v: string | null): void {
    const t = v?.trim()
    episodeId.value = t ? t : null
  }

  function focusGraphNode(cyNodeId: string): void {
    const t = cyNodeId.trim()
    clearFields()
    if (!t) {
      kind.value = null
      return
    }
    kind.value = 'graph-node'
    graphNodeCyId.value = t
  }

  /**
   * Focus a Topic OR a non-Person Entity in the unified node view. Both open the
   * generic {@link focusGraphNode} → ``NodeDetail`` rail (the standalone
   * ``TopicEntityView`` panel is retired; its overview is folded into NodeDetail's
   * Details tab). The id may be a corpus id (``topic:…`` / ``entity:…``) or a graph
   * cy id (``g:…`` / ``tc:…``) — NodeDetail's lookups resolve either form.
   */
  function focusTopic(id: string): void {
    focusGraphNode(id)
  }

  /** Alias of {@link focusTopic} for non-Person Entity subjects. Same node view. */
  function focusEntity(id: string): void {
    focusGraphNode(id)
  }

  function focusPerson(id: string): void {
    const t = id.trim()
    clearFields()
    if (!t) {
      kind.value = null
      return
    }
    kind.value = 'person'
    personId.value = t
  }

  /**
   * #1049 — Pivot the Person Landing rail's Position Tracker tab onto a
   * specific Topic. No-op when no Person is currently focused (the
   * Position Tracker only renders when ``kind === 'person'``); silent
   * clear on empty / whitespace input.
   */
  function selectTopicForPositionTracker(topicGraphId: string): void {
    const t = topicGraphId.trim()
    if (kind.value !== 'person') return
    positionTrackerTopicId.value = t || null
  }

  function clearPositionTrackerTopic(): void {
    positionTrackerTopicId.value = null
  }

  function clearSubject(): void {
    kind.value = null
    clearFields()
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
      // E2E-only mutators. The TEV contract Playwright spec drives the
      // panel via ``focusTopic`` after the V2 architectural change removed
      // the digest topic-band-title click affordance; other handoff specs
      // use the FSM store dev hook. Mutators stay DEV-gated.
      focusTopic,
      focusEntity,
      focusPerson,
      clearSubject,
      selectTopicForPositionTracker,
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
    focusEpisode,
    focusGraphNode,
    focusTopic,
    focusEntity,
    focusPerson,
    clearSubject,
    setEpisodeUiLabel,
    setEpisodeId,
    selectTopicForPositionTracker,
    clearPositionTrackerTopic,
  }
})
