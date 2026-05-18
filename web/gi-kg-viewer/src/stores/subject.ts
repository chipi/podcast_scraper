import { defineStore } from 'pinia'
import { ref } from 'vue'

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

  function clearFields(): void {
    episodeMetadataPath.value = null
    episodeUiLabel.value = null
    episodeId.value = null
    graphNodeCyId.value = null
    graphConnectionsCyId.value = null
    topicId.value = null
    personId.value = null
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
   * #672 — Focus a Topic OR a non-Person Entity. The same rail panel
   * (``TopicEntityView``) renders both: it inspects the resolved node's
   * ``type`` to label the header ``Topic`` / ``Entity`` accordingly. Use
   * {@link focusEntity} for call-site clarity at non-Topic entry points
   * (graph clicks still flow through {@link focusGraphNode} → ``NodeDetail``;
   * this entry point is for handoffs from Digest / Search / Explore that
   * want the higher-level subject overview).
   */
  function focusTopic(id: string): void {
    const t = id.trim()
    clearFields()
    if (!t) {
      kind.value = null
      return
    }
    kind.value = 'topic'
    topicId.value = t
  }

  /** Alias of {@link focusTopic} for non-Topic Entity subjects. Same rail panel. */
  function focusEntity(id: string): void {
    focusTopic(id)
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
  if (typeof window !== 'undefined' && import.meta.env?.DEV) {
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
      // E2E-only mutators. The TEV contract Playwright spec drives the
      // panel via ``focusTopic`` after the V2 architectural change removed
      // the digest topic-band-title click affordance; other handoff specs
      // use the FSM store dev hook. Mutators stay DEV-gated.
      focusTopic,
      focusEntity,
      clearSubject,
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
    focusEpisode,
    focusGraphNode,
    focusTopic,
    focusEntity,
    focusPerson,
    clearSubject,
    setEpisodeUiLabel,
    setEpisodeId,
  }
})
