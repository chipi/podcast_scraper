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
  const graphNodeCyId = ref<string | null>(null)
  const graphConnectionsCyId = ref<string | null>(null)
  const topicId = ref<string | null>(null)
  const personId = ref<string | null>(null)

  function clearFields(): void {
    episodeMetadataPath.value = null
    episodeUiLabel.value = null
    graphNodeCyId.value = null
    graphConnectionsCyId.value = null
    topicId.value = null
    personId.value = null
  }

  function focusEpisode(
    metadataPath: string,
    opts?: { graphConnectionsCyId?: string | null; uiTitle?: string | null },
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

  return {
    kind,
    episodeMetadataPath,
    episodeUiLabel,
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
  }
})
