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
    clearFields()
    if (!t) {
      kind.value = null
      return
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
    focusPerson,
    clearSubject,
    setEpisodeUiLabel,
  }
})
