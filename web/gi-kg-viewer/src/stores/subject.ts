import { defineStore } from 'pinia'
import { ref } from 'vue'

export type SubjectKind = 'episode' | 'topic' | 'person' | 'graph-node' | null

/**
 * Right-hand **subject** rail: one focused entity (episode, graph node, …).
 * Search / Explore live in the left query panel and do not share this column.
 */
export const useSubjectStore = defineStore('subject', () => {
  const kind = ref<SubjectKind>(null)
  const episodeMetadataPath = ref<string | null>(null)
  const graphNodeCyId = ref<string | null>(null)
  const graphConnectionsCyId = ref<string | null>(null)
  const topicId = ref<string | null>(null)
  const personId = ref<string | null>(null)

  function clearFields(): void {
    episodeMetadataPath.value = null
    graphNodeCyId.value = null
    graphConnectionsCyId.value = null
    topicId.value = null
    personId.value = null
  }

  function focusEpisode(
    metadataPath: string,
    opts?: { graphConnectionsCyId?: string | null },
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

  return {
    kind,
    episodeMetadataPath,
    graphNodeCyId,
    graphConnectionsCyId,
    topicId,
    personId,
    focusEpisode,
    focusGraphNode,
    focusTopic,
    focusPerson,
    clearSubject,
  }
})
