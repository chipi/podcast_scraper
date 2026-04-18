import { defineStore } from 'pinia'
import { ref } from 'vue'

/** Tracks RFC-076 progressive graph expansion (seed cy id -> appended artifact paths). */
export const useGraphExpansionStore = defineStore('graphExpansion', () => {
  const expandedBySeed = ref<Record<string, { addedRelPaths: string[] }>>({})
  const truncationLine = ref<string | null>(null)
  const expansionBusyCyId = ref<string | null>(null)

  /** Probe result: ``node-episodes`` would append GI/KG not already in ``selectedRelPaths``. */
  const corpusBeyondByCyId = ref<Record<string, boolean>>({})
  /** Bumped when selection or corpus root changes so in-flight probes ignore stale commits. */
  const corpusBeyondProbeGen = ref(0)

  function isExpanded(seedCyId: string): boolean {
    return Boolean(expandedBySeed.value[seedCyId.trim()])
  }

  function clearTruncationLine(): void {
    truncationLine.value = null
  }

  function setTruncationLine(msg: string | null): void {
    truncationLine.value = msg
  }

  function recordExpand(seedCyId: string, addedRelPaths: string[]): void {
    const id = seedCyId.trim()
    if (!id) {
      return
    }
    expandedBySeed.value = { ...expandedBySeed.value, [id]: { addedRelPaths } }
  }

  async function collapseSeed(seedCyId: string): Promise<void> {
    const id = seedCyId.trim()
    const block = expandedBySeed.value[id]
    if (!block) {
      return
    }
    const { useArtifactsStore } = await import('./artifacts')
    const artifacts = useArtifactsStore()
    await artifacts.removeRelativeArtifacts(block.addedRelPaths)
    const next = { ...expandedBySeed.value }
    delete next[id]
    expandedBySeed.value = next
    truncationLine.value = null
  }

  function setBusy(cyId: string | null): void {
    expansionBusyCyId.value = cyId?.trim() ? cyId.trim() : null
  }

  function resetExpansionState(): void {
    expandedBySeed.value = {}
    truncationLine.value = null
    expansionBusyCyId.value = null
    invalidateCorpusBeyondHints()
  }

  function invalidateCorpusBeyondHints(): void {
    corpusBeyondByCyId.value = {}
    corpusBeyondProbeGen.value += 1
  }

  function peekCorpusBeyondProbeGen(): number {
    return corpusBeyondProbeGen.value
  }

  function commitCorpusBeyondProbe(waveGen: number, cyId: string, wouldAppend: boolean): void {
    if (waveGen !== corpusBeyondProbeGen.value) {
      return
    }
    const id = cyId.trim()
    if (!id) {
      return
    }
    corpusBeyondByCyId.value = { ...corpusBeyondByCyId.value, [id]: wouldAppend }
  }

  function corpusBeyondAppendKnown(cyId: string): boolean | undefined {
    const id = cyId.trim()
    if (!id || !Object.hasOwn(corpusBeyondByCyId.value, id)) {
      return undefined
    }
    return corpusBeyondByCyId.value[id]
  }

  return {
    expandedBySeed,
    truncationLine,
    expansionBusyCyId,
    isExpanded,
    clearTruncationLine,
    setTruncationLine,
    recordExpand,
    collapseSeed,
    setBusy,
    resetExpansionState,
    invalidateCorpusBeyondHints,
    peekCorpusBeyondProbeGen,
    commitCorpusBeyondProbe,
    corpusBeyondAppendKnown,
  }
})
