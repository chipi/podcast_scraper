import type { Core, NodeSingular } from 'cytoscape'

/**
 * Resolve an FSM-applied cy id to a live node on ``core``, tolerating a prefix flip.
 *
 * The same logical node may now be addressable under a different ``g:`` (GI) / ``k:`` (KG)
 * prefix if KG joined GI-only data (or vice-versa) between the apply and a later redraw.
 * Resolution order: the exact id, then the alternative-prefix variant, finally — for a bare
 * (unprefixed) id — both prefixed guesses.
 *
 * Shared by GraphCanvas's two selection-restore paths (#967): the layoutstop restore and the
 * full-replacement early-restore, so both resolve the handoff target identically. Pure +
 * side-effect free (only reads ``core``) → unit-testable without mounting the canvas.
 */
export function resolveHandoffCandidateNode(core: Core, appliedCyId: string): NodeSingular | null {
  const id = appliedCyId.trim()
  if (!id) return null
  const candidateIds: string[] = [id]
  if (id.startsWith('g:')) {
    candidateIds.push('k:' + id.slice(2))
  } else if (id.startsWith('k:')) {
    candidateIds.push('g:' + id.slice(2))
  } else {
    candidateIds.push('g:' + id, 'k:' + id)
  }
  for (const candId of candidateIds) {
    const n = core.$id(candId)
    if (n.length > 0) return n.first() as NodeSingular
  }
  return null
}
