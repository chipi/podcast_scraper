import type { Core } from 'cytoscape'

import type { RawGraphNode } from '../types/artifact'
import { graphCyIdRepresentsEpisodeNode } from './graphEpisodeMetadata'
import { stripLayerPrefixesForCil } from './mergeGiKg'

/**
 * True when a graph node may run RFC-076 cross-episode expand (plain ``dbltap``).
 * Mirrors ``GraphCanvas`` gate: Topic / Person / Entity, canonical CIL id, degree &gt; 1.
 */
export function graphNodeExpandableForRfc076(
  core: Core,
  cyId: string,
  rawNode: RawGraphNode | null,
): boolean {
  if (graphCyIdRepresentsEpisodeNode(cyId, rawNode)) {
    return false
  }
  const t = rawNode?.type
  if (t !== 'Topic' && t !== 'Person' && t !== 'Entity') {
    return false
  }
  const bare = stripLayerPrefixesForCil(cyId)
  if (!/^(person|org|topic):/.test(bare)) {
    return false
  }
  try {
    const n = core.$id(cyId)
    if (n.empty() || typeof n.isNode !== 'function' || !n.isNode()) {
      return false
    }
    if (n.degree() <= 1) {
      return false
    }
  } catch {
    return false
  }
  return true
}

const EXPANDABLE_CLASS = 'rfc076-expandable'
const EXPANDED_SEED_CLASS = 'rfc076-expanded-seed'

/** Apply Cytoscape classes so the stylesheet can show expand / collapse affordance. */
export function syncRfc076ExpansionNodeClasses(
  core: Core,
  deps: {
    isExpandedSeed: (cyId: string) => boolean
    rawNode: (cyId: string) => RawGraphNode | null
    /**
     * When set, teal ``rfc076-expandable`` only if this returns ``true`` (corpus has GI/KG beyond
     * the merged selection). ``undefined`` means probe not finished — no ring yet.
     */
    corpusWouldAppendOutsideGraph?: (cyId: string) => boolean | undefined
  },
): void {
  core.batch(() => {
    core.nodes().forEach((ele) => {
      try {
        if (!ele || typeof ele.isNode !== 'function' || !ele.isNode()) {
          return
        }
        const cyId = ele.id()
        ele.removeClass(EXPANDABLE_CLASS)
        ele.removeClass(EXPANDED_SEED_CLASS)
        if (deps.isExpandedSeed(cyId)) {
          ele.addClass(EXPANDED_SEED_CLASS)
        } else if (graphNodeExpandableForRfc076(core, cyId, deps.rawNode(cyId))) {
          const corpus = deps.corpusWouldAppendOutsideGraph?.(cyId)
          const allowTeal =
            deps.corpusWouldAppendOutsideGraph == null ? true : corpus === true
          if (allowTeal) {
            ele.addClass(EXPANDABLE_CLASS)
          }
        }
      } catch {
        /* ignore per-node */
      }
    })
  })
}
