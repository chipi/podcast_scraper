/**
 * Map a CIL digest topic pill to ``graphNavigation.requestFocusNode`` arguments.
 *
 * Mirrors SearchPanel RFC-075 behaviour: primary selection stays on the ``topic:…``
 * node when present; optional ``tc:…`` compound ids widen the camera bbox only.
 */

import type { CilDigestTopicPill } from '../api/digestApi'

export type CilPillGraphFocusLike = Pick<
  CilDigestTopicPill,
  'topic_id' | 'in_topic_cluster' | 'topic_cluster_compound_id'
>

/** Compound parent ids for ``pendingFocusCameraIncludeRawIds`` (empty when not clustered). */
export function cameraIncludeRawIdsFromCilPill(pill: CilPillGraphFocusLike): string[] {
  if (!pill.in_topic_cluster) {
    return []
  }
  const id = pill.topic_cluster_compound_id?.trim()
  return id ? [id] : []
}

export type CilPillGraphFocusPlan =
  | { kind: 'none' }
  | { kind: 'episode_only'; primary: string }
  | {
      kind: 'topic'
      primary: string
      fallback: string | null
      cameraInclude: string[] | undefined
    }

/**
 * Resolve how to focus the graph from a pill + optional episode id (Digest / Episode rail).
 */
export function graphFocusPlanFromCilPill(
  pill: CilPillGraphFocusLike | null | undefined,
  episodeId: string | null | undefined,
): CilPillGraphFocusPlan {
  const topicHint = pill?.topic_id?.trim() ?? ''
  const eid = episodeId?.trim() ? episodeId.trim() : ''
  const cam = pill ? cameraIncludeRawIdsFromCilPill(pill) : []
  const cameraInclude = cam.length > 0 ? cam : undefined

  if (topicHint && eid) {
    return { kind: 'topic', primary: topicHint, fallback: eid, cameraInclude }
  }
  if (eid) {
    return { kind: 'episode_only', primary: eid }
  }
  if (topicHint) {
    return { kind: 'topic', primary: topicHint, fallback: null, cameraInclude }
  }
  return { kind: 'none' }
}

export type GraphNavFocusApi = {
  requestFocusNode: (
    nodeId: string,
    fallbackNodeId?: string | null,
    cameraIncludeRawIds?: string[] | null,
  ) => void
  clearPendingFocus: () => void
}

/** Apply ``graphFocusPlanFromCilPill`` to the graph navigation store. */
export function applyGraphFocusPlan(nav: GraphNavFocusApi, plan: CilPillGraphFocusPlan): void {
  if (plan.kind === 'none') {
    nav.clearPendingFocus()
    return
  }
  if (plan.kind === 'episode_only') {
    nav.requestFocusNode(plan.primary)
    return
  }
  if (plan.fallback) {
    nav.requestFocusNode(plan.primary, plan.fallback, plan.cameraInclude)
  } else {
    nav.requestFocusNode(plan.primary, undefined, plan.cameraInclude)
  }
}

/** Plan + apply in one step (call after artifacts are loaded). */
export function requestGraphFocusFromCilPill(
  nav: GraphNavFocusApi,
  pill: CilPillGraphFocusLike | null | undefined,
  episodeId: string | null | undefined,
): void {
  applyGraphFocusPlan(nav, graphFocusPlanFromCilPill(pill, episodeId))
}
