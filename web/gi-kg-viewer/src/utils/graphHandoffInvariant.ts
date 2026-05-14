/**
 * Pure helper for the self-healing invariant in the graph handoff orchestrator
 * (decision #5 / ADR-079 § Self-healing reconciliation).
 *
 * The invariant: every node in the logical artifact's `viewWithEgo(focusNodeId)`
 * exists in Cytoscape's `core.nodes()`. Violations get one targeted
 * `core.add()` retry, then accept divergence + log if still violated.
 *
 * This module is pure — no Vue, no Pinia, no Cytoscape. The set-difference
 * computation is done over plain ID sets so it can be unit-tested in
 * isolation. The actual `core.add()` reconciliation lives in
 * `GraphCanvas.vue:finishLayoutPass` (it needs Cytoscape access). This helper
 * is the predicate; the consumer applies it.
 */

/**
 * Compute the set-difference between expected and actual node IDs.
 *
 * - `missing`: IDs in `expected` but not in `actual` (logical artifact has
 *   the node but Cytoscape doesn't render it; reconciliation candidate)
 * - `extra`: IDs in `actual` but not in `expected` (Cytoscape has nodes the
 *   logical artifact doesn't expect; usually benign — leftover from prior
 *   layout that hasn't been cleared yet)
 *
 * The invariant holds when both arrays are empty.
 */
export function computeNodeIdSetDifference(
  expectedIds: Iterable<string>,
  actualIds: Iterable<string>,
): { missing: string[]; extra: string[] } {
  const expected = expectedIds instanceof Set ? expectedIds : new Set(expectedIds)
  const actual = actualIds instanceof Set ? actualIds : new Set(actualIds)
  const missing: string[] = []
  for (const id of expected) {
    if (!actual.has(id)) missing.push(id)
  }
  const extra: string[] = []
  for (const id of actual) {
    if (!expected.has(id)) extra.push(id)
  }
  return { missing, extra }
}

/**
 * Decide whether a violation is reconcilable via a targeted `core.add()`.
 *
 * Reconciliation rules (decision #5 / FSM spec):
 * - 0 missing nodes → invariant holds; no action.
 * - 1..MAX_TARGETED_RECONCILE_MISSING missing nodes → targeted `core.add()`
 *   is safe; one retry per envelope generation.
 * - More than MAX_TARGETED_RECONCILE_MISSING missing → too many to safely
 *   re-add inline; accept divergence + log a structured warning.
 *
 * Returns one of:
 *   - `'ok'` (invariant holds; nothing to do)
 *   - `'reconcile'` (targeted `core.add()` recommended)
 *   - `'accept-divergence'` (too many missing; log + give up)
 */
export const MAX_TARGETED_RECONCILE_MISSING = 20

export type ReconcileDecision = 'ok' | 'reconcile' | 'accept-divergence'

export function decideReconcileAction(
  missing: readonly string[],
  alreadyRetried: boolean,
): ReconcileDecision {
  if (missing.length === 0) return 'ok'
  if (alreadyRetried) return 'accept-divergence'
  if (missing.length >= MAX_TARGETED_RECONCILE_MISSING) {
    return 'accept-divergence'
  }
  return 'reconcile'
}
