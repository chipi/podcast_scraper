/**
 * Interest tokens — the one place that knows their shape.
 *
 * A saved interest is a prefixed id: `topic:` and `tc:` (interest clusters), `thc:` (storylines,
 * "theme clusters"), and `person:` (followed from an entity card). Four prefixes, and the app had
 * three different opinions about them: `DiscoveryList` matched all four, `noteTarget` sliced at the
 * colon (correct by construction), and `ProfileView` hand-rolled `^(tc|topic|person):` — which
 * omits `thc`, so a followed storyline rendered as the literal
 * "thc:managing on the edge of chaos" on the user's own profile (operator 2026-09-18).
 *
 * The lesson is the omission, not the typo: a list of prefixes copied into a regex is a list that
 * goes stale the moment a fifth one appears. This slices at the separator instead, so it cannot.
 */

/** Anything followable carries one of these. */
export const INTEREST_PREFIX = /^(topic|tc|thc|person):/

export type InterestKind = 'topic' | 'storyline' | 'person'

/** What kind of thing an interest token names — drives chip hue and grouping. */
export function interestKind(id: string): InterestKind {
  if (id.startsWith('person:')) return 'person'
  if (id.startsWith('thc:') || id.startsWith('tc:')) return 'storyline'
  return 'topic'
}

/**
 * A human label for an interest token.
 *
 * `known` maps ids to real labels (the cluster set, the storyline list) and wins when it has one.
 * Otherwise the id is de-slugged: sliced at the FIRST colon rather than matched against a list of
 * prefixes, so an unknown fifth prefix degrades to a readable label instead of leaking raw.
 */
export function interestLabel(id: string, known?: Map<string, string>): string {
  const hit = known?.get(id)
  if (hit) return hit
  const bare = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id
  return bare.replace(/[-_]+/g, ' ').trim()
}

/**
 * De-duplicate interests by the LABEL a user would read, not by id.
 *
 * `topic:open-source-ai-models` and `tc:open-source-ai-models` are different tokens that render
 * identically, and the profile showed "open source ai models" twice (operator 2026-09-18). Keeps
 * the first occurrence, so ordering is unchanged.
 */
export function dedupeByLabel(
  ids: string[],
  known?: Map<string, string>,
): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const id of ids) {
    const key = interestLabel(id, known).toLowerCase()
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(id)
  }
  return out
}
