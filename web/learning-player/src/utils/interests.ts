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

export type InterestKind = 'topic' | 'theme' | 'storyline' | 'person'

/**
 * What kind of thing an interest token names — drives chip hue, labelling and grouping.
 *
 * `tc:` and `thc:` are SEPARATE kinds (operator 2026-09-19). They were both reported as
 * `'storyline'`, which was harmless only while the kind picked a hue and nothing said it out loud:
 * the profile now names each pill's kind, and calling a semantic cluster a storyline would be
 * false. They are genuinely different objects — `thc:` is co-occurrence ("these topics keep coming
 * up together", #1603), `tc:` is vector similarity ("these topics mean similar things") — and
 * conflating them is the same confusion the operator hit on the topic card.
 *
 * NOTE THE INVERSION between the wire names and the product ones, because it WILL mislead: the
 * backend's `thc:` is a "theme cluster" and is what a reader calls a STORYLINE, while `tc:` is a
 * "topic cluster" and is what a reader calls a THEME. This function is the boundary where the wire
 * names stop mattering, and every surface should take its word from here rather than the prefix.
 *
 * The inversion is DEFERRED, not permanent. Pre-launch there are no users whose tokens must be
 * preserved, so renaming the prefixes — and the modules, and the artifact — is a bounded
 * mechanical refactor rather than a migration. It goes all the way down (`storylines.py`
 * serves storylines; `topic_clusters.py` serves themes; the artifact is `topic_theme_clusters`),
 * which is why #1603 keeps being reopened by people reading it the natural way. A comment warning
 * that something "WILL mislead" is the codebase admitting a fix was available and declined; that
 * option closes at launch, and after it this note becomes retroactively true.
 */
export function interestKind(id: string): InterestKind {
  if (id.startsWith('person:')) return 'person'
  if (id.startsWith('thc:')) return 'storyline'
  if (id.startsWith('tc:')) return 'theme'
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
