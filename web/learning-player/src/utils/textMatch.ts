/**
 * Does `text` contain EVERY word of `query`, in any order? Case-insensitive.
 *
 * The rule for the client-side parts of search (operator 2026-10-04: "confirm how we search if we
 * enter a few words"). Notes used a single substring test, so "memory sleep" missed a note that
 * said "sleep and memory" — two words both present, just not side by side in that order. Shows
 * take the same rule, so a search for "pragmatic engineer" and one for "engineer pragmatic" find
 * the same show.
 *
 * The corpus passages are a different matcher entirely (server-side hybrid: meaning + keywords),
 * and a person/topic card needs a near-exact name — neither is this function's business.
 */
export function matchesAllWords(text: string | null | undefined, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length || !text) return false
  const hay = text.toLowerCase()
  return words.every((w) => hay.includes(w))
}
