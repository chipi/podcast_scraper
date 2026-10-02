/**
 * Display casing for a person's name — a UI concern, applied at render.
 *
 * People-centric surfaces (top voices, trending, related people, the people browser) show names
 * exactly as the envelope sends them, and the pipeline normalises a lot of them to lowercase. A
 * pill reading "simon wilson" next to one reading "Lenny Rachitsky" looks like a data bug to a
 * reader, because a name is a proper noun and that is the one word class a reader expects
 * capitalised (operator 2026-10-01).
 *
 * Deliberately NOT fixed in the pipeline: the stored form is the matching key, and changing it
 * would reopen every identity join. This is presentation only, so it can be wrong without
 * consequence beyond the pixel — which matters, because the rules below are heuristics and a
 * person's name is theirs to spell.
 *
 * ## The one invariant
 *
 * **It never lowercases a letter that is already uppercase.** That single rule preserves the cases
 * a naive title-case destroys — `McCarthy`, `MacLeod`, `O'Brien`, `LeBron`, `DeAndre`, `IBM` — and
 * it means the function can only ever ADD capitalisation. A name that already arrives correct
 * passes through untouched, so the blast radius is limited to names that arrive lowercase.
 *
 * ## Known and accepted wrong
 *
 * A name that is *deliberately* lowercase (danah boyd, bell hooks, k.d. lang) arrives as lowercase
 * and gets capitalised, because nothing in the data distinguishes it from a normalised one. That is
 * a real cost and it is accepted rather than unnoticed: the alternative — capitalise nothing — is
 * wrong for the overwhelming majority. If a reader reports one, the fix is an exception list here.
 */

/**
 * Name particles that stay lowercase mid-name — `van der Berg`, `Ludwig van Beethoven`.
 *
 * Capitalised when they LEAD the name, because then they are the name's first word and carry it
 * ("Van Halen", a surname used alone). Mid-name they are connectives, not names.
 */
const PARTICLES = new Set([
  'van', 'von', 'de', 'der', 'den', 'del', 'della', 'di', 'da', 'dos', 'das', 'du', 'la', 'le',
  'les', 'ter', 'ten', 'op', 'af', 'al', 'bin', 'ibn', 'bint', 'y', 'e',
])

/** Capitalise one word, never lowering what is already upper. */
function capitalizeWord(word: string): string {
  // Split on the separators that start a new capitalised part INSIDE a word: `Jean-Luc`,
  // `O'Brien`. The separators are kept so the word can be rebuilt exactly.
  return word
    .split(/([-'’])/)
    .map((part, i) => {
      if (i % 2 === 1) return part // the separator itself
      if (!part) return part
      // Only the FIRST character is touched, and only upward. `McCarthy` keeps its C because
      // nothing here lowercases, and `o'brien` becomes `O'Brien` via the separator split.
      return part.charAt(0).toUpperCase() + part.slice(1)
    })
    .join('')
}

/**
 * A person's name as it should be shown.
 *
 * Returns the input unchanged when it is empty or whitespace, so a caller can pass a possibly
 * absent name straight through without a guard.
 */
export function personName(name: string | null | undefined): string {
  if (!name) return ''
  const trimmed = name.trim()
  if (!trimmed) return ''

  // Collapse runs of whitespace so "simon   wilson" does not render with its original gaps; the
  // envelope is not always tidy and this is the display path.
  const words = trimmed.split(/\s+/)
  return words
    .map((word, i) => {
      // A particle mid-name is left EXACTLY as it arrived — not lowercased. Returning
      // `word.toLowerCase()` here read naturally and broke the invariant above: it turned the
      // operator-spelled "Jan Van Der Berg" into "Jan van der Berg", i.e. it lowered letters the
      // person had capitalised themselves. Leaving it alone gets both right, because the only
      // thing this function may do is ADD capitalisation: "van" stays "van", "Van" stays "Van".
      if (i > 0 && PARTICLES.has(word.toLowerCase())) return word
      return capitalizeWord(word)
    })
    .join(' ')
}

/**
 * The same, for a value that may be a slug rather than a name (`simon-wilson`, `person:ada-lovelace`).
 *
 * Used where a surface falls back to an id because the envelope carried no name. Separate from
 * `personName` so the common path does not pay for de-slugging, and so a real name containing a
 * hyphen (`Jean-Luc Picard`) is never split on it.
 */
export function personNameFromId(id: string | null | undefined): string {
  if (!id) return ''
  const bare = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id
  return personName(bare.replace(/[-_]+/g, ' '))
}
