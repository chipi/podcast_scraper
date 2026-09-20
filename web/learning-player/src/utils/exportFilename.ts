/**
 * A filename a person would recognise in their Files app.
 *
 * Exports were named from the episode SLUG, which is `{feed_slug}-{sha256hex}` — so saving your
 * notes produced `long-horizon-notes-9f2c4a1b…-notes.md` and the operator reported it as "some
 * crazy name" (2026-09-19). The slug is an addressing detail; a saved file is something you find
 * by reading it six months later.
 *
 * Deliberately conservative. This string reaches a filesystem, a share sheet and whatever app
 * receives it, and it is bounded — a 200-character podcast title is a real thing and is not a
 * filename. When a title yields nothing usable (emoji-only, or CJK that has no ASCII form) it
 * falls back rather than producing a bare extension.
 */

/**
 * Longest stem we will emit, before the extension. Keeps the whole name inside every filesystem
 * limit with room for the extension and any "(1)" a receiving app appends.
 */
const MAX_STEM = 60

/**
 * `title` → a safe filename stem, or `fallback` when nothing usable survives.
 *
 * An ALLOWLIST, not a blacklist. The first version banned the characters that looked dangerous —
 * and an em-dash and an emoji both walked straight through it, because a blacklist only knows the
 * hostile characters somebody thought of. Keeping `[a-z0-9]` and joining the runs with dashes is
 * the only rule that is true for every filesystem, share sheet and receiving app at once.
 *
 * Diacritics are folded first, so "Beyoncé" reads as "beyonce" rather than losing the letter.
 */
export function filenameStem(title: string | null | undefined, fallback: string): string {
  const words = (title ?? "")
    .normalize("NFKD")
    // Combining marks, left behind by NFKD once the base letter is separated out.
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .match(/[a-z0-9]+/g)
  if (!words) return fallback

  let stem = ""
  for (const w of words) {
    const next = stem ? `${stem}-${w}` : w
    // Cut on a WORD boundary rather than mid-word: a stem ending "…-conversa" reads as corruption.
    if (next.length > MAX_STEM) break
    stem = next
  }
  // A single word longer than the cap would leave nothing; take a hard slice in that one case.
  return stem || words[0].slice(0, MAX_STEM)
}

/** `filenameStem` plus an extension — the form every caller actually wants. */
export function exportFilename(
  title: string | null | undefined,
  ext: string,
  fallback: string
): string {
  return `${filenameStem(title, fallback)}.${ext}`
}
