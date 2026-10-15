/**
 * The Language facet of a list toolbar (operator 2026-10-10; episodes too, 2026-10-15): offered only
 * when the list spans two or more languages — one language has nothing to choose between. Options
 * are named in the reader's language ("English", "Spanish"), the codes the list filters by as values.
 */
export function languageName(code: string, locale: string): string {
  try {
    const name = new Intl.DisplayNames([locale], { type: 'language' }).of(code) ?? code
    return name.charAt(0).toUpperCase() + name.slice(1)
  } catch {
    return code.toUpperCase()
  }
}

export function languageOptions(
  codes: readonly string[],
  locale: string,
  allLabel: string,
): { value: string; label: string }[] | undefined {
  if (codes.length < 2) return undefined
  return [{ value: '', label: allLabel }, ...codes.map((c) => ({ value: c, label: languageName(c, locale) }))]
}
