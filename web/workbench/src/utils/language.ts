/** `en-US` / `EN` / ` en ` → `en`; empty or missing → null. */
export function primaryLanguage(tag: string | null | undefined): string | null {
  const primary = (tag ?? '').trim().split(/[-_]/)[0].toLowerCase()
  return primary || null
}

/**
 * Whether these shows span more than one language (PRD-047 FR7.5) — the only case in which a
 * language badge says anything. Regional spellings of one language count once.
 */
export function spansLanguages(tags: readonly (string | null | undefined)[]): boolean {
  const found = new Set<string>()
  for (const tag of tags) {
    const lang = primaryLanguage(tag)
    if (lang) found.add(lang)
  }
  return found.size > 1
}
