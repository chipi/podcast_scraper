/** Small, locale-aware formatters for catalog/player UI (i18n-friendly — no hard-coded text). */

/** Human episode duration: `48 min` under an hour, else `1h 03m`. Null passes through. */
export function formatDuration(seconds: number | null | undefined): string | null {
  if (seconds == null || seconds <= 0) return null
  const total = Math.round(seconds)
  const mins = Math.round(total / 60)
  if (total < 3600) return `${mins} min`
  const h = Math.floor(total / 3600)
  const m = Math.round((total % 3600) / 60)
  return `${h}h ${String(m).padStart(2, '0')}m`
}

/**
 * Humanise a KG speaker id for display: `person:matthew-walker` → `matthew walker`; a non-`person:`
 * value passes through. ONE canonical place so the `person:` slug convention can't drift across the
 * transcript, the player and the insights panel.
 */
export function speakerLabel(s: string | null): string | null {
  if (!s) return null
  return s.startsWith('person:') ? s.slice('person:'.length).replace(/-/g, ' ') : s
}

/** Format a `YYYY-MM-DD` publish date for the given locale; raw string on parse failure. */
export function formatPublishDate(iso: string | null | undefined, locale = 'en'): string | null {
  if (!iso) return null
  const d = new Date(`${iso.slice(0, 10)}T00:00:00`)
  if (Number.isNaN(d.getTime())) return iso
  try {
    return new Intl.DateTimeFormat(locale, { dateStyle: 'medium' }).format(d)
  } catch {
    return iso
  }
}
