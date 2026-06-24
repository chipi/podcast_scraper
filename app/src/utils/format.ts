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
