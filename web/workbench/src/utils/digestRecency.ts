/**
 * Recency dot for Digest / Library episode rows: published within the last rolling 24h.
 * Callers pass ``published_at || publish_date``: the stored UTC instant when the server has one,
 * else the day alone, which is treated as **local calendar midnight** for that YYYY-MM-DD.
 */

import { parseStoredInstant } from './formatting'

const YMD = /^(\d{4})-(\d{2})-(\d{2})$/

const MS_PER_HOUR = 3_600_000
const MS_PER_DAY = 86_400_000

/** Parse ``YYYY-MM-DD`` as local midnight; invalid patterns return null. */
export function parsePublishYmdLocalMidnight(ymd: string): Date | null {
  const m = YMD.exec(ymd.trim())
  if (!m) {
    return null
  }
  const y = Number(m[1])
  const mo = Number(m[2]) - 1
  const d = Number(m[3])
  const dt = new Date(y, mo, d)
  if (Number.isNaN(dt.getTime())) {
    return null
  }
  if (dt.getFullYear() !== y || dt.getMonth() !== mo || dt.getDate() !== d) {
    return null
  }
  return dt
}

function publishedMoment(raw: string): Date | null {
  return parseStoredInstant(raw) ?? parsePublishYmdLocalMidnight(raw)
}

/** True when the publish moment is within the last 24h (non-negative age). */
export function isPublishDateWithin24hRolling(
  publishDate: string | null | undefined,
): boolean {
  const s = publishDate?.trim()
  if (!s) {
    return false
  }
  const pub = publishedMoment(s)
  if (!pub) {
    return false
  }
  const age = Date.now() - pub.getTime()
  return age >= 0 && age < MS_PER_DAY
}

/** Hover copy: whole hours since the publish moment. */
export function recencyDotHoverTitle(
  publishDate: string | null | undefined,
): string | undefined {
  const s = publishDate?.trim()
  if (!s) {
    return undefined
  }
  const pub = publishedMoment(s)
  if (!pub) {
    return undefined
  }
  const age = Date.now() - pub.getTime()
  if (age < 0) {
    return undefined
  }
  const hours = Math.floor(age / MS_PER_HOUR)
  if (hours <= 0) {
    return 'Published less than 1 hour ago'
  }
  return `Published ${hours} hour${hours === 1 ? '' : 's'} ago`
}
