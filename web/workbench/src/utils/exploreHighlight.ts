/** One run of plain text or highlighted substring (Explore topic/speaker "contains"). */
export type ExploreTextSegment = { text: string; mark: boolean }

/**
 * Split display text into plain / highlighted runs for a case-insensitive substring needle.
 * When ``needle`` is empty, returns a single non-highlight segment (optionally truncated).
 */
export function segmentsForSubstringNeedle(
  text: string,
  needle: string,
  maxLength: number,
): ExploreTextSegment[] {
  const n = needle.trim()
  const truncated = text.length > maxLength ? `${text.slice(0, maxLength)}…` : text
  if (!n) {
    return [{ text: truncated, mark: false }]
  }
  const lower = truncated.toLowerCase()
  const key = n.toLowerCase()
  const out: ExploreTextSegment[] = []
  let i = 0
  while (i < truncated.length) {
    const j = lower.indexOf(key, i)
    if (j < 0) {
      out.push({ text: truncated.slice(i), mark: false })
      break
    }
    if (j > i) {
      out.push({ text: truncated.slice(i, j), mark: false })
    }
    out.push({ text: truncated.slice(j, j + key.length), mark: true })
    i = j + key.length
  }
  return out.length > 0 ? out : [{ text: truncated, mark: false }]
}
