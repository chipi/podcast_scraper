export interface TextSegment {
  text: string
  match: boolean
  /** 0-based index among matches (only set when ``match`` is true). */
  matchIndex?: number
}

/**
 * Split ``text`` into match / non-match segments for case-insensitive in-place
 * search highlighting (the pipeline job-log viewer find, #695). Empty query →
 * the whole text as one non-match segment. Each match segment carries its
 * 0-based ``matchIndex`` so the UI can mark the active hit + navigate.
 */
export function textSearchSegments(
  text: string,
  query: string,
): { segments: TextSegment[]; matchCount: number } {
  if (!query) {
    return { segments: [{ text, match: false }], matchCount: 0 }
  }
  const lower = text.toLowerCase()
  const ql = query.toLowerCase()
  const segments: TextSegment[] = []
  let i = 0
  let count = 0
  while (i <= text.length) {
    const idx = lower.indexOf(ql, i)
    if (idx < 0) {
      if (i < text.length) segments.push({ text: text.slice(i), match: false })
      break
    }
    if (idx > i) segments.push({ text: text.slice(i, idx), match: false })
    segments.push({ text: text.slice(idx, idx + ql.length), match: true, matchIndex: count })
    count += 1
    i = idx + ql.length
  }
  return { segments, matchCount: count }
}
