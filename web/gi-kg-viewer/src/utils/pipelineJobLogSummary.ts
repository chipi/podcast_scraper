/**
 * Pull structured JSON blobs from the tail of a pipeline job log (INFO lines from
 * ``corpus_operations``: ``multi_feed_batch:`` / ``corpus_multi_feed_summary:``).
 */

function parseBalancedJson(s: string, start: number): unknown | null {
  const c = s[start]
  if (c !== '{' && c !== '[') {
    return null
  }
  const stack: string[] = [c]
  let inStr = false
  let esc = false
  for (let i = start + 1; i < s.length; i++) {
    const ch = s[i]!
    if (inStr) {
      if (esc) {
        esc = false
      } else if (ch === '\\') {
        esc = true
      } else if (ch === '"') {
        inStr = false
      }
      continue
    }
    if (ch === '"') {
      inStr = true
      continue
    }
    if (ch === '{') {
      stack.push('{')
      continue
    }
    if (ch === '[') {
      stack.push('[')
      continue
    }
    if (ch === '}') {
      if (stack.length === 0 || stack[stack.length - 1] !== '{') {
        return null
      }
      stack.pop()
      if (stack.length === 0) {
        const chunk = s.slice(start, i + 1)
        try {
          return JSON.parse(chunk) as unknown
        } catch {
          return null
        }
      }
      continue
    }
    if (ch === ']') {
      if (stack.length === 0 || stack[stack.length - 1] !== '[') {
        return null
      }
      stack.pop()
      if (stack.length === 0) {
        const chunk = s.slice(start, i + 1)
        try {
          return JSON.parse(chunk) as unknown
        } catch {
          return null
        }
      }
    }
  }
  return null
}

function extractAfterMarker(tail: string, marker: string): unknown | null {
  const idx = tail.lastIndexOf(marker)
  if (idx < 0) {
    return null
  }
  const rest = tail.slice(idx + marker.length)
  const j = rest.search(/[\[{]/)
  if (j < 0) {
    return null
  }
  return parseBalancedJson(rest, j)
}

export interface PipelineLogStructuredSummaries {
  multiFeedBatch: unknown | null
  corpusMultiFeedSummary: unknown | null
}

export function extractStructuredSummariesFromLogTail(tail: string): PipelineLogStructuredSummaries {
  return {
    multiFeedBatch: extractAfterMarker(tail, 'multi_feed_batch:'),
    corpusMultiFeedSummary: extractAfterMarker(tail, 'corpus_multi_feed_summary:'),
  }
}
