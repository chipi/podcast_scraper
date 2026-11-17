/**
 * Turn a non-OK ``fetch`` response into a short user-facing string.
 * FastAPI often returns ``{ "detail": "…" }`` or ``{ "detail": { "error", "keys" } }``.
 */
export function formatFastApiDetail(detail: unknown, status: number): string {
  if (Array.isArray(detail)) {
    try {
      return JSON.stringify(detail)
    } catch {
      return `HTTP ${status}`
    }
  }
  if (typeof detail === 'string') {
    if (status === 409 && detail.toLowerCase().includes('forbidden')) {
      return `${detail} Use the Feeds tab for RSS URLs, or edit the operator file on disk to remove forbidden keys.`
    }
    return detail
  }
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const o = detail as Record<string, unknown>
    const err = o.error
    const keys = o.keys
    if (err === 'forbidden_operator_feed_keys' && Array.isArray(keys)) {
      const k = (keys as unknown[]).map(String).join(', ')
      return `Remove feed keys from operator YAML (use the Feeds tab): ${k}`
    }
    if (err === 'forbidden_operator_keys' && Array.isArray(keys)) {
      const k = (keys as unknown[]).map(String).join(', ')
      return `Forbidden operator keys: ${k}`
    }
  }
  if (typeof detail === 'object' && detail !== null) {
    try {
      return JSON.stringify(detail)
    } catch {
      return `HTTP ${status}`
    }
  }
  return `HTTP ${status}`
}

export async function readApiErrorMessage(res: Response): Promise<string> {
  const raw = await res.text()
  const ct = (res.headers.get('content-type') || '').toLowerCase()
  const trimmed = raw.trim()
  if (ct.includes('application/json') && trimmed.startsWith('{')) {
    try {
      const j = JSON.parse(trimmed) as { detail?: unknown }
      if ('detail' in j) {
        return formatFastApiDetail(j.detail, res.status)
      }
    } catch {
      /* fall through */
    }
  }
  return trimmed || `HTTP ${res.status}`
}
