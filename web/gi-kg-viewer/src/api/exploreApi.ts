export type ExploreKind = 'explore' | 'natural_language'

export interface ExploreApiBody {
  kind: ExploreKind
  error?: string | null
  detail?: string | null
  data?: Record<string, unknown> | null
  question?: string | null
  answer?: Record<string, unknown> | null
  explanation?: string | null
}

function buildBaseParams(corpusPath: string): URLSearchParams {
  const params = new URLSearchParams()
  params.set('path', corpusPath.trim())
  return params
}

export async function fetchExploreFiltered(
  corpusPath: string,
  options: {
    topic?: string
    speaker?: string
    groundedOnly?: boolean
    minConfidence?: number | null
    sortBy?: 'confidence' | 'time'
    limit?: number
    strict?: boolean
  },
): Promise<ExploreApiBody> {
  const params = buildBaseParams(corpusPath)
  if (options.topic?.trim()) params.set('topic', options.topic.trim())
  if (options.speaker?.trim()) params.set('speaker', options.speaker.trim())
  if (options.groundedOnly) params.set('grounded_only', 'true')
  if (options.minConfidence != null && Number.isFinite(options.minConfidence)) {
    params.set('min_confidence', String(options.minConfidence))
  }
  params.set('sort_by', options.sortBy ?? 'confidence')
  params.set('limit', String(Math.min(500, Math.max(1, options.limit ?? 50))))
  if (options.strict) params.set('strict', 'true')
  const res = await fetch(`/api/explore?${params}`)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as ExploreApiBody
}

export async function fetchExploreNaturalLanguage(
  corpusPath: string,
  question: string,
  options: { limit?: number; strict?: boolean } = {},
): Promise<ExploreApiBody> {
  const params = buildBaseParams(corpusPath)
  params.set('q', question.trim())
  params.set('limit', String(Math.min(500, Math.max(1, options.limit ?? 50))))
  if (options.strict) params.set('strict', 'true')
  const res = await fetch(`/api/explore?${params}`)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as ExploreApiBody
}
