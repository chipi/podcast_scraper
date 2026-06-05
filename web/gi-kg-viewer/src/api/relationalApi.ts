import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

/**
 * Client for the relational-query layer (RFC-094 / #882): `GET /api/relational/*`.
 * Thin traversals over the typed corpus graph — positions, who-said, cross-show
 * synthesis, a show's episodes, related insights. Surfaces consume these to ground
 * Detail/Digest panels (PRD-033 FR3/FR4).
 */

function raiseRelationalHttpError(res: Response, bodyText: string): never {
  if (res.status === 404) {
    throw new Error(
      'Relational endpoint not found (404). Restart the viewer API from a current checkout (pip install -e ".[dev]", then make serve-api or podcast serve).',
    )
  }
  throw new Error(bodyText || `HTTP ${res.status}`)
}

/** One graph node projected for a relational result (mirrors RelatedNodeModel). */
export interface RelatedNode {
  id: string
  type: string
  text: string
  show_id: string
  episode_id: string
}

/** Flat relational response (positions / insights-about / entities-in / episodes / related). */
export interface RelationalListResponse {
  subject: string
  results: RelatedNode[]
  error?: string | null
}

/** Grouped relational response (who-said keyed by person; cross-show keyed by show). */
export interface RelationalGroupedResponse {
  subject: string
  groups: Record<string, RelatedNode[]>
  error?: string | null
}

async function getList(url: string): Promise<RelationalListResponse> {
  return dedupeInFlight(`GET|${url}`, async () => {
    const res = await fetchWithTimeout(url)
    if (!res.ok) raiseRelationalHttpError(res, await res.text())
    return (await res.json()) as RelationalListResponse
  })
}

async function getGrouped(url: string): Promise<RelationalGroupedResponse> {
  return dedupeInFlight(`GET|${url}`, async () => {
    const res = await fetchWithTimeout(url)
    if (!res.ok) raiseRelationalHttpError(res, await res.text())
    return (await res.json()) as RelationalGroupedResponse
  })
}

function withK(q: URLSearchParams, k?: number): void {
  if (k != null) q.set('k', String(Math.max(1, Math.floor(k))))
}

/** FR3.2 — top insight per distinct show covering a topic. Groups keyed by show id. */
export async function fetchCrossShow(
  corpusPath: string,
  topic: string,
  perShow = 1,
): Promise<RelationalGroupedResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim(), topic: topic.trim() })
  if (perShow !== 1) q.set('per_show', String(Math.max(1, Math.floor(perShow))))
  return getGrouped(`/api/relational/cross-show?${q.toString()}`)
}

/** FR4.2 — per-person insights about a topic. Groups keyed by person id. */
export async function fetchWhoSaid(
  corpusPath: string,
  topic: string,
  k?: number,
): Promise<RelationalGroupedResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim(), topic: topic.trim() })
  withK(q, k)
  return getGrouped(`/api/relational/who-said?${q.toString()}`)
}

/** FR4.1 — insights a person stated. */
export async function fetchPositions(
  corpusPath: string,
  person: string,
  k?: number,
): Promise<RelationalListResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim(), person: person.trim() })
  withK(q, k)
  return getList(`/api/relational/positions?${q.toString()}`)
}

/** Insights that mention an entity. */
export async function fetchInsightsAbout(
  corpusPath: string,
  entity: string,
  k?: number,
): Promise<RelationalListResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim(), entity: entity.trim() })
  withK(q, k)
  return getList(`/api/relational/insights-about?${q.toString()}`)
}

/** A show's episodes (HAS_EPISODE). */
export async function fetchShowEpisodes(
  corpusPath: string,
  podcast: string,
  k?: number,
): Promise<RelationalListResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim(), podcast: podcast.trim() })
  withK(q, k)
  return getList(`/api/relational/episodes?${q.toString()}`)
}

/** Sibling insights sharing a topic or mentioned entity. */
export async function fetchRelatedInsights(
  corpusPath: string,
  insight: string,
  k?: number,
): Promise<RelationalListResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim(), insight: insight.trim() })
  withK(q, k)
  return getList(`/api/relational/related-insights?${q.toString()}`)
}
