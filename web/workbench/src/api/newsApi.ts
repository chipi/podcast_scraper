/**
 * News items (UXS-019; ``GET /api/corpus/news/item`` and ``/items``). Metadata only: no route
 * returns an article's body (ADR-179).
 */
import type { NewsItem } from './searchApi'
import { fetchWithTimeout } from './httpClient'

/** A topic, person, organisation or object the item's KG names; ``role`` ``author`` for a byline. */
export interface NewsKgNode {
  id: string
  type: string
  label: string
  role?: string | null
}

export interface NewsItemDetail {
  path: string
  item: NewsItem
  kg_relative_path?: string | null
  kg_nodes: NewsKgNode[]
}

export interface NewsItemsPage {
  path: string
  items: NewsItem[]
  total: number
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetchWithTimeout(url)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export async function fetchNewsItem(corpusPath: string, itemId: string): Promise<NewsItemDetail> {
  const qs = new URLSearchParams({ path: corpusPath, item_id: itemId })
  return getJson<NewsItemDetail>(`/api/corpus/news/item?${qs.toString()}`)
}

export async function fetchNewsItems(
  corpusPath: string,
  opts: { publisherId?: string; limit?: number; offset?: number } = {},
): Promise<NewsItemsPage> {
  const qs = new URLSearchParams({ path: corpusPath })
  if (opts.publisherId) qs.set('publisher_id', opts.publisherId)
  if (opts.limit != null) qs.set('limit', String(opts.limit))
  if (opts.offset != null) qs.set('offset', String(opts.offset))
  return getJson<NewsItemsPage>(`/api/corpus/news/items?${qs.toString()}`)
}
