import { fetchWithTimeout } from './httpClient'

export interface FeedsList {
  path: string
  file_relpath: string
  urls: string[]
}

export async function getFeeds(corpusPath: string): Promise<FeedsList> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(`/api/feeds?${q}`, undefined, { timeoutDetail: 'feeds' })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as FeedsList
}

export async function putFeeds(corpusPath: string, urls: string[]): Promise<FeedsList> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(
    `/api/feeds?${q}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls }),
    },
    { timeoutDetail: 'feeds' },
  )
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as FeedsList
}
