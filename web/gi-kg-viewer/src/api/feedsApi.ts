import { fetchWithTimeout } from './httpClient'
import { readApiErrorMessage } from './readApiErrorMessage'

/** One feed: URL string or object with `url` plus optional per-feed overrides (RFC-077). */
export type FeedApiEntry = string | Record<string, unknown>

export interface FeedsList {
  path: string
  file_relpath: string
  feeds: FeedApiEntry[]
}

export async function getFeeds(corpusPath: string): Promise<FeedsList> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(`/api/feeds?${q}`, undefined, { timeoutDetail: 'feeds' })
  if (!res.ok) {
    throw new Error(await readApiErrorMessage(res))
  }
  return (await res.json()) as FeedsList
}

export async function putFeeds(corpusPath: string, feeds: FeedApiEntry[]): Promise<FeedsList> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(
    `/api/feeds?${q}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feeds }),
    },
    { timeoutDetail: 'feeds' },
  )
  if (!res.ok) {
    throw new Error(await readApiErrorMessage(res))
  }
  return (await res.json()) as FeedsList
}
