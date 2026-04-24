import type { CorpusManifestFeed, CorpusRunSummaryItem } from '../api/corpusMetricsApi'

function normalizeFeedUrl(u: string): string {
  const t = u.trim()
  if (!t) {
    return ''
  }
  try {
    const x = new URL(t)
    return `${x.origin}${x.pathname}`.toLowerCase().replace(/\/+$/, '')
  } catch {
    return t.toLowerCase().replace(/\/+$/, '')
  }
}

function normalizeRelPath(p: string): string {
  return p.replace(/\\/g, '/').toLowerCase()
}

/** Whether ``relative_path`` (to ``run.json``) sits under ``feeds/<segment>/`` (or ends with ``/<segment>/run.json``). */
export function runPathContainsFeedDir(relativePath: string, stableFeedDir: string): boolean {
  const p = normalizeRelPath(relativePath)
  const s = stableFeedDir.trim().toLowerCase().replace(/^\/+|\/+$/g, '')
  if (!s) {
    return false
  }
  if (p.includes(`feeds/${s}/`) || p.includes(`/feeds/${s}/`)) {
    return true
  }
  if (p.endsWith(`/${s}/run.json`) || p.endsWith(`${s}/run.json`)) {
    return true
  }
  if (p.includes(`/${s}/run.json`)) {
    return true
  }
  return false
}

export function stableFeedDirForUrl(manifestFeeds: CorpusManifestFeed[], feedUrl: string): string | null {
  const want = normalizeFeedUrl(feedUrl)
  if (!want) {
    return null
  }
  for (const f of manifestFeeds) {
    const u = f.feed_url
    const d = f.stable_feed_dir
    if (typeof u !== 'string' || typeof d !== 'string' || !d.trim()) {
      continue
    }
    if (normalizeFeedUrl(u) === want) {
      return d.trim()
    }
  }
  return null
}

/** Exactly one discovered run under this feed dir → safe 1:1 link target. */
export function uniqueRunForFeedDir(
  runs: CorpusRunSummaryItem[],
  stableFeedDir: string,
): CorpusRunSummaryItem | null {
  const hits = runs.filter((r) => runPathContainsFeedDir(r.relative_path, stableFeedDir))
  if (hits.length !== 1) {
    return null
  }
  return hits[0] ?? null
}

/** Per feed row: ``run.json`` ``relative_path`` when manifest + runs uniquely identify it. */
export function feedRowRunRelativePaths(
  rowFeedUrls: string[],
  runs: CorpusRunSummaryItem[],
  manifestFeeds: CorpusManifestFeed[] | undefined,
): (string | null)[] {
  if (!manifestFeeds?.length || !runs.length || !rowFeedUrls.length) {
    return rowFeedUrls.map(() => null)
  }
  return rowFeedUrls.map((url) => {
    const dir = stableFeedDirForUrl(manifestFeeds, url)
    if (!dir) {
      return null
    }
    const run = uniqueRunForFeedDir(runs, dir)
    return run?.relative_path ?? null
  })
}
