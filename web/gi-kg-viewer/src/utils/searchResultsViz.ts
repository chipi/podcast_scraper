import type { SearchHit } from '../api/searchApi'

export interface CountRow {
  key: string
  /** Truncated for the list row. */
  label: string
  /** Pre-truncation display string (episode/feed title or id); use for tooltips. */
  fullLabel?: string
  count: number
  pct: number
}

/** Character caps for episode / feed title columns in search insights. */
export const EPISODE_INSIGHT_LABEL_MAX = 52
export const FEED_INSIGHT_LABEL_MAX = 44

/** Top-N rows plus how many categories and hits were omitted from the list. */
export interface RankedWithTail {
  rows: CountRow[]
  /** Distinct keys ranked below the top-N slice. */
  tailDistinct: number
  /** Total hits belonging to those tail keys. */
  tailHitCount: number
}

export interface ScoreStats {
  min: number
  max: number
  mean: number
  spread: number
}

export interface ScoreBarRow {
  rank: number
  score: number
  /** Bar length ∝ score / max(score) in this result set (honest magnitude). */
  widthPct: number
  docType: string
}

export interface DateBucket {
  label: string
  count: number
}

export interface PublishTimeline {
  buckets: DateBucket[]
  unparsed: number
}

export interface TermCount {
  term: string
  count: number
}

const DOC_TYPE_LABELS: Record<string, string> = {
  insight: 'Insights',
  quote: 'Quotes',
  kg_entity: 'KG entities',
  kg_topic: 'KG topics',
  summary: 'Summary bullets',
  transcript: 'Transcript chunks',
}

/** Basic English stopwords + short tokens we don't want to dominate “themes”. */
const STOPWORDS = new Set(
  [
    'that',
    'this',
    'with',
    'from',
    'have',
    'were',
    'been',
    'their',
    'there',
    'would',
    'could',
    'should',
    'about',
    'which',
    'these',
    'those',
    'what',
    'when',
    'where',
    'your',
    'into',
    'more',
    'some',
    'than',
    'then',
    'them',
    'very',
    'just',
    'like',
    'also',
    'only',
    'other',
    'such',
    'will',
    'here',
    'make',
    'made',
    'many',
    'most',
    'much',
    'each',
    'both',
    'even',
    'well',
    'being',
    'over',
    'after',
    'before',
    'between',
    'through',
    'during',
    'without',
    'against',
    'among',
    'while',
    'because',
    'however',
    'something',
    'nothing',
    'everything',
    'anything',
    'really',
    'actually',
    'maybe',
    'think',
    'said',
    'says',
    'going',
    'want',
    'need',
    'know',
    'people',
    'things',
    'thing',
    'time',
    'year',
    'years',
    'work',
    'good',
    'right',
    'back',
    'come',
    'came',
    'take',
    'took',
    'first',
    'last',
    'long',
    'same',
    'different',
    'another',
  ].map((w) => w.toLowerCase()),
)

function metaString(hit: SearchHit, key: string): string | null {
  const v = hit.metadata?.[key]
  return typeof v === 'string' && v.trim() ? v.trim() : null
}

/** Episode/feed ids are sometimes serialized as numbers in JSON. */
function metaId(hit: SearchHit, key: string): string | null {
  const v = hit.metadata?.[key]
  if (typeof v === 'string' && v.trim()) return v.trim()
  if (typeof v === 'number' && Number.isFinite(v)) return String(v)
  return null
}

/** Titles and similar fields: accept string; tolerate rare numeric ids mistaken as title. */
function metaText(hit: SearchHit, key: string): string | null {
  const v = hit.metadata?.[key]
  if (typeof v === 'string' && v.trim()) return v.trim()
  return null
}

function truncateKey(key: string, maxLen: number): string {
  if (key.length <= maxLen) return key
  return `${key.slice(0, Math.max(0, maxLen - 1))}…`
}

/** First non-empty title seen per id (any hit may carry titles after API backfill). */
function firstTitlePerScopeId(
  hits: SearchHit[],
  idKey: 'episode_id' | 'feed_id',
  titleKey: 'episode_title' | 'feed_title',
  missingIdLabel: string,
): Map<string, string> {
  const titles = new Map<string, string>()
  for (const h of hits) {
    const id = metaId(h, idKey) ?? missingIdLabel
    if (titles.has(id)) continue
    const t = metaText(h, titleKey)
    if (t) titles.set(id, t)
  }
  return titles
}

function rankedWithTailFromMap(
  counts: Map<string, number>,
  totalHits: number,
  limit: number,
  labelForKey: (key: string) => { label: string; fullLabel: string },
): RankedWithTail {
  const total = totalHits || 1
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1])
  const top = sorted.slice(0, limit)
  const tail = sorted.slice(limit)
  const rows = top.map(([key, count]) => {
    const { label, fullLabel } = labelForKey(key)
    return {
      key,
      label,
      fullLabel,
      count,
      pct: (count / total) * 100,
    }
  })
  const tailHitCount = tail.reduce((s, [, c]) => s + c, 0)
  return {
    rows,
    tailDistinct: tail.length,
    tailHitCount,
  }
}

/** Native `title` for episode rows (full title + stable id when title exists). */
export function episodeRowTooltip(row: CountRow): string {
  const full = row.fullLabel ?? row.label
  if (full === row.key || row.key.startsWith('(')) {
    return full
  }
  return `${full} · episode id: ${row.key}`
}

/** Native `title` for feed rows. */
export function feedRowTooltip(row: CountRow): string {
  const full = row.fullLabel ?? row.label
  if (full === row.key || row.key.startsWith('(')) {
    return full
  }
  return `${full} · feed id: ${row.key}`
}

export function docTypeDistribution(hits: SearchHit[]): CountRow[] {
  const counts = new Map<string, number>()
  for (const h of hits) {
    const raw = metaString(h, 'doc_type') ?? '?'
    const k = raw.toLowerCase()
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  const total = hits.length || 1
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([key, count]) => ({
      key,
      label: DOC_TYPE_LABELS[key] ?? key,
      count,
      pct: (count / total) * 100,
    }))
}

export function episodeDistribution(hits: SearchHit[], limit = 12): RankedWithTail {
  const counts = new Map<string, number>()
  const missing = '(no episode id)'
  for (const h of hits) {
    const id = metaId(h, 'episode_id') ?? missing
    counts.set(id, (counts.get(id) ?? 0) + 1)
  }
  const titles = firstTitlePerScopeId(hits, 'episode_id', 'episode_title', missing)
  return rankedWithTailFromMap(counts, hits.length, limit, (key) => {
    const fullLabel = titles.get(key) ?? key
    return {
      fullLabel,
      label: truncateKey(fullLabel, EPISODE_INSIGHT_LABEL_MAX),
    }
  })
}

export function feedDistribution(hits: SearchHit[], limit = 8): RankedWithTail {
  const counts = new Map<string, number>()
  const missing = '(no feed id)'
  for (const h of hits) {
    const id = metaId(h, 'feed_id') ?? missing
    counts.set(id, (counts.get(id) ?? 0) + 1)
  }
  const titles = firstTitlePerScopeId(hits, 'feed_id', 'feed_title', missing)
  return rankedWithTailFromMap(counts, hits.length, limit, (key) => {
    const fullLabel = titles.get(key) ?? key
    return {
      fullLabel,
      label: truncateKey(fullLabel, FEED_INSIGHT_LABEL_MAX),
    }
  })
}

export function computeScoreStats(hits: SearchHit[]): ScoreStats | null {
  if (!hits.length) return null
  const scores = hits.map((h) => h.score)
  const min = Math.min(...scores)
  const max = Math.max(...scores)
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length
  return { min, max, mean, spread: max - min }
}

/**
 * Bar width = score / max(score) in this list (0–100%). Length matches relative similarity strength.
 */
export function scoreBarsForHits(hits: SearchHit[]): ScoreBarRow[] {
  if (!hits.length) return []
  const scores = hits.map((h) => h.score)
  const max = Math.max(...scores)
  const denom = max > 0 ? max : 1
  return hits.map((h, i) => ({
    rank: i + 1,
    score: h.score,
    widthPct: Math.min(100, (Math.max(0, h.score) / denom) * 100),
    docType: metaString(h, 'doc_type') ?? '?',
  }))
}

function parsePublishMonth(raw: unknown): string | null {
  if (typeof raw !== 'string' || !raw.trim()) return null
  const s = raw.trim()
  const iso = /^(\d{4}-\d{2})/.exec(s)
  if (iso) return iso[1]
  return null
}

export function publishMonthTimeline(hits: SearchHit[]): PublishTimeline {
  const byMonth = new Map<string, number>()
  let unparsed = 0
  for (const h of hits) {
    const pub = h.metadata?.publish_date
    const month = parsePublishMonth(pub)
    if (!month) {
      unparsed += 1
      continue
    }
    byMonth.set(month, (byMonth.get(month) ?? 0) + 1)
  }
  const buckets = [...byMonth.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([label, count]) => ({ label, count }))
  return { buckets, unparsed }
}

function tokenizeForTerms(text: string): string[] {
  const lower = text.toLowerCase()
  const out: string[] = []
  for (const m of lower.matchAll(/[a-z][a-z0-9]{3,}/gi)) {
    const w = m[0]
    if (!STOPWORDS.has(w)) out.push(w)
  }
  return out
}

export function topTermsFromHits(hits: SearchHit[], limit = 28): TermCount[] {
  const freq = new Map<string, number>()
  for (const h of hits) {
    const t = typeof h.text === 'string' ? h.text : ''
    for (const w of tokenizeForTerms(t)) {
      freq.set(w, (freq.get(w) ?? 0) + 1)
    }
  }
  return [...freq.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([term, count]) => ({ term, count }))
}

/** One-line takeaway: dominant doc type in this hit list. */
export function insightDominantDocType(docTypes: CountRow[], totalHits: number): string | null {
  if (!docTypes.length || totalHits < 1) return null
  const top = docTypes[0]
  return `${top.label} dominates (${top.count} of ${totalHits} hits, ${top.pct.toFixed(0)}%).`
}

/** One-line takeaway for the publish-month timeline. */
export function insightTimeline(tl: PublishTimeline, totalHits: number): string | null {
  if (totalHits < 1) return null
  if (!tl.buckets.length) {
    if (tl.unparsed === totalHits) {
      return 'No parseable publish month in these hits.'
    }
    return tl.unparsed > 0
      ? `${tl.unparsed} hit(s) lack a parseable month; none bucketed.`
      : null
  }
  const peak = tl.buckets.reduce((best, b) => (b.count > best.count ? b : best))
  const parts = [`Peak month ${peak.label} (${peak.count} hits).`]
  if (tl.unparsed > 0) {
    parts.push(`${tl.unparsed} hit(s) not dated.`)
  }
  return parts.join(' ')
}

/** One-line takeaway for term frequencies. */
export function insightTopTerm(terms: TermCount[]): string | null {
  if (!terms.length) return null
  const t = terms[0]
  return `Strongest token: “${t.term}” (${t.count}×).`
}
