// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import DigestView from './DigestView.vue'
import { fetchCorpusFeeds } from '../../api/corpusLibraryApi'
import {
  fetchCorpusDigest,
  type CorpusDigestResponse,
  type CorpusDigestRow,
  type CorpusDigestTopicBand,
} from '../../api/digestApi'
import { fetchCrossShow } from '../../api/relationalApi'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import { useArtifactsStore } from '../../stores/artifacts'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

/**
 * What the operator does in Digest, against realistic bands and rows (2026-10-09).
 *
 * The mount test that pins the corpus-revision reload brought this view into the coverage
 * denominator with 6 of its 85 functions exercised. These tests drive the rest the way an operator
 * does: topic bands (ranking, show more, trend arrows, the topic panel, search, cross-show), hit
 * rows into the graph, the Recent list (rows, keyboard, feed links, topic pills), and the states
 * around them.
 */
vi.mock('../../api/digestApi', async (orig) => ({
  ...(await orig<typeof import('../../api/digestApi')>()),
  fetchCorpusDigest: vi.fn(),
}))
vi.mock('../../api/corpusLibraryApi', async (orig) => ({
  ...(await orig<typeof import('../../api/corpusLibraryApi')>()),
  fetchCorpusFeeds: vi.fn(),
}))
vi.mock('../../api/relationalApi', async (orig) => ({
  ...(await orig<typeof import('../../api/relationalApi')>()),
  fetchCrossShow: vi.fn(),
}))
vi.mock('../../composables/useEnrichmentEnvelopeCache', async (orig) => ({
  ...(await orig<typeof import('../../composables/useEnrichmentEnvelopeCache')>()),
  fetchCachedCorpusEnvelope: vi.fn(),
}))
// The shell store probes /api/health when the path changes; a probe that never answers leaves the
// health this test sets alone.
vi.mock('../../api/httpClient', async (orig) => ({
  ...(await orig<typeof import('../../api/httpClient')>()),
  fetchWithTimeout: () => new Promise(() => {}),
}))

function row(n: number, over: Partial<CorpusDigestRow> = {}): CorpusDigestRow {
  return {
    metadata_relative_path: `feeds/f${n % 2}/ep${n}.metadata.json`,
    feed_id: `f${n % 2}`,
    episode_id: `ep-${n}`,
    episode_title: `Recent ${n}`,
    publish_date: '2026-01-0' + ((n % 9) + 1),
    summary_title: `Title ${n}`,
    summary_bullets_preview: [`bullet ${n}`],
    summary_preview: `Recap of ${n}`,
    gi_relative_path: `feeds/f${n % 2}/ep${n}.gi.json`,
    kg_relative_path: `feeds/f${n % 2}/ep${n}.kg.json`,
    has_gi: true,
    has_kg: true,
    duration_seconds: 900,
    episode_number: n,
    cil_digest_topics: [{ topic_id: 'topic:risk', label: 'risk management' }],
    ...over,
  }
}

function band(id: string, score: number, over: Partial<CorpusDigestTopicBand> = {}): CorpusDigestTopicBand {
  return {
    topic_id: id,
    label: `Band ${id}`,
    query: `query ${id}`,
    graph_topic_id: `topic:${id}`,
    hits: [
      {
        metadata_relative_path: `feeds/f1/hit-${id}.metadata.json`,
        episode_title: `Hit for ${id}`,
        feed_id: 'f1',
        score,
        episode_id: `hit-${id}`,
        publish_date: '2026-01-05',
        has_gi: true,
        gi_relative_path: `feeds/f1/hit-${id}.gi.json`,
        has_kg: false,
        summary_preview: `Why ${id} matters`,
      },
    ],
    ...over,
  }
}

function digestOf(over: Partial<CorpusDigestResponse> = {}): CorpusDigestResponse {
  return {
    path: '/corpus',
    window: 'all',
    window_start_utc: '2026-01-01T00:00:00Z',
    window_end_utc: '2026-02-01T00:00:00Z',
    compact: false,
    rows: [row(1), row(2), row(3)],
    topics: [band('a', 0.2), band('b', 0.9), band('c', 0.5), band('d', 0.1, { graph_topic_id: undefined })],
    topics_unavailable_reason: null,
    ...over,
  }
}

async function mountDigest(d: CorpusDigestResponse = digestOf()) {
  vi.mocked(fetchCorpusDigest).mockResolvedValue(d)
  vi.mocked(fetchCorpusFeeds).mockResolvedValue({
    path: '/corpus',
    feeds: [
      { feed_id: 'f0', display_title: 'Show Zero', episode_count: 2 },
      { feed_id: 'f1', display_title: 'Show One', episode_count: 2 },
    ],
  })
  vi.mocked(fetchCachedCorpusEnvelope).mockResolvedValue({
    data: { topics: [{ topic_id: 'topic:b', velocity_last_over_6mo: 2.4 }] },
  } as never)
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  shell.corpusLibraryApiAvailable = true
  shell.corpusDigestApiAvailable = true
  const artifacts = useArtifactsStore()
  vi.spyOn(artifacts, 'appendRelativeArtifacts').mockResolvedValue(undefined as never)
  const w = mount(DigestView, { attachTo: document.body })
  await flushPromises()
  return { w, artifacts }
}

const bandLabels = (w: ReturnType<typeof mount>) =>
  w.findAll('section[aria-label^="Topic "]').map((s) => s.attributes('aria-label'))
const recentRows = (w: ReturnType<typeof mount>) => w.findAll('[data-digest-recent-row]')

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => {
  vi.clearAllMocks()
  vi.useRealTimers()
  document.body.innerHTML = ''
})

describe('DigestView — topic bands', () => {
  it('ranks bands by retrieval signal and shows three until "Show more"', async () => {
    const { w } = await mountDigest()
    expect(bandLabels(w)).toEqual(['Topic Band b', 'Topic Band c', 'Topic Band a'])
    await w.get('[data-testid="digest-topic-bands-show-more"]').trigger('click')
    expect(bandLabels(w)).toHaveLength(4)
  })

  it('a band with velocity data shows its trend arrow', async () => {
    const { w } = await mountDigest()
    const trend = w.get('[data-testid="digest-band-trend"]')
    expect(trend.attributes('aria-label')).toBe('trending 2.4x its 6-mo average')
  })

  it("a mapped band's title opens its topic; Search topic prefills search", async () => {
    const { w } = await mountDigest()
    await w.findAll('[data-testid="digest-band-topic-link"]')[0].trigger('click')
    expect(useSubjectStore().graphNodeCyId).toBe('topic:b')
    const search = w.findAll('button').find((b) => b.text() === 'Search topic')!
    await search.trigger('click')
    expect(w.emitted('focus-search')?.at(-1)).toEqual([{ feed: '', query: 'query b' }])
  })

  it('a hit row opens the episode in Library and the topic in the graph', async () => {
    const { w, artifacts } = await mountDigest()
    await w.get('section[aria-label="Topic Band b"] [role="button"]').trigger('click')
    await flushPromises()
    expect(w.emitted('open-library-episode')?.at(-1)).toEqual([
      { metadata_relative_path: 'feeds/f1/hit-b.metadata.json' },
    ])
    expect(w.emitted('switch-main-tab')?.at(-1)).toEqual(['graph'])
    expect(artifacts.appendRelativeArtifacts).toHaveBeenCalledWith(['feeds/f1/hit-b.gi.json'])
  })

  it('a hit with no GI/KG on disk says so instead of opening an empty graph', async () => {
    const d = digestOf({
      topics: [band('x', 0.9, { hits: [{ ...band('x', 0.9).hits[0], has_gi: false, has_kg: false }] })],
    })
    const { w } = await mountDigest(d)
    await w.get('section[aria-label="Topic Band x"] [role="button"]').trigger('keydown', { key: 'Enter' })
    await flushPromises()
    expect(w.text()).toContain('No GI/KG artifacts on disk for this episode.')
  })

  it('"Across shows" loads one insight per show, labelled with the show name', async () => {
    vi.mocked(fetchCrossShow).mockResolvedValue({
      groups: { f0: [{ id: 'i1', type: 'insight', text: 'Show Zero says this', show_id: 'f0', episode_id: 'e' }] },
      error: null,
    } as never)
    const { w } = await mountDigest()
    await w.findAll('[data-testid="digest-cross-show-toggle"]')[0].trigger('click')
    await flushPromises()
    const rows = w.findAll('[data-testid="digest-cross-show-row"]')
    expect(rows).toHaveLength(1)
    expect(rows[0].text()).toContain('Show Zero')
    expect(rows[0].text()).toContain('Show Zero says this')
    await w.findAll('[data-testid="digest-cross-show-toggle"]')[0].trigger('click')
    expect(w.find('[data-testid="digest-cross-show-band"]').exists()).toBe(false)
  })

  it('"Across shows" says so when it fails or finds nothing', async () => {
    vi.mocked(fetchCrossShow).mockRejectedValueOnce(new Error('relational down'))
    const { w } = await mountDigest()
    const toggles = w.findAll('[data-testid="digest-cross-show-toggle"]')
    await toggles[0].trigger('click')
    await flushPromises()
    expect(w.text()).toContain('relational down')
    vi.mocked(fetchCrossShow).mockResolvedValueOnce({ groups: {}, error: null } as never)
    await toggles[1].trigger('click')
    await flushPromises()
    expect(w.text()).toContain('No cross-show coverage yet for this topic.')
  })

  it('topics that need an index say so, and the rows still show', async () => {
    const { w } = await mountDigest(digestOf({ topics: [], topics_unavailable_reason: 'no_index' }))
    expect(w.text()).toContain('Semantic topics need a vector index for this corpus.')
    expect(recentRows(w)).toHaveLength(3)
  })
})

describe('DigestView — Recent', () => {
  it('lists the rows with show names, and a click or Enter opens one in Library', async () => {
    const { w } = await mountDigest()
    expect(w.findAll('[data-testid="digest-recent-row-title"]').map((t) => t.text())).toEqual([
      'Recent 1',
      'Recent 2',
      'Recent 3',
    ])
    expect(w.findAll('[data-testid="digest-feed-name-link"]')[0].text()).toBe('Show One')
    await recentRows(w)[1].trigger('click')
    expect(w.emitted('open-library-episode')?.at(-1)).toEqual([{ metadata_relative_path: row(2).metadata_relative_path }])
    await recentRows(w)[2].trigger('keydown', { key: 'Enter' })
    expect(w.emitted('open-library-episode')?.at(-1)).toEqual([{ metadata_relative_path: row(3).metadata_relative_path }])
  })

  it('arrow keys move through the rows', async () => {
    const { w } = await mountDigest()
    await recentRows(w)[0].trigger('keydown', { key: 'ArrowDown' })
    expect(w.emitted('open-library-episode')?.at(-1)).toEqual([{ metadata_relative_path: row(2).metadata_relative_path }])
  })

  it("a row's show name opens Library scoped to that show", async () => {
    const { w } = await mountDigest()
    await w.findAll('[data-testid="digest-feed-name-link"]')[0].trigger('click')
    expect(useDashboardNavStore().pending).toEqual({ kind: 'library', feedId: 'f1' })
    expect(w.emitted('switch-main-tab')?.at(-1)).toEqual(['library'])
  })

  it("a row's topic pill opens the graph on that topic", async () => {
    const { w, artifacts } = await mountDigest()
    w.findAllComponents({ name: 'CilTopicPillsRow' })[0].vm.$emit('pill-click', 0)
    await flushPromises()
    expect(artifacts.appendRelativeArtifacts).toHaveBeenCalledWith([
      row(1).gi_relative_path,
      row(1).kg_relative_path,
    ])
    expect(w.emitted('switch-main-tab')?.at(-1)).toEqual(['graph'])
  })

  it('an empty window says so', async () => {
    const { w } = await mountDigest(digestOf({ rows: [], topics: [] }))
    expect(w.text()).toContain('No episodes in this window.')
  })
})

describe('DigestView — states', () => {
  it('a failed digest shows its error', async () => {
    vi.mocked(fetchCorpusDigest).mockRejectedValue(new Error('digest exploded'))
    vi.mocked(fetchCorpusFeeds).mockResolvedValue({ path: '/corpus', feeds: [] })
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    shell.corpusLibraryApiAvailable = true
    shell.corpusDigestApiAvailable = true
    const w = mount(DigestView)
    await flushPromises()
    expect(w.text()).toContain('digest exploded')
  })

  it('an API without the digest endpoint says so', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    shell.corpusLibraryApiAvailable = true
    shell.corpusDigestApiAvailable = false
    vi.mocked(fetchCorpusFeeds).mockResolvedValue({ path: '/corpus', feeds: [] })
    const w = mount(DigestView)
    await flushPromises()
    expect(w.text()).toContain('does not expose the digest endpoint')
    expect(fetchCorpusDigest).not.toHaveBeenCalled()
  })

  it('a new date reloads the digest for that window, after the debounce', async () => {
    const { w } = await mountDigest()
    vi.useFakeTimers()
    const before = vi.mocked(fetchCorpusDigest).mock.calls.length
    w.findComponent({ name: 'DateChip' }).vm.$emit('update:modelValue', '2026-01-15')
    await vi.advanceTimersByTimeAsync(1000)
    expect(vi.mocked(fetchCorpusDigest).mock.calls.length).toBeGreaterThan(before)
    expect(vi.mocked(fetchCorpusDigest).mock.calls.at(-1)![1]).toMatchObject({ window: 'since', since: '2026-01-15' })
  })
})

/**
 * Stack test flake, CI run 40736007754 attempt 1 (2026-10-28): three digest loads overlapped and
 * answered 6.6-9.4 s late, the API had 4 rows, the Recent list rendered none and no further digest
 * request followed. These pin that overlapping loads, with the gate's inputs flickering falsy in
 * between, still end with the rows on screen.
 */
describe('DigestView — overlapping loads', () => {
  function deferred<T>() {
    let resolve!: (v: T) => void
    const promise = new Promise<T>((r) => (resolve = r))
    return { promise, resolve }
  }

  for (const flicker of ['healthStatus', 'corpusDigestApiAvailable'] as const) {
    it(`a slow load superseded while ${flicker} is momentarily falsy still ends with rows shown`, async () => {
      const slow = deferred<CorpusDigestResponse>()
      vi.mocked(fetchCorpusDigest).mockReturnValueOnce(slow.promise)
      const { w } = await mountDigest()
      expect(recentRows(w)).toHaveLength(0)
      const shell = useShellStore()
      if (flicker === 'healthStatus') shell.healthStatus = null
      else shell.corpusDigestApiAvailable = false
      await flushPromises()
      if (flicker === 'healthStatus') shell.healthStatus = 'ok'
      else shell.corpusDigestApiAvailable = true
      await flushPromises()
      slow.resolve(digestOf())
      await flushPromises()
      expect(recentRows(w).length).toBeGreaterThan(0)
    })
  }

  it('a slow load superseded while health is falsy, resolving BEFORE health returns, still ends with rows shown', async () => {
    const slow = deferred<CorpusDigestResponse>()
    vi.mocked(fetchCorpusDigest).mockReturnValueOnce(slow.promise)
    const { w } = await mountDigest()
    const shell = useShellStore()
    shell.healthStatus = null
    await flushPromises()
    slow.resolve(digestOf())
    await flushPromises()
    shell.healthStatus = 'ok'
    await flushPromises()
    expect(recentRows(w).length).toBeGreaterThan(0)
  })

  it('two overlapping slow loads (a corpus revision bump mid-flight) end with the latest rows shown', async () => {
    const first = deferred<CorpusDigestResponse>()
    const second = deferred<CorpusDigestResponse>()
    vi.mocked(fetchCorpusDigest).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    const { w } = await mountDigest()
    const shell = useShellStore()
    shell.corpusRevision = (shell.corpusRevision ?? 0) + 1
    await flushPromises()
    second.resolve(digestOf({ rows: [row(7)] }))
    await flushPromises()
    first.resolve(digestOf({ rows: [row(8), row(9)] }))
    await flushPromises()
    expect(recentRows(w)).toHaveLength(1)
  })
})

/**
 * The real cause of that flake, from the failed attempt's own page snapshot: the Digest tab showed
 * "NetworkError when attempting to fetch resource." A transient network failure (a fetch
 * ``TypeError``, not an HTTP status) left the Recent list empty until a reload.
 */
describe('DigestView — transient network failure', () => {
  afterEach(() => vi.useRealTimers())

  it('a network error on the first digest fetch is retried, and the rows render', async () => {
    vi.useFakeTimers()
    vi.mocked(fetchCorpusDigest).mockRejectedValueOnce(
      new TypeError('NetworkError when attempting to fetch resource.'),
    )
    const { w } = await mountDigest()
    await vi.advanceTimersByTimeAsync(5000)
    await flushPromises()
    expect(recentRows(w).length).toBeGreaterThan(0)
    expect(w.text()).not.toContain('NetworkError')
  })

  it('a retry whose load was superseded during the backoff fetches nothing more', async () => {
    vi.useFakeTimers()
    vi.mocked(fetchCorpusDigest).mockRejectedValueOnce(
      new TypeError('NetworkError when attempting to fetch resource.'),
    )
    const { w } = await mountDigest()
    // Fetch #1 failed; its retry is waiting out the 500 ms backoff. A corpus revision bump starts
    // a newer load (fetch #2), so the old chain is stale and must not fetch again.
    const shell = useShellStore()
    shell.corpusRevision = (shell.corpusRevision ?? 0) + 1
    await flushPromises()
    await vi.advanceTimersByTimeAsync(5000)
    await flushPromises()
    expect(vi.mocked(fetchCorpusDigest)).toHaveBeenCalledTimes(2)
    expect(recentRows(w).length).toBeGreaterThan(0)
  })

  it('an HTTP error is shown and not retried', async () => {
    vi.useFakeTimers()
    vi.mocked(fetchCorpusDigest).mockRejectedValueOnce(new Error('GET /api/corpus/digest failed 500'))
    const { w } = await mountDigest()
    const calls = vi.mocked(fetchCorpusDigest).mock.calls.length
    await vi.advanceTimersByTimeAsync(5000)
    await flushPromises()
    expect(vi.mocked(fetchCorpusDigest).mock.calls.length).toBe(calls)
    expect(w.text()).toContain('failed 500')
  })

  it('a network error that persists is shown after the retries, not retried forever', async () => {
    vi.useFakeTimers()
    // mountDigest installs a resolved default, so the persistent failure is queued once per try:
    // the first fetch and both retries fail.
    const netErr = () => new TypeError('NetworkError when attempting to fetch resource.')
    vi.mocked(fetchCorpusDigest)
      .mockRejectedValueOnce(netErr())
      .mockRejectedValueOnce(netErr())
      .mockRejectedValueOnce(netErr())
    const { w } = await mountDigest()
    await vi.advanceTimersByTimeAsync(30000)
    await flushPromises()
    expect(vi.mocked(fetchCorpusDigest)).toHaveBeenCalledTimes(3)
    expect(w.text()).toContain('NetworkError')
  })
})
