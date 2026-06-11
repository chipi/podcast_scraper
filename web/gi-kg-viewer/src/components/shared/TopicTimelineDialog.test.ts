// @vitest-environment happy-dom
import { mount, type VueWrapper } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The shell store fires posthog.capture; mock the SDK so nothing reaches the
// network during mount.
vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

// The dialog loads timelines through cilApi. Mock both entry points and drive
// their resolution per test.
const fetchTopicTimeline = vi.fn()
const fetchTopicTimelineMerged = vi.fn()
vi.mock('../../api/cilApi', () => ({
  fetchTopicTimeline: (...a: unknown[]) => fetchTopicTimeline(...a),
  fetchTopicTimelineMerged: (...a: unknown[]) => fetchTopicTimelineMerged(...a),
}))

import TopicTimelineDialog from './TopicTimelineDialog.vue'
import { useShellStore } from '../../stores/shell'
import type {
  CilArcEpisodeBlock,
  CilTopicTimelineResponse,
  CilTopicTimelineMergedResponse,
} from '../../api/cilApi'

// PodcastCover and HelpTip are heavy / Teleport-driven children with their own
// tests. Stub them to lightweight passthroughs so we test this dialog only.
const STUBS = {
  PodcastCover: { name: 'PodcastCover', template: '<div data-stub="cover" />' },
  HelpTip: {
    name: 'HelpTip',
    template: '<div data-stub="help-tip"><slot /></div>',
  },
}

type Exposed = {
  open: (id: string, title?: unknown) => Promise<void>
  openCluster: (ids: string[]) => Promise<void>
  close: () => void
}

const mounted: VueWrapper[] = []
afterEach(() => {
  while (mounted.length) mounted.pop()!.unmount()
  fetchTopicTimeline.mockReset()
  fetchTopicTimelineMerged.mockReset()
})

beforeEach(() => {
  setActivePinia(createPinia())
})

function makeEpisode(over: Partial<CilArcEpisodeBlock> = {}): CilArcEpisodeBlock {
  return {
    episode_id: 'ep-internal-1',
    publish_date: '2024-01-15',
    episode_title: 'A Great Episode',
    feed_title: 'My Podcast',
    episode_number: 7,
    episode_image_url: null,
    episode_image_local_relpath: null,
    feed_image_url: null,
    feed_image_local_relpath: null,
    insights: [{ properties: { text: 'An insight about the topic' } }],
    ...over,
  }
}

function singleResponse(
  episodes: CilArcEpisodeBlock[],
): CilTopicTimelineResponse {
  return { path: '/corpus', topic_id: 't:1', episodes }
}

function mergedResponse(
  episodes: CilArcEpisodeBlock[],
): CilTopicTimelineMergedResponse {
  return { path: '/corpus', topic_ids: ['t:1', 't:2'], episodes }
}

function mountDialog(corpusPath = '/corpus') {
  const store = useShellStore()
  store.corpusPath = corpusPath
  const w = mount(TopicTimelineDialog, {
    attachTo: document.body,
    global: { stubs: STUBS },
  })
  mounted.push(w)
  return w
}

function exposed(w: VueWrapper): Exposed {
  return w.vm as unknown as Exposed
}

const dialogEl = (w: VueWrapper) =>
  w.get('[data-testid="topic-timeline-dialog"]').element as HTMLDialogElement

const settle = async (w: VueWrapper) => {
  await Promise.resolve()
  await w.vm.$nextTick()
  await Promise.resolve()
  await w.vm.$nextTick()
}

describe('TopicTimelineDialog', () => {
  it('renders a closed dialog with the default entity title before open()', () => {
    const w = mountDialog()
    expect(dialogEl(w).open).toBe(false)
    // Default singleTitleSpec is { entity, 'Topic' } → "Topic timeline".
    expect(w.get('#topic-timeline-title').text()).toBe('Topic timeline')
  })

  it('opens the modal and renders episodes from a single-topic timeline', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1', { variant: 'entity', entityLabel: 'Bitcoin' })
    await settle(w)
    expect(dialogEl(w).open).toBe(true)
    expect(fetchTopicTimeline).toHaveBeenCalledWith('/corpus', 't:1')
    const episodes = w.get('[data-testid="topic-timeline-episodes"]')
    expect(episodes.text()).toContain('A Great Episode')
    expect(episodes.text()).toContain('An insight about the topic')
  })

  it('derives an entity heading from the title spec', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1', { variant: 'entity', entityLabel: 'bitcoin' })
    await settle(w)
    expect(w.get('#topic-timeline-title').text()).toBe('Bitcoin timeline')
  })

  it('uses a plain "Timeline" heading for the plain title variant', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1', { variant: 'plain' })
    await settle(w)
    expect(w.get('#topic-timeline-title').text()).toBe('Timeline')
  })

  it('shows a loading state while the fetch is in flight', async () => {
    let resolve!: (v: CilTopicTimelineResponse) => void
    fetchTopicTimeline.mockReturnValue(
      new Promise<CilTopicTimelineResponse>((r) => {
        resolve = r
      }),
    )
    const w = mountDialog()
    void exposed(w).open('t:1')
    await w.vm.$nextTick()
    expect(w.find('[data-testid="topic-timeline-loading"]').exists()).toBe(true)
    resolve(singleResponse([makeEpisode()]))
    await settle(w)
    expect(w.find('[data-testid="topic-timeline-loading"]').exists()).toBe(false)
  })

  it('renders the empty state when no episodes match', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([]))
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    expect(w.find('[data-testid="topic-timeline-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="topic-timeline-episodes"]').exists()).toBe(false)
  })

  it('renders the error state when the fetch rejects', async () => {
    fetchTopicTimeline.mockRejectedValue(new Error('server exploded'))
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    const err = w.get('[data-testid="topic-timeline-error"]')
    expect(err.text()).toContain('server exploded')
  })

  it('reports a missing-corpus-path error without calling the API', async () => {
    const w = mountDialog('') // blank corpus path
    await exposed(w).open('t:1')
    await settle(w)
    expect(fetchTopicTimeline).not.toHaveBeenCalled()
    expect(w.get('[data-testid="topic-timeline-error"]').text()).toContain(
      'Set a corpus path',
    )
  })

  it('reports a missing-topic-id error when opened with a blank id', async () => {
    const w = mountDialog()
    await exposed(w).open('   ')
    await settle(w)
    expect(fetchTopicTimeline).not.toHaveBeenCalled()
    expect(w.get('[data-testid="topic-timeline-error"]').text()).toContain(
      'Missing topic id',
    )
  })

  it('opens a cluster timeline via openCluster and dedupes the ids', async () => {
    fetchTopicTimelineMerged.mockResolvedValue(mergedResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).openCluster(['t:1', 't:1', 't:2', '  '])
    await settle(w)
    expect(fetchTopicTimelineMerged).toHaveBeenCalledWith('/corpus', ['t:1', 't:2'])
    expect(w.get('#topic-timeline-title').text()).toBe('Cluster timeline')
    expect(w.find('[data-testid="topic-timeline-episodes"]').exists()).toBe(true)
  })

  it('shows the topic id sr-only span for a single timeline', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:abc')
    await settle(w)
    const idEl = w.get('[data-testid="topic-timeline-topic-id"]')
    expect(idEl.text()).toContain('Topic id: t:abc')
  })

  it('sorts episodes newest-first by default and flips on the Oldest-first button', async () => {
    const older = makeEpisode({
      episode_id: 'old',
      episode_title: 'Older Ep',
      publish_date: '2023-01-01',
    })
    const newer = makeEpisode({
      episode_id: 'new',
      episode_title: 'Newer Ep',
      publish_date: '2024-06-01',
    })
    fetchTopicTimeline.mockResolvedValue(singleResponse([older, newer]))
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)

    const titleAt = (i: number) =>
      w.get(`[data-testid="topic-timeline-episode-title-${i}"]`).text()
    // Default desc → newer first.
    expect(titleAt(0)).toBe('Newer Ep')
    expect(titleAt(1)).toBe('Older Ep')

    await w.get('[data-testid="topic-timeline-sort-asc"]').trigger('click')
    expect(titleAt(0)).toBe('Older Ep')
    expect(titleAt(1)).toBe('Newer Ep')
  })

  it('falls back to "Episode N" heading when the title is missing', async () => {
    fetchTopicTimeline.mockResolvedValue(
      singleResponse([
        makeEpisode({ episode_title: null, feed_title: null, episode_number: 12 }),
      ]),
    )
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    expect(
      w.get('[data-testid="topic-timeline-episode-title-0"]').text(),
    ).toBe('Episode 12')
  })

  it('shows a "Date unknown" line when an episode has no publish date', async () => {
    fetchTopicTimeline.mockResolvedValue(
      singleResponse([makeEpisode({ publish_date: null })]),
    )
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    expect(w.get('[data-testid="topic-timeline-episode-0"]').text()).toContain(
      'Date unknown',
    )
  })

  it('renders one insight line per insight', async () => {
    fetchTopicTimeline.mockResolvedValue(
      singleResponse([
        makeEpisode({
          insights: [
            { properties: { text: 'First insight' } },
            { properties: { text: 'Second insight' } },
          ],
        }),
      ]),
    )
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    expect(w.find('[data-testid="topic-timeline-insight-0-0"]').text()).toContain(
      'First insight',
    )
    expect(w.find('[data-testid="topic-timeline-insight-0-1"]').text()).toContain(
      'Second insight',
    )
  })

  it('closes via the Close button', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    expect(dialogEl(w).open).toBe(true)
    await w.get('[data-testid="topic-timeline-close"]').trigger('click')
    expect(dialogEl(w).open).toBe(false)
  })

  it('closes via the exposed close() method', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    exposed(w).close()
    await w.vm.$nextTick()
    expect(dialogEl(w).open).toBe(false)
  })

  it('closes on a backdrop click that targets the dialog element itself', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    await w.get('[data-testid="topic-timeline-dialog"]').trigger('click')
    expect(dialogEl(w).open).toBe(false)
  })

  it('does not close on a click from inside the dialog content', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1')
    await settle(w)
    await w.get('[data-testid="topic-timeline-body"]').trigger('click')
    expect(dialogEl(w).open).toBe(true)
  })

  it('reopening as a different mode resets prior episodes/title', async () => {
    fetchTopicTimeline.mockResolvedValue(singleResponse([makeEpisode()]))
    const w = mountDialog()
    await exposed(w).open('t:1', { variant: 'entity', entityLabel: 'Bitcoin' })
    await settle(w)
    expect(w.get('#topic-timeline-title').text()).toBe('Bitcoin timeline')

    fetchTopicTimelineMerged.mockResolvedValue(mergedResponse([]))
    await exposed(w).openCluster(['t:9'])
    await settle(w)
    expect(w.get('#topic-timeline-title').text()).toBe('Cluster timeline')
    expect(w.find('[data-testid="topic-timeline-empty"]').exists()).toBe(true)
  })
})
