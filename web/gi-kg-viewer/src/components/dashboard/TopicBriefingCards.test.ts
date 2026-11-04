// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { CorpusDigestResponse, CorpusDigestTopicBand } from '../../api/digestApi'
import { fetchCrossShow } from '../../api/relationalApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import TopicBriefingCards from './TopicBriefingCards.vue'

vi.mock('../../api/relationalApi', () => ({ fetchCrossShow: vi.fn() }))
vi.mock('../../lib/analytics', () => ({ track: vi.fn() }))

const mockCross = vi.mocked(fetchCrossShow)

type Hit = { score: number | null; feed?: string; preview?: string | null; title?: string }

function band(id: string, hits: Hit[], graph?: string): CorpusDigestTopicBand {
  return {
    topic_id: id,
    label: `Topic ${id}`,
    query: id,
    graph_topic_id: graph,
    hits: hits.map((x, i) => ({
      metadata_relative_path: null,
      episode_title: x.title ?? `Episode ${id}-${i}`,
      feed_id: x.feed ?? 'f1',
      score: x.score,
      summary_preview: x.preview,
    })),
  }
}

function digest(topics: CorpusDigestTopicBand[]): CorpusDigestResponse {
  return {
    path: '/c',
    window: '7d',
    window_start_utc: '',
    window_end_utc: '',
    compact: true,
    rows: [],
    topics,
    topics_unavailable_reason: null,
  }
}

function ready(): void {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
}

describe('TopicBriefingCards', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mockCross.mockReset()
    mockCross.mockResolvedValue({ groups: {} } as never)
  })

  it('renders nothing for a null digest or when no band has hits', () => {
    expect(mount(TopicBriefingCards, { props: { digest: null } }).find('section').exists()).toBe(false)
    const w = mount(TopicBriefingCards, { props: { digest: digest([band('a', [])]) } })
    expect(w.find('section').exists()).toBe(false)
  })

  it('shows the best hit preview with its score and the episode count', () => {
    const w = mount(TopicBriefingCards, {
      props: {
        digest: digest([
          band('a', [
            { score: 0.5, preview: 'low preview' },
            { score: 0.9, preview: '  best preview  ' },
          ]),
        ]),
      },
    })
    const card = w.get('[data-testid="topic-briefing-card"]')
    expect(card.text()).toContain('Topic a')
    expect(card.text()).toContain('2 ep')
    expect(card.text()).toContain('0.90')
    expect(card.text()).toContain('best preview')
    expect(card.text()).not.toContain('low preview')
  })

  it('falls back to the episode title, and shows no segment when both are blank', () => {
    const w = mount(TopicBriefingCards, {
      props: {
        digest: digest([
          band('a', [{ score: 0.4, preview: '  ', title: 'Fallback Title' }]),
          band('b', [{ score: 0.3, preview: null, title: '   ' }]),
        ]),
      },
    })
    const cards = w.findAll('[data-testid="topic-briefing-card"]')
    expect(cards[0].text()).toContain('Fallback Title')
    expect(cards[1].find('p').exists()).toBe(false)
  })

  it('treats a missing score as 0.00', () => {
    const w = mount(TopicBriefingCards, {
      props: { digest: digest([band('a', [{ score: null, preview: 'p' }])]) },
    })
    expect(w.text()).toContain('0.00')
  })

  it('ranks by retrieval signal (score, density, show breadth) and keeps four cards', () => {
    const w = mount(TopicBriefingCards, {
      props: {
        digest: digest([
          band('low', [{ score: 0.1 }]),
          band('top', [
            { score: 0.9, feed: 'x' },
            { score: 0.8, feed: 'y' },
            { score: 0.7, feed: 'z' },
          ]),
          band('mid', [{ score: 0.5 }]),
          band('d', [{ score: 0.4 }]),
          band('e', [{ score: 0.3 }]),
        ]),
      },
    })
    const labels = w.findAll('[data-testid="topic-briefing-card"]').map((c) => c.text())
    expect(labels).toHaveLength(4)
    expect(labels[0]).toContain('Topic top')
    expect(labels[1]).toContain('Topic mid')
    expect(labels.join(' ')).not.toContain('Topic low')
  })

  it('renders a link only for topics mapped to a graph node, and focuses the topic on click', async () => {
    const w = mount(TopicBriefingCards, {
      props: {
        digest: digest([
          band('mapped', [{ score: 0.9 }], '  topic:ai  '),
          band('plain', [{ score: 0.2 }]),
        ]),
      },
    })
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusTopic').mockImplementation(() => {})
    const links = w.findAll('[data-testid="topic-briefing-card-link"]')
    expect(links).toHaveLength(1)
    expect(links[0].attributes('title')).toBe('Open Topic panel for Topic mapped')
    await links[0].trigger('click')
    expect(spy).toHaveBeenCalledWith('topic:ai')
  })

  it('loads the cross-show insight for mapped topics when the API is healthy', async () => {
    ready()
    mockCross.mockResolvedValue({
      groups: { showA: [], showB: [{ text: 'Cross-show insight text' }], showC: [{ text: 'other' }] },
    } as never)
    const w = mount(TopicBriefingCards, {
      props: { digest: digest([band('a', [{ score: 0.9 }], 'topic:ai')]) },
    })
    await flushPromises()
    expect(mockCross).toHaveBeenCalledWith('/corpus', 'topic:ai')
    const cs = w.get('[data-testid="topic-briefing-card-cross-show"]')
    expect(cs.text()).toContain('Across 3 shows:')
    expect(cs.text()).toContain('Cross-show insight text')
  })

  it('shows no cross-show line when groups are empty or the request fails', async () => {
    ready()
    mockCross.mockRejectedValueOnce(new Error('nope'))
    const w = mount(TopicBriefingCards, {
      props: {
        digest: digest([
          band('a', [{ score: 0.9 }], 'topic:a'),
          band('b', [{ score: 0.8 }], 'topic:b'),
        ]),
      },
    })
    await flushPromises()
    expect(mockCross).toHaveBeenCalledTimes(2)
    expect(w.find('[data-testid="topic-briefing-card-cross-show"]').exists()).toBe(false)
  })

  it('does not call the relational API without a corpus path, health, or graph topic id', async () => {
    mount(TopicBriefingCards, {
      props: { digest: digest([band('a', [{ score: 0.9 }], 'topic:a')]) },
    })
    await flushPromises()
    expect(mockCross).not.toHaveBeenCalled()
    ready()
    mount(TopicBriefingCards, {
      props: { digest: digest([band('b', [{ score: 0.9 }])]) },
    })
    await flushPromises()
    expect(mockCross).not.toHaveBeenCalled()
  })

  it('fetches each topic once even when the digest object is replaced', async () => {
    ready()
    const w = mount(TopicBriefingCards, {
      props: { digest: digest([band('a', [{ score: 0.9 }], 'topic:a')]) },
    })
    await flushPromises()
    await w.setProps({ digest: digest([band('a', [{ score: 0.9 }], 'topic:a')]) })
    await flushPromises()
    expect(mockCross).toHaveBeenCalledTimes(1)
  })
})
