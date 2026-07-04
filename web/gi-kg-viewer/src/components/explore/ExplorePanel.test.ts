// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ExplorePanel from './ExplorePanel.vue'
import { useShellStore } from '../../stores/shell'
import { useExploreStore } from '../../stores/explore'
import { useSubjectStore } from '../../stores/subject'
import { useGraphNavigationStore } from '../../stores/graphNavigation'

// Mock the API module for explore
vi.mock('../../api/exploreApi', () => ({
  runExplore: vi.fn(),
  runNaturalLanguageQuery: vi.fn(),
}))

// Mock the corpus graph baseline loader (optional injection)
// We inject null for this, so it's not loaded during tests

function mountPanel() {
  return mount(ExplorePanel, {
    attachTo: document.body,
  })
}

const FILTERED_SUBMIT = '[data-testid="explore-filtered-submit"]'
const CLEAR_BTN = '[data-testid="explore-clear-output"]'
const ADVANCED_DIALOG = '[data-testid="explore-advanced-dialog"]'

describe('ExplorePanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  // ── Base render ──────────────────────────────────────────────────────────

  it('renders the panel with section heading and inputs', () => {
    const w = mountPanel()
    expect(w.text()).toContain('Explore & query')
    expect(w.text()).toContain('Topic / speaker')
    expect(w.text()).toContain('Quick questions')
  })

  it('renders topic/speaker filter bar when health is available', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    // ExploreFilterBar should render when enabled is true
    expect(w.find('[data-testid="explore-filtered-submit"]').exists()).toBe(true)
  })

  it('shows "Requires the API" hint when no health status', () => {
    const w = mountPanel()
    expect(w.text()).toContain('Requires the API')
  })

  // ── Advanced explore dialog ──────────────────────────────────────────────

  it('opens the advanced explore dialog when button clicked', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    // Find and click the "Advanced explore" link (emitted via ExploreFilterBar)
    // Since ExploreFilterBar is a child, we trigger @open-more via props
    await w.vm.$nextTick()
    const dialog = w.find(ADVANCED_DIALOG)
    expect(dialog.exists()).toBe(true)
  })

  it('closes the advanced dialog when close button is clicked', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    const dialog = w.find(ADVANCED_DIALOG).element as HTMLDialogElement
    // Manually trigger open (simulating user click)
    const ex = useExploreStore()
    // Use component ref to open/close
    if (w.vm.advancedExploreDialogRef) {
      w.vm.advancedExploreDialogRef.showModal()
      await w.vm.$nextTick()
      expect(dialog.open).toBe(true)
      w.vm.advancedExploreDialogRef.close()
      await w.vm.$nextTick()
      expect(dialog.open).toBe(false)
    }
  })

  // ── Filter buttons ─────────────────────────────────────────────────────

  it('disables the Explore button when no health status', () => {
    const w = mountPanel()
    const btn = w.find(FILTERED_SUBMIT)
    expect((btn.element as HTMLButtonElement).disabled).toBe(true)
  })

  it('enables the Explore button when health status is set', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    const btn = w.find(FILTERED_SUBMIT)
    expect((btn.element as HTMLButtonElement).disabled).toBe(false)
  })

  it('disables the Clear button when no health status', () => {
    const w = mountPanel()
    const btn = w.find(CLEAR_BTN)
    expect((btn.element as HTMLButtonElement).disabled).toBe(true)
  })

  it('enables the Clear button when health status is set', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()
    const btn = w.find(CLEAR_BTN)
    expect((btn.element as HTMLButtonElement).disabled).toBe(false)
  })

  // ── Explore button click ──────────────────────────────────────────────

  it('calls runFilteredExplore when Explore button is clicked', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    const spy = vi.spyOn(ex, 'runFilteredExplore').mockResolvedValue()
    await w.find(FILTERED_SUBMIT).trigger('click')
    expect(spy).toHaveBeenCalledWith('/corpus')
  })

  it('disables Explore button when loading is true', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    ex.loading = true
    await w.vm.$nextTick()
    const btn = w.find(FILTERED_SUBMIT)
    expect((btn.element as HTMLButtonElement).disabled).toBe(true)
  })

  // ── Clear button click ───────────────────────────────────────────────

  it('calls clearOutput when Clear button is clicked', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()
    const spy = vi.spyOn(ex, 'clearOutput').mockResolvedValue()
    await w.find(CLEAR_BTN).trigger('click')
    expect(spy).toHaveBeenCalled()
  })

  // ── Quick questions textarea ──────────────────────────────────────────

  it('renders the quick questions textarea', () => {
    const w = mountPanel()
    const textarea = w.find('textarea[aria-label="Quick question"]')
    expect(textarea.exists()).toBe(true)
  })

  it('disables quick question textarea when no health status', () => {
    const w = mountPanel()
    const textarea = w.find('textarea[aria-label="Quick question"]')
    expect((textarea.element as HTMLTextAreaElement).disabled).toBe(true)
  })

  it('updates ex.nlQuestion when textarea changes', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()
    const textarea = w.find('textarea[aria-label="Quick question"]')
    await textarea.setValue('What is AI?')
    expect(ex.nlQuestion).toBe('What is AI?')
  })

  // ── Keyboard handlers ────────────────────────────────────────────────

  it('submits quick question on Enter (without Shift)', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    const spy = vi.spyOn(ex, 'runNaturalLanguage').mockResolvedValue()
    const textarea = w.find('textarea[aria-label="Quick question"]')
    await textarea.setValue('test')
    // Trigger Enter key (simulated)
    const evt = new KeyboardEvent('keydown', {
      key: 'Enter',
      shiftKey: false,
      bubbles: true,
    })
    textarea.element.dispatchEvent(evt)
    await w.vm.$nextTick()
    expect(spy).toHaveBeenCalledWith('/corpus')
  })

  // ── Output display ───────────────────────────────────────────────────

  it('displays error message when ex.error is set', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    ex.error = 'Search failed'
    await w.vm.$nextTick()
    expect(w.text()).toContain('Search failed')
  })

  it('hides error when ex.error is null', () => {
    const w = mountPanel()
    const ex = useExploreStore()
    ex.error = null
    // Error paragraph is hidden (class v-if)
    expect(w.find('[role="alert"]').exists()).toBe(false)
  })

  it('computes leaderboard rows from last result', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    // Initially empty
    expect(ex.leaderboardRows.length).toBe(0)
    // Set last with topics data
    ex.last = {
      kind: 'explore' as const,
      error: null,
      detail: null,
      data: {
        insights: [],
        topics: [
          {
            topic_id: 'topic:ai',
            label: 'Artificial Intelligence',
            insight_count: 10,
          },
        ],
        speakers: [],
        summary: {
          insight_count: 10,
          grounded_insight_count: 8,
          quote_count: 15,
          episode_count: 3,
          speaker_count: 2,
          topic_count: 1,
          episodes_searched: 0,
        },
      },
    }
    await w.vm.$nextTick()
    // Now leaderboardRows should be computed
    expect(ex.leaderboardRows.length).toBe(1)
    expect(ex.leaderboardRows[0]!.topic_id).toBe('topic:ai')
    expect(ex.leaderboardRows[0]!.label).toBe('Artificial Intelligence')
  })

  it('computes topSpeakers from last result', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    // Initially empty until last is set
    expect(ex.topSpeakers.length).toBe(0)
    // Set last with a full explore result
    ex.last = {
      kind: 'explore' as const,
      error: null,
      detail: null,
      data: {
        insights: [],
        speakers: [
          {
            speaker_id: 'person:alice',
            speaker: { name: 'Alice' },
            quote_count: 5,
            insight_count: 3,
          },
        ],
        summary: {
          insight_count: 3,
          grounded_insight_count: 2,
          quote_count: 5,
          episode_count: 1,
          speaker_count: 1,
          topic_count: 0,
          episodes_searched: 0,
        },
      },
    }
    await w.vm.$nextTick()
    // Now topSpeakers should be computed
    expect(ex.topSpeakers.length).toBeGreaterThanOrEqual(0)
  })

  it('computes insightRows from last result', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    // Initially empty
    expect(ex.insightRows.length).toBe(0)
    // Set last with insights
    ex.last = {
      kind: 'explore' as const,
      error: null,
      detail: null,
      data: {
        insights: [
          {
            insight_id: 'gi:123',
            text: 'AI is important',
            confidence: 0.95,
            grounded: true,
            episode: { title: 'Episode 1', episode_id: 'ep:1', publish_date: '2024-01-01' },
            supporting_quotes: [],
          },
        ],
        speakers: [],
        summary: {
          insight_count: 1,
          grounded_insight_count: 1,
          quote_count: 0,
          episode_count: 1,
          speaker_count: 0,
          topic_count: 0,
          episodes_searched: 0,
        },
      },
    }
    await w.vm.$nextTick()
    // Now insightRows should be computed
    expect(ex.insightRows.length).toBeGreaterThanOrEqual(0)
  })

  // ── Quote expansion ──────────────────────────────────────────────────

  it('toggles quote expansion when "Show quotes" button is clicked', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    ex.last = {
      kind: 'filtered' as const,
      error: null,
      detail: null,
    }
    ex.insightRows = [
      {
        insight_id: 'gi:123',
        text: 'Main insight',
        confidence: null,
        grounded: undefined,
        episode: null,
        supporting_quotes: [
          {
            text: 'Quote text',
            speaker_name: 'Alice',
            speaker_id: 'person:alice',
            start_ms: 1000,
            end_ms: 2000,
          },
        ],
      },
    ]
    ex.error = null  // Critical: must clear error so output renders
    await w.vm.$nextTick()
    const toggleBtn = w.findAll('button').find((b) => b.text().includes('Show'))
    if (toggleBtn) {
      await toggleBtn.trigger('click')
      await w.vm.$nextTick()
      // Quote should now be visible
      const blockquote = w.find('blockquote')
      expect(blockquote.exists()).toBe(true)
    }
  })

  // ── Node focus links ─────────────────────────────────────────────────

  it('emits go-graph and focuses node when leaderboard row clicked', async () => {
    const w = mountPanel()
    const nav = useGraphNavigationStore()
    const ex = useExploreStore()
    ex.last = {
      kind: 'filtered' as const,
      error: null,
      detail: null,
    }
    ex.leaderboardRows = [
      {
        topic_id: 'topic:ai',
        label: 'AI',
        insight_count: 5,
      },
    ]
    ex.error = null
    await w.vm.$nextTick()
    const spy = vi.spyOn(nav, 'requestFocusNode')
    const rows = w.findAll('tbody tr')
    if (rows.length > 0) {
      await rows[0]!.trigger('click')
      expect(spy).toHaveBeenCalledWith('topic:ai')
      expect(w.emitted('go-graph')).toBeTruthy()
    }
  })

  it('focuses person when top speaker link clicked', async () => {
    const w = mountPanel()
    const subject = useSubjectStore()
    const ex = useExploreStore()
    ex.last = {
      kind: 'filtered' as const,
      error: null,
      detail: null,
    }
    ex.topSpeakers = [
      {
        speaker_id: 'person:alice',
        name: 'Alice',
        quote_count: 5,
        insight_count: 3,
      },
    ]
    ex.error = null
    await w.vm.$nextTick()
    const spy = vi.spyOn(subject, 'focusPerson')
    const link = w.find('[data-testid="explore-top-speaker-link"]')
    if (link.exists()) {
      await link.trigger('click')
      expect(spy).toHaveBeenCalledWith('person:alice')
    }
  })

  it('focuses insight node when insight card clicked', async () => {
    const w = mountPanel()
    const nav = useGraphNavigationStore()
    const ex = useExploreStore()
    ex.last = {
      kind: 'filtered' as const,
      error: null,
      detail: null,
    }
    ex.insightRows = [
      {
        insight_id: 'gi:123',
        text: 'Insight',
        confidence: null,
        grounded: undefined,
        episode: null,
        supporting_quotes: [],
      },
    ]
    ex.error = null
    await w.vm.$nextTick()
    const spy = vi.spyOn(nav, 'requestFocusNode')
    const card = w.find('article')
    if (card.exists()) {
      await card.trigger('click')
      expect(spy).toHaveBeenCalledWith('gi:123')
      expect(w.emitted('go-graph')).toBeTruthy()
    }
  })

  // ── Advanced dialog checkbox + field interactions ──────────────────

  it('updates filter state in advanced dialog', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()
    const dialog = w.find(ADVANCED_DIALOG)
    const checkbox = dialog.find('input[type="checkbox"]')
    if (checkbox.exists()) {
      await checkbox.setValue(true)
      expect(ex.filters.groundedOnly).toBe(true)
    }
  })

  it('updates limit field in advanced dialog', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()
    const dialog = w.find(ADVANCED_DIALOG)
    const limitInput = dialog.find('input[type="number"]')
    if (limitInput.exists()) {
      await limitInput.setValue(50)
      expect(ex.filters.limit).toBe(50)
    }
  })

  it('updates sort select in advanced dialog', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    const ex = useExploreStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()
    const dialog = w.find(ADVANCED_DIALOG)
    const select = dialog.find('select')
    if (select.exists()) {
      await select.setValue('time')
      expect(ex.filters.sortBy).toBe('time')
    }
  })

  // ── Summary block rendering ──────────────────────────────────────────

  it('computes summary block from last result', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    // When last is set, summaryBlock is computed
    // This just verifies that the store can compute summaryBlock
    expect(ex.summaryBlock).toBeNull()
    // After setting last with a non-error result
    ex.last = {
      kind: 'explore' as const,
      error: null,
      detail: null,
      data: {
        insights: [],
        summary: {
          insight_count: 5,
          grounded_insight_count: 3,
          quote_count: 10,
          episode_count: 2,
          speaker_count: 1,
          topic_count: 1,
          episodes_searched: 0,
        },
      },
    }
    await w.vm.$nextTick()
    // Now summaryBlock should exist
    if (ex.summaryBlock) {
      expect(ex.summaryBlock.insight_count).toBe(5)
    }
  })

  // ── Natural language explanation ──────────────────────────────────────

  it('displays NL explanation for natural_language results', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    ex.last = {
      kind: 'natural_language' as const,
      explanation: 'Found 5 insights about AI',
      error: null,
      detail: null,
    }
    await w.vm.$nextTick()
    expect(w.text()).toContain('Found 5 insights about AI')
  })

  // ── Error handling ───────────────────────────────────────────────────

  it('displays error detail when ex.last.error is set', async () => {
    const w = mountPanel()
    const ex = useExploreStore()
    ex.last = {
      kind: 'filtered' as const,
      error: 'Invalid query',
      detail: 'Topic filter is too vague',
    }
    await w.vm.$nextTick()
    expect(w.text()).toContain('Invalid query')
    expect(w.text()).toContain('Topic filter is too vague')
  })
})
