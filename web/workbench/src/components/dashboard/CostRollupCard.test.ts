// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import CostRollupCard from './CostRollupCard.vue'
import { useShellStore } from '../../stores/shell'
import { fetchCorpusManifest } from '../../api/corpusMetricsApi'

vi.mock('../../api/corpusMetricsApi', () => ({ fetchCorpusManifest: vi.fn() }))
const mockFetch = vi.mocked(fetchCorpusManifest)

async function mountCard() {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const w = mount(CostRollupCard)
  await flushPromises()
  await w.vm.$nextTick()
  return w
}

describe('CostRollupCard (P2.8)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mockFetch.mockReset()
  })

  it('renders total cost + run count + transcription/LLM split from cost_rollup', async () => {
    mockFetch.mockResolvedValue({
      cost_rollup: {
        total_cost_usd: 12.5,
        total_transcription_cost_usd: 4.5,
        total_llm_cost_usd: 8.0,
        run_count: 3,
      },
    })
    const w = await mountCard()
    expect(w.find('[data-testid="cost-total"]').text()).toBe('$12.50')
    expect(w.text()).toContain('over 3 runs')
    expect(w.text()).toContain('Transcription $4.50')
    expect(w.text()).toContain('LLM $8.00')
  })

  it('warns when cost appears uninstrumented (#823)', async () => {
    mockFetch.mockResolvedValue({
      cost_rollup: { total_cost_usd: 0, run_count: 2, cost_appears_uninstrumented: true },
    })
    const w = await mountCard()
    expect(w.find('[data-testid="cost-uninstrumented"]').exists()).toBe(true)
  })

  it('shows a graceful message when the manifest has no cost_rollup', async () => {
    mockFetch.mockResolvedValue({ schema_version: '1' })
    const w = await mountCard()
    expect(w.text()).toContain('No cost_rollup')
    expect(w.find('[data-testid="cost-total"]').exists()).toBe(false)
  })

  it('degrades when the manifest is unavailable (404)', async () => {
    mockFetch.mockRejectedValue(new Error('HTTP 404'))
    const w = await mountCard()
    expect(w.text()).toContain('not available')
  })
})
