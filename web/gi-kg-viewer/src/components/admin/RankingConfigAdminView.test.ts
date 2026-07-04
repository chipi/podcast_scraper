// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import RankingConfigAdminView from './RankingConfigAdminView.vue'
import * as authApi from '../../api/authApi'

vi.mock('../../api/authApi', () => ({
  fetchRankingConfig: vi.fn(),
  saveRankingConfig: vi.fn(),
}))
const fetchCfg = vi.mocked(authApi.fetchRankingConfig)
const saveCfg = vi.mocked(authApi.saveRankingConfig)

const CFG = {
  signals: [
    { name: 'significance', enabled: true, weight: 1, params: { gi_bonus: 2 } },
    { name: 'trend_velocity', enabled: false, weight: 0.4, params: { cap: 1.5 } },
  ],
}
const clone = () => JSON.parse(JSON.stringify(CFG)) as typeof CFG

describe('RankingConfigAdminView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    fetchCfg.mockResolvedValue(clone())
    saveCfg.mockImplementation(async (c) => c)
  })

  it('lists the signals with enabled toggle, weight and params', async () => {
    const w = mount(RankingConfigAdminView)
    await flushPromises()
    expect(w.findAll('[data-testid^="ranking-signal-"]')).toHaveLength(2)
    expect(w.find('[data-testid="ranking-enabled-trend_velocity"]').exists()).toBe(true)
    expect(w.find('[data-testid="ranking-weight-trend_velocity"]').exists()).toBe(true)
    expect(w.find('[data-testid="ranking-param-trend_velocity-cap"]').exists()).toBe(true)
  })

  it('saves the edited config and shows a confirmation', async () => {
    const w = mount(RankingConfigAdminView)
    await flushPromises()
    await w.get('[data-testid="ranking-enabled-trend_velocity"]').setValue(true)
    await w.get('[data-testid="ranking-config-save"]').trigger('click')
    await flushPromises()
    expect(saveCfg).toHaveBeenCalledTimes(1)
    const sent = saveCfg.mock.calls[0][0]
    expect(sent.signals.find((s) => s.name === 'trend_velocity')?.enabled).toBe(true)
    expect(w.find('[data-testid="ranking-config-saved"]').exists()).toBe(true)
  })

  it('surfaces a load error', async () => {
    fetchCfg.mockRejectedValue(new Error('nope'))
    const w = mount(RankingConfigAdminView)
    await flushPromises()
    expect(w.find('[data-testid="ranking-config-error"]').text()).toContain('nope')
  })
})
