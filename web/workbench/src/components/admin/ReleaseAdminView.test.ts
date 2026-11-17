// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ReleaseAdminView from './ReleaseAdminView.vue'
import * as authApi from '../../api/authApi'

vi.mock('../../api/authApi', () => ({
  fetchRelease: vi.fn(),
  saveRelease: vi.fn(),
}))
const fetchRel = vi.mocked(authApi.fetchRelease)
const saveRel = vi.mocked(authApi.saveRelease)

const DEPLOY_ONLY = { player_version: '1.0.1', override: null, deploy_default: '1.0.1' }

describe('ReleaseAdminView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    fetchRel.mockResolvedValue({ ...DEPLOY_ONLY })
    saveRel.mockImplementation(async (v) => ({
      player_version: v ?? '1.0.1',
      override: v,
      deploy_default: '1.0.1',
    }))
  })

  it('shows what is served now and the deploy default', async () => {
    const w = mount(ReleaseAdminView)
    await flushPromises()
    expect(w.get('[data-testid="release-served"]').text()).toBe('1.0.1')
    expect(w.get('[data-testid="release-deploy-default"]').text()).toBe('1.0.1')
    // Nothing to clear yet.
    expect(w.get('[data-testid="release-clear"]').attributes('disabled')).toBeDefined()
  })

  it('saves an override and shows it as served', async () => {
    const w = mount(ReleaseAdminView)
    await flushPromises()
    await w.get('[data-testid="release-input"]').setValue('1.0.2')
    await w.get('[data-testid="release-save"]').trigger('click')
    await flushPromises()
    expect(saveRel).toHaveBeenCalledWith('1.0.2')
    expect(w.get('[data-testid="release-served"]').text()).toBe('1.0.2')
    expect(w.find('[data-testid="release-saved"]').exists()).toBe(true)
  })

  it('refuses a malformed version without calling the server', async () => {
    const w = mount(ReleaseAdminView)
    await flushPromises()
    await w.get('[data-testid="release-input"]').setValue('v1.0.2')
    await w.get('[data-testid="release-save"]').trigger('click')
    await flushPromises()
    expect(saveRel).not.toHaveBeenCalled()
    expect(w.get('[data-testid="release-error"]').text()).toContain('dotted number')
  })

  it('"Use deploy default" clears the override', async () => {
    fetchRel.mockResolvedValue({ player_version: '1.0.2', override: '1.0.2', deploy_default: '1.0.1' })
    const w = mount(ReleaseAdminView)
    await flushPromises()
    await w.get('[data-testid="release-clear"]').trigger('click')
    await flushPromises()
    expect(saveRel).toHaveBeenCalledWith(null)
    expect(w.get('[data-testid="release-served"]').text()).toBe('1.0.1')
  })

  it('surfaces a load error', async () => {
    fetchRel.mockRejectedValue(new Error('nope'))
    const w = mount(ReleaseAdminView)
    await flushPromises()
    expect(w.get('[data-testid="release-error"]').text()).toContain('nope')
  })
})
