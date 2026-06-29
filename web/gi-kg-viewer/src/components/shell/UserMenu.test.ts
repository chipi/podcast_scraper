// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const logout = vi.fn()
vi.mock('../../api/authApi', () => ({
  logout: (...a: unknown[]) => logout(...a),
  loginUrl: () => '/api/app/auth/login?grant=creator',
  getMe: vi.fn(),
}))

import { useAuthStore } from '../../stores/auth'
import UserMenu from './UserMenu.vue'

beforeEach(() => {
  setActivePinia(createPinia())
  useAuthStore().user = {
    user_id: 'u1',
    email: 'boss@x.io',
    name: 'Boss',
    role: 'admin',
    disabled: false,
  }
})
afterEach(() => logout.mockReset())

describe('UserMenu', () => {
  it('is closed until clicked, then shows name, email, and role', async () => {
    const w = mount(UserMenu)
    expect(w.find('[data-testid="user-menu"]').exists()).toBe(false)
    expect(w.find('[data-testid="user-menu-button"]').text()).toBe('B') // initial
    await w.find('[data-testid="user-menu-button"]').trigger('click')
    expect(w.find('[data-testid="user-menu"]').text()).toContain('boss@x.io')
    expect(w.find('[data-testid="user-menu-role"]').text()).toBe('Admin')
  })

  it('the backdrop closes the menu', async () => {
    const w = mount(UserMenu)
    await w.find('[data-testid="user-menu-button"]').trigger('click')
    expect(w.find('[data-testid="user-menu"]').exists()).toBe(true)
    await w.find('[data-testid="user-menu-backdrop"]').trigger('click')
    expect(w.find('[data-testid="user-menu"]').exists()).toBe(false)
  })

  it('sign out calls the logout API', async () => {
    const w = mount(UserMenu)
    await w.find('[data-testid="user-menu-button"]').trigger('click')
    await w.find('[data-testid="user-menu-signout"]').trigger('click')
    expect(logout).toHaveBeenCalled()
  })
})
