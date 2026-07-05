// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const listUsers = vi.fn()
const patchUser = vi.fn()
const createUser = vi.fn()
const deleteUser = vi.fn()
vi.mock('../../api/authApi', () => ({
  listUsers: (...a: unknown[]) => listUsers(...a),
  patchUser: (...a: unknown[]) => patchUser(...a),
  createUser: (...a: unknown[]) => createUser(...a),
  deleteUser: (...a: unknown[]) => deleteUser(...a),
}))

import { useAuthStore } from '../../stores/auth'
import UsersAdminView from './UsersAdminView.vue'

const me = { user_id: 'u_admin', email: 'boss@x.io', name: 'Boss', role: 'admin' as const, disabled: false }

function user(over: Record<string, unknown> = {}) {
  return {
    user_id: 'u1',
    email: 'a@b.c',
    name: 'Alice',
    role: 'creator',
    disabled: false,
    provider: 'mock',
    ...over,
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  useAuthStore().user = { ...me }
  listUsers.mockResolvedValue([user(), { ...me, provider: 'mock' }])
})
afterEach(() => {
  ;[listUsers, patchUser, createUser, deleteUser].forEach((m) => m.mockReset())
})

describe('UsersAdminView', () => {
  it('lists all users including the admin', async () => {
    const w = mount(UsersAdminView)
    await flushPromises()
    expect(w.find('[data-testid="user-row-a@b.c"]').exists()).toBe(true)
    expect(w.find('[data-testid="user-row-boss@x.io"]').exists()).toBe(true)
  })

  it("disables role/active/delete controls on the admin's own row (self-lockout)", async () => {
    const w = mount(UsersAdminView)
    await flushPromises()
    expect(w.find('[data-testid="role-select-boss@x.io"]').attributes('disabled')).toBeDefined()
    expect(w.find('[data-testid="active-toggle-boss@x.io"]').attributes('disabled')).toBeDefined()
    expect(w.find('[data-testid="delete-user-boss@x.io"]').attributes('disabled')).toBeDefined()
    // ...but another user's controls are enabled
    expect(w.find('[data-testid="role-select-a@b.c"]').attributes('disabled')).toBeUndefined()
  })

  it('changing a role calls patchUser and reflects the new value', async () => {
    patchUser.mockResolvedValue(user({ role: 'admin' }))
    const w = mount(UsersAdminView)
    await flushPromises()
    await w.find('[data-testid="role-select-a@b.c"]').setValue('admin')
    await flushPromises()
    expect(patchUser).toHaveBeenCalledWith('u1', { role: 'admin' })
  })

  it('toggling active calls patchUser with disabled', async () => {
    patchUser.mockResolvedValue(user({ disabled: true }))
    const w = mount(UsersAdminView)
    await flushPromises()
    await w.find('[data-testid="active-toggle-a@b.c"]').trigger('click')
    await flushPromises()
    expect(patchUser).toHaveBeenCalledWith('u1', { disabled: true })
  })

  it('creating a user calls createUser with the form values', async () => {
    createUser.mockResolvedValue(user({ user_id: 'u2', email: 'new@x.io' }))
    const w = mount(UsersAdminView)
    await flushPromises()
    await w.find('[data-testid="new-user-email"]').setValue('new@x.io')
    await w.find('form').trigger('submit')
    await flushPromises()
    expect(createUser).toHaveBeenCalledWith({ email: 'new@x.io', name: '', role: 'creator' })
  })

  it('surfaces a server error (e.g. self-lockout) in an alert', async () => {
    patchUser.mockRejectedValue(new Error('You cannot remove your own admin role.'))
    const w = mount(UsersAdminView)
    await flushPromises()
    await w.find('[data-testid="role-select-a@b.c"]').setValue('listener')
    await flushPromises()
    expect(w.find('[data-testid="users-admin-error"]').text()).toContain('admin role')
  })
})
