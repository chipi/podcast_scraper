import { afterEach, describe, expect, it, vi } from 'vitest'

const fetchMock = vi.fn()
vi.mock('./httpClient', () => ({
  fetchWithTimeout: (...a: unknown[]) => fetchMock(...a),
}))

import { createUser, deleteUser, getAuthStatus, loginUrl, patchUser } from './authApi'

function res(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response
}

afterEach(() => fetchMock.mockReset())

describe('authApi', () => {
  it('loginUrl defaults to a creator grant; can add an ?as hint; can drop grant', () => {
    expect(loginUrl()).toBe('/api/app/auth/login?grant=creator')
    expect(loginUrl('creator', 'alice')).toBe('/api/app/auth/login?grant=creator&as=alice')
    expect(loginUrl(null)).toBe('/api/app/auth/login')
  })

  it('getAuthStatus reports enabled + user; degrades to {enabled:false} on any failure', async () => {
    fetchMock.mockResolvedValueOnce(res(200, { enabled: true, user: { user_id: 'u1', role: 'admin' } }))
    const ok = await getAuthStatus()
    expect(ok.enabled).toBe(true)
    expect(ok.user?.role).toBe('admin')

    // anonymous but auth enabled
    fetchMock.mockResolvedValueOnce(res(200, { enabled: true, user: null }))
    expect(await getAuthStatus()).toEqual({ enabled: true, user: null })

    // no auth backend / non-2xx → open
    fetchMock.mockResolvedValueOnce(res(404, {}))
    expect(await getAuthStatus()).toEqual({ enabled: false, user: null })

    // network throw → open (never throws)
    fetchMock.mockRejectedValueOnce(new Error('refused'))
    expect(await getAuthStatus()).toEqual({ enabled: false, user: null })
  })

  it('createUser posts the body and returns the created user', async () => {
    fetchMock.mockResolvedValueOnce(res(201, { user_id: 'u2', email: 'a@b.c', role: 'creator' }))
    const u = await createUser({ email: 'a@b.c', role: 'creator' })
    expect(u.user_id).toBe('u2')
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/app/admin/users')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ email: 'a@b.c', role: 'creator' })
  })

  it('surfaces the FastAPI {detail} on errors (e.g. self-lockout)', async () => {
    fetchMock.mockResolvedValueOnce(res(400, { detail: 'You cannot remove your own admin role.' }))
    await expect(patchUser('u1', { role: 'creator' })).rejects.toThrow(
      'You cannot remove your own admin role.',
    )
  })

  it('deleteUser tolerates 204', async () => {
    fetchMock.mockResolvedValueOnce(res(204, null))
    await expect(deleteUser('u9')).resolves.toBeUndefined()
    expect(fetchMock.mock.calls[0][1].method).toBe('DELETE')
  })
})
