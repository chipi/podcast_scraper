import { afterEach, describe, expect, it, vi } from 'vitest'

const fetchMock = vi.fn()
vi.mock('./httpClient', () => ({
  fetchWithTimeout: (...a: unknown[]) => fetchMock(...a),
}))

import { createUser, deleteUser, getMe, loginUrl, patchUser } from './authApi'

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

  it('getMe returns the user, or null on 401', async () => {
    fetchMock.mockResolvedValueOnce(res(200, { user_id: 'u1', role: 'admin' }))
    expect((await getMe())?.role).toBe('admin')
    fetchMock.mockResolvedValueOnce(res(401, {}))
    expect(await getMe()).toBeNull()
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
