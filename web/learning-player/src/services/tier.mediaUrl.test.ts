import { afterEach, describe, expect, it, vi } from 'vitest'

const isNativePlatform = vi.fn(() => false)
vi.mock('@capacitor/core', () => ({
  Capacitor: { isNativePlatform: () => isNativePlatform(), getPlatform: () => 'web' },
}))

const { resolveMediaUrl } = await import('./tier')

afterEach(() => vi.restoreAllMocks())

describe('resolveMediaUrl (#1905 — images on native)', () => {
  it('leaves a relative URL alone on web, where the origin is already right', () => {
    // BASE is '/api/app' here, so the browser resolves it against the app origin correctly.
    expect(resolveMediaUrl('/api/app/artwork?ref=x')).toBe('/api/app/artwork?ref=x')
  })

  it('passes absolute URLs through untouched', () => {
    expect(resolveMediaUrl('https://cdn.example.com/a.jpg')).toBe('https://cdn.example.com/a.jpg')
    // A downloaded artwork file must not be rewritten.
    expect(resolveMediaUrl('capacitor://localhost/_capacitor_file_/a.jpg')).toBe(
      'capacitor://localhost/_capacitor_file_/a.jpg',
    )
    expect(resolveMediaUrl('data:image/png;base64,AAA')).toBe('data:image/png;base64,AAA')
  })

  it('returns null for nothing', () => {
    expect(resolveMediaUrl(null)).toBeNull()
    expect(resolveMediaUrl(undefined)).toBeNull()
    expect(resolveMediaUrl('')).toBeNull()
  })
})

describe('resolveMediaUrl on native', () => {
  it('absolutises against the API base, because capacitor://localhost has no artwork', async () => {
    vi.resetModules()
    isNativePlatform.mockReturnValue(true)
    vi.stubEnv('VITE_API_BASE_URL', 'https://closelistening.app/api/app')
    const { resolveMediaUrl: nativeResolve } = await import('./tier')
    expect(nativeResolve('/api/app/artwork?ref=x')).toBe(
      'https://closelistening.app/api/app/artwork?ref=x',
    )
    vi.unstubAllEnvs()
  })
})
