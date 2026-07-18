// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

const storage = new Map<string, string>()

vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
})

beforeEach(() => {
  storage.clear()
  document.documentElement.removeAttribute('data-theme')
  setActivePinia(createPinia())
})

async function freshStore() {
  vi.resetModules()
  const { useThemeStore } = await import('./theme')
  setActivePinia(createPinia())
  return useThemeStore()
}

describe('useThemeStore', () => {
  it('defaults to dark when nothing saved', async () => {
    const t = await freshStore()
    expect(t.choice).toBe('dark')
  })

  it('applies data-theme attribute for dark', async () => {
    const t = await freshStore()
    expect(t.choice).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
  })

  it('reads saved preference from localStorage', async () => {
    storage.set('gi-kg-viewer-theme', 'light')
    const t = await freshStore()
    expect(t.choice).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })

  it('ignores invalid localStorage values', async () => {
    storage.set('gi-kg-viewer-theme', 'neon')
    const t = await freshStore()
    expect(t.choice).toBe('dark')
  })

  it('cycle() rotates light → dark → auto → light', async () => {
    storage.set('gi-kg-viewer-theme', 'light')
    const t = await freshStore()
    expect(t.choice).toBe('light')

    t.cycle()
    expect(t.choice).toBe('dark')

    t.cycle()
    expect(t.choice).toBe('auto')

    t.cycle()
    expect(t.choice).toBe('light')
  })

  it('auto removes data-theme attribute', async () => {
    const t = await freshStore()
    t.choice = 'auto'
    await new Promise((r) => setTimeout(r, 0))
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false)
  })

  it('persists choice to localStorage on change', async () => {
    const t = await freshStore()
    t.choice = 'light'
    await new Promise((r) => setTimeout(r, 0))
    expect(storage.get('gi-kg-viewer-theme')).toBe('light')
  })
})

describe('useThemeStore — USERPREFS-1 remote-watch (cross-device sync)', () => {
  /* The store watches ``userPrefs.get('theme')`` — when a server-hydrated
   * value or a cross-tab BroadcastChannel update lands, apply it locally.
   * ``applyingRemote`` breaks the write-back watch loop. We mock the
   * preferences store so we can drive the reactive value directly. */
  it('applies a server-hydrated theme on the initial resolve', async () => {
    const themeRemote = { value: 'light' as unknown }
    const setSpy = vi.fn()
    vi.doMock('./userPreferences', () => ({
      useUserPreferencesStore: () => ({
        get: <T,>(_key: string) => themeRemote.value as T,
        set: setSpy,
      }),
    }))

    vi.resetModules()
    const { useThemeStore } = await import('./theme')
    setActivePinia(createPinia())
    const t = useThemeStore()
    await new Promise((r) => setTimeout(r, 0))

    expect(t.choice).toBe('light')
    // Remote-applied write must NOT loop back through userPrefs.set —
    // ``applyingRemote`` guards the write-back.
    expect(setSpy).not.toHaveBeenCalled()

    vi.doUnmock('./userPreferences')
  })

  it('ignores a null remote value (missing key on server)', async () => {
    vi.doMock('./userPreferences', () => ({
      useUserPreferencesStore: () => ({
        get: <T,>(_key: string) => null as T,
        set: vi.fn(),
      }),
    }))
    vi.resetModules()
    const { useThemeStore } = await import('./theme')
    setActivePinia(createPinia())
    const t = useThemeStore()
    // localStorage was cleared in beforeEach → default is dark.
    expect(t.choice).toBe('dark')
    vi.doUnmock('./userPreferences')
  })

  it('rejects an invalid remote value and keeps the current choice', async () => {
    vi.doMock('./userPreferences', () => ({
      useUserPreferencesStore: () => ({
        get: <T,>(_key: string) => 'neon' as unknown as T,
        set: vi.fn(),
      }),
    }))
    vi.resetModules()
    const { useThemeStore } = await import('./theme')
    setActivePinia(createPinia())
    const t = useThemeStore()
    await new Promise((r) => setTimeout(r, 0))
    // 'neon' fails coerceThemeChoice → fallback to current choice (dark).
    expect(t.choice).toBe('dark')
    vi.doUnmock('./userPreferences')
  })

  it('writes to userPrefs.set on a LOCAL choice change (not remote-flag on)', async () => {
    const setSpy = vi.fn()
    vi.doMock('./userPreferences', () => ({
      useUserPreferencesStore: () => ({
        get: <T,>(_key: string) => null as T,
        set: setSpy,
      }),
    }))
    vi.resetModules()
    const { useThemeStore } = await import('./theme')
    setActivePinia(createPinia())
    const t = useThemeStore()
    t.choice = 'light'
    await new Promise((r) => setTimeout(r, 0))
    expect(setSpy).toHaveBeenCalledWith('theme', 'light')
    vi.doUnmock('./userPreferences')
  })
})
