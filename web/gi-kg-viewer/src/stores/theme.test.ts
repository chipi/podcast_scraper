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
