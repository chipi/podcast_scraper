// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const storage = new Map<string, string>()

vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
  clear: () => storage.clear(),
})

describe('shell corpus path persistence', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it('persists corpus path to localStorage when it changes', async () => {
    const { useShellStore } = await import('./shell')
    const shell = useShellStore()
    shell.corpusPath = '/data/corpus'
    await nextTick()
    expect(storage.get('ps_corpus_path')).toBe('/data/corpus')
  })

  it('reads initial corpus path from localStorage when present', async () => {
    storage.set('ps_corpus_path', '/from/storage')
    setActivePinia(createPinia())
    const { useShellStore } = await import('./shell')
    const shell = useShellStore()
    expect(shell.corpusPath).toBe('/from/storage')
  })
})
