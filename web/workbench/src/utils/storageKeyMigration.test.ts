// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { migrateStorageKey } from './storageKeyMigration'

const storage = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
})

describe('migrateStorageKey', () => {
  beforeEach(() => storage.clear())

  it('moves the old value to the new key and removes the old key', () => {
    storage.set('old', 'light')
    migrateStorageKey('new', 'old')
    expect(storage.get('new')).toBe('light')
    expect(storage.has('old')).toBe(false)
  })

  it('keeps a value already saved under the new key', () => {
    storage.set('new', 'dark')
    storage.set('old', 'light')
    migrateStorageKey('new', 'old')
    expect(storage.get('new')).toBe('dark')
    expect(storage.has('old')).toBe(false)
  })

  it('does nothing when there is no old value', () => {
    migrateStorageKey('new', 'old')
    expect(storage.has('new')).toBe(false)
  })
})
