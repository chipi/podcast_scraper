import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import * as native from '../services/native'
import { isVersionNewer, useAppUpdate } from './useAppUpdate'

describe('isVersionNewer', () => {
  it('compares numerically, part by part', () => {
    expect(isVersionNewer('1.2.0', '1.1.9')).toBe(true)
    expect(isVersionNewer('2.0.0', '1.9.9')).toBe(true)
    expect(isVersionNewer('1.0.10', '1.0.2')).toBe(true) // not lexicographic
    expect(isVersionNewer('1.0.0', '1.0.0')).toBe(false) // equal is not newer
    expect(isVersionNewer('1.0.0', '1.2.0')).toBe(false) // older
  })

  it('tolerates ragged lengths and non-numeric suffixes', () => {
    expect(isVersionNewer('1.1', '1.0.9')).toBe(true)
    expect(isVersionNewer('1.0', '1.0.0')).toBe(false)
    expect(isVersionNewer('2.7.0', '2.7.0.dev0')).toBe(false) // .dev0 → 0, equal
  })
})

describe('useAppUpdate.check', () => {
  beforeEach(() => vi.restoreAllMocks())

  it('is a no-op on the web (service worker owns web updates)', async () => {
    vi.spyOn(native, 'isNative').mockReturnValue(false)
    const health = vi.spyOn(api, 'getHealth')
    const u = useAppUpdate()
    await u.check()
    expect(health).not.toHaveBeenCalled()
    expect(u.updateAvailable.value).toBe(false)
  })

  it('skips when the server has not published a player_version (no mismatched-scale compare)', async () => {
    vi.spyOn(native, 'isNative').mockReturnValue(true)
    vi.spyOn(api, 'getHealth').mockResolvedValue({ code_version: '2.7.0.dev0', player_version: null })
    const u = useAppUpdate()
    await u.check()
    expect(u.updateAvailable.value).toBe(false)
  })

  it('flags an update on native when the server player_version is newer than the baked build', async () => {
    // __APP_VERSION__ is the baked build; server publishes a strictly-newer one.
    vi.spyOn(native, 'isNative').mockReturnValue(true)
    const newer = `${parseInt(__APP_VERSION__, 10) + 1}.0.0`
    vi.spyOn(api, 'getHealth').mockResolvedValue({ code_version: 'x', player_version: newer })
    const u = useAppUpdate()
    await u.check()
    expect(u.updateAvailable.value).toBe(true)
    expect(u.latestVersion.value).toBe(newer)
  })

  it('does not flag when the server version equals the baked build', async () => {
    vi.spyOn(native, 'isNative').mockReturnValue(true)
    vi.spyOn(api, 'getHealth').mockResolvedValue({ code_version: 'x', player_version: __APP_VERSION__ })
    const u = useAppUpdate()
    await u.check()
    expect(u.updateAvailable.value).toBe(false)
  })
})
