import { describe, expect, it, vi } from 'vitest'

/**
 * Android downloads must not be handed a directory the plugin cannot resolve (2026-09-24, #2139).
 *
 * `Filesystem.downloadFile` is served by the plugin's LEGACY Android implementation, whose
 * `getDirectory()` handles DOCUMENTS / DATA / LIBRARY / CACHE / EXTERNAL / EXTERNAL_STORAGE and
 * has no case for `LIBRARY_NO_CLOUD`. It returns null, that null reaches
 * `FileOutputStream(null)`, and it throws. Every download on Android failed with "Error
 * downloading file: null", recorded itself as retryable, and every retry failed the same way —
 * the feature did not work on the platform at all.
 *
 * Nothing could have caught it before this: the browser tiers never reach `downloadFile`, the iOS
 * device tier exercises a different native implementation, and the unit suite mocks the plugin.
 * The Android device tier found it on its first real run. This guard is the cheap half of that —
 * it cannot prove the download works, but it does fail if the directory regresses.
 *
 * A SEPARATE FILE because `DOWNLOAD_FETCH_DIR` is resolved once at module evaluation:
 * `downloads.test.ts` mocks the platform as `web`, and a per-test override would not re-run the
 * constant.
 */
vi.mock('@capacitor/core', () => ({
  Capacitor: {
    convertFileSrc: (u: string) => `capacitor-file://${u}`,
    isNativePlatform: () => true,
    getPlatform: () => 'android',
  },
  // `downloads.ts` pulls in plugins transitively; without this the mock replaces the whole module
  // and the import graph fails before the constant under test is ever read.
  registerPlugin: () => ({}),
  WebPlugin: class {},
}))
vi.mock('@capacitor/filesystem', () => ({
  Directory: {
    LibraryNoCloud: 'LIBRARY_NO_CLOUD',
    Documents: 'DOCUMENTS',
    Data: 'DATA',
    Cache: 'CACHE',
  },
  Encoding: { UTF8: 'utf8' },
  Filesystem: {},
}))

describe('the download directory on Android', () => {
  it('fetches into DATA, which the legacy implementation can resolve', async () => {
    const { DOWNLOAD_FETCH_DIR } = await import('./downloads')
    expect(
      DOWNLOAD_FETCH_DIR,
      'downloadFile on Android must not be given LIBRARY_NO_CLOUD — the legacy implementation ' +
        'returns null for it and FileOutputStream(null) throws, so every download fails for ever.',
    ).toBe('DATA')
  })

  it('still READS from LibraryNoCloud, so the two point at the same bytes', async () => {
    const { DOWNLOAD_DIR } = await import('./downloads')
    // On Android both resolve to `context.filesDir` — `Data` there, and `LIBRARY_NO_CLOUD` in the
    // MODERN implementation that serves stat/getUri/readdir/deleteFile. That is what makes the
    // substitution correct rather than merely working: a file fetched into one is found by the
    // other at the same relative path. If this ever changes, downloads would "succeed" and then be
    // unreadable, which is worse than the failure it replaced.
    expect(DOWNLOAD_DIR).toBe('LIBRARY_NO_CLOUD')
  })
})
