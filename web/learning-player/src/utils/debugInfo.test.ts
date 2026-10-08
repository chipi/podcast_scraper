// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest'
import * as lifecycle from '../services/lifecycle'
import { collectDebugInfo } from './debugInfo'

const ctx = {
  version: '1.0.3',
  sha: 'abc1234',
  builtAt: '2026-10-08T10:00:00Z',
  platform: 'android',
  target: 'prod',
  route: '/episode/ep-1',
  userId: 'u_123',
}

afterEach(() => vi.restoreAllMocks())

describe('collectDebugInfo (operator 2026-10-08)', () => {
  it('leads with when, the app, the screen the tester came from and the account', async () => {
    vi.spyOn(lifecycle, 'nativeMemoryInfo').mockResolvedValue(null)
    const text = await collectDebugInfo(ctx)
    const lines = text.split('\n')
    expect(lines[0]).toBe('Close Listening debug info')
    expect(lines[1]).toMatch(/^Time: .*UTC \d{4}-\d{2}-\d{2}T/)
    expect(text).toContain('App: 1.0.3 · abc1234 · built 2026-10-08T10:00:00Z · android · backend prod')
    expect(text).toContain('Screen in app: /episode/ep-1')
    expect(text).toContain('Account: u_123')
    expect(text).toContain('User agent: ')
    // Says what it could not read rather than leaving it out.
    expect(text).toContain('Native: (not available')
    expect(text).toContain('Not readable by any app: free GPU memory, GPU load')
  })

  it('includes the native memory facts, in a stable order, when the app provides them', async () => {
    vi.spyOn(lifecycle, 'nativeMemoryInfo').mockResolvedValue({
      totalMb: 3800,
      availMb: 412,
      model: 'SM-A135F',
      lowMemory: true,
      webView: 'com.google.android.webview 129.0.6668.100',
    })
    const text = await collectDebugInfo(ctx)
    expect(text).toContain('Native model: SM-A135F')
    expect(text).toContain('Native availMb: 412')
    expect(text).toContain('Native lowMemory: true')
    expect(text.indexOf('Native model')).toBeLessThan(text.indexOf('Native availMb'))
  })
})
