// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { copyTextToClipboard } from './clipboard'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('copyTextToClipboard', () => {
  it('returns true when the async Clipboard API succeeds', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })

    const ok = await copyTextToClipboard('hello')

    expect(ok).toBe(true)
    expect(writeText).toHaveBeenCalledWith('hello')
  })

  it('falls back to execCommand when the Clipboard API rejects', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error('blocked'))
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const execCommand = vi.fn().mockReturnValue(true)
    // happy-dom does not implement execCommand; install a stub.
    ;(document as unknown as { execCommand: typeof execCommand }).execCommand = execCommand

    const ok = await copyTextToClipboard('fallback text')

    expect(ok).toBe(true)
    expect(execCommand).toHaveBeenCalledWith('copy')
    // The scratch textarea must be cleaned up.
    expect(document.querySelector('textarea')).toBeNull()
  })

  it('returns false when both paths fail', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error('blocked'))
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    ;(document as unknown as { execCommand: () => boolean }).execCommand = () => {
      throw new Error('execCommand unavailable')
    }

    const ok = await copyTextToClipboard('nope')

    expect(ok).toBe(false)
  })
})
