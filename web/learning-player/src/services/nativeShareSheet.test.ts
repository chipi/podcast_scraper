import { describe, expect, it, vi } from 'vitest'

const { share } = vi.hoisted(() => ({ share: vi.fn() }))
vi.mock('@capacitor/share', () => ({ Share: { share } }))

import { openShareSheet } from './native'

describe('openShareSheet (operator 2026-10-07)', () => {
  // Each test sets its own implementation. (A `beforeEach(mockReset)` here made vitest 4 report the
  // mock's thrown error as the test's failure even though the code under test caught it.)

  it('treats a dismissed sheet as done, not as a failure', async () => {
    // What the plugin rejects with on iOS (completed == false) and Android (RESULT_CANCELED, which
    // many targets return even after a successful share). This showed "Couldn't make the card".
    share.mockImplementation(async () => {
      throw new Error('Share canceled')
    })
    await expect(openShareSheet('card.png', 'file:///c/card.png')).resolves.toBeUndefined()
  })

  it('still throws a real sharing error', async () => {
    share.mockImplementation(async () => {
      throw new Error('Error sharing item')
    })
    await expect(openShareSheet('card.png', 'file:///c/card.png')).rejects.toThrow('Error sharing item')
  })

  it('hands the file to the sheet', async () => {
    share.mockResolvedValue({ activityType: 'x' })
    await openShareSheet('card.png', 'file:///c/card.png')
    expect(share).toHaveBeenCalledWith({ title: 'card.png', url: 'file:///c/card.png', dialogTitle: 'card.png' })
  })
})
