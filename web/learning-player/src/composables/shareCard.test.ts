import { afterEach, describe, expect, it, vi } from 'vitest'

import * as api from '../services/api'
import * as native from '../services/native'
import { shareCard } from './shareCard'

const PNG = new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], { type: 'image/png' })

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('shareCard — the server card, shared as an IMAGE (operator 2026-10-05)', () => {
  it('fetches the card for this kind + id', async () => {
    const fetch = vi.spyOn(api, 'fetchShareCard').mockResolvedValue(PNG)
    vi.spyOn(native, 'isNative').mockReturnValue(true)
    vi.spyOn(native, 'deliverFile').mockResolvedValue(undefined)
    await shareCard('theme', 'tc:risk', 'Risk')
    expect(fetch).toHaveBeenCalledWith('theme', 'tc:risk')
  })

  it('on NATIVE hands the PNG to the share sheet as a file — never a text file', async () => {
    vi.spyOn(api, 'fetchShareCard').mockResolvedValue(PNG)
    vi.spyOn(native, 'isNative').mockReturnValue(true)
    const deliver = vi.spyOn(native, 'deliverFile').mockResolvedValue(undefined)
    const text = vi.spyOn(native, 'saveAndShareText').mockResolvedValue(undefined)
    await shareCard('episode', 'ep-1', 'Index Investing')
    expect(deliver).toHaveBeenCalledWith('index-investing.png', PNG)
    expect(text).not.toHaveBeenCalled()
  })

  it("on the WEB uses the browser's file share where it can", async () => {
    vi.spyOn(api, 'fetchShareCard').mockResolvedValue(PNG)
    vi.spyOn(native, 'isNative').mockReturnValue(false)
    const deliver = vi.spyOn(native, 'deliverFile').mockResolvedValue(undefined)
    const share = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { ...navigator, canShare: () => true, share })
    await shareCard('show', 'p05', 'Long Horizon Notes')
    expect(share).toHaveBeenCalledTimes(1)
    const files = share.mock.calls[0][0].files as File[]
    expect(files[0].type).toBe('image/png')
    expect(deliver).not.toHaveBeenCalled()
  })

  it('on the WEB without file share, downloads the PNG', async () => {
    vi.spyOn(api, 'fetchShareCard').mockResolvedValue(PNG)
    vi.spyOn(native, 'isNative').mockReturnValue(false)
    const deliver = vi.spyOn(native, 'deliverFile').mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { ...navigator, canShare: undefined, share: undefined })
    await shareCard('topic', 'topic:risk', 'Risk')
    expect(deliver).toHaveBeenCalledWith('risk.png', PNG)
  })

  it('a closed share sheet is not a failure and does not fall through to a download', async () => {
    vi.spyOn(api, 'fetchShareCard').mockResolvedValue(PNG)
    vi.spyOn(native, 'isNative').mockReturnValue(false)
    const deliver = vi.spyOn(native, 'deliverFile').mockResolvedValue(undefined)
    const abort = Object.assign(new Error('closed'), { name: 'AbortError' })
    vi.stubGlobal('navigator', { ...navigator, canShare: () => true, share: vi.fn().mockRejectedValue(abort) })
    await shareCard('topic', 'topic:risk', 'Risk')
    expect(deliver).not.toHaveBeenCalled()
  })

  it('a failed fetch rejects, so the menu can say so', async () => {
    vi.spyOn(api, 'fetchShareCard').mockRejectedValue(new Error('offline'))
    await expect(shareCard('topic', 'topic:risk', 'Risk')).rejects.toThrow('offline')
  })
})
