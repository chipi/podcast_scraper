import { afterEach, describe, expect, it, vi } from 'vitest'

const resumeHandlers: Array<() => void> = []
vi.mock('@capacitor/app', () => ({
  App: {
    addListener: vi.fn((event: string, cb: () => void) => {
      if (event === 'resume') resumeHandlers.push(cb)
      return Promise.resolve({ remove: vi.fn() })
    }),
  },
}))
const native = { value: false }
vi.mock('./tier', () => ({ isNativeShell: () => native.value }))

import appSrc from '../App.vue?raw'
import { onForeground } from './foreground'

function setVisibility(state: 'visible' | 'hidden'): void {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state })
  document.dispatchEvent(new Event('visibilitychange'))
}

afterEach(() => {
  resumeHandlers.length = 0
  native.value = false
})

describe('onForeground (2026-10-09: the bell on every return, not only on sign-in)', () => {
  it('web: runs when the page becomes visible, not when it is hidden', () => {
    const run = vi.fn()
    const off = onForeground(run)
    setVisibility('hidden')
    expect(run).not.toHaveBeenCalled()
    setVisibility('visible')
    expect(run).toHaveBeenCalledTimes(1)
    off()
    setVisibility('visible')
    expect(run).toHaveBeenCalledTimes(1)
  })

  it("native: runs on Capacitor's resume (a tapped push brings the app back this way)", () => {
    native.value = true
    const run = vi.fn()
    onForeground(run)
    expect(resumeHandlers).toHaveLength(1)
    resumeHandlers[0]()
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('App.vue reloads the notification inbox from it', () => {
    const block = appSrc.slice(appSrc.indexOf('onForeground(() => {'))
    expect(block, 'App.vue no longer registers a foreground refresh').not.toBe('')
    expect(block.slice(0, 400)).toContain('useNotificationsStore().load()')
  })
})
