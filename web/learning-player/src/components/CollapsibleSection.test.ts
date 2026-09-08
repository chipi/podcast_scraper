import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it } from 'vitest'
import CollapsibleSection from './CollapsibleSection.vue'

function make(props: Record<string, unknown> = {}) {
  return mount(CollapsibleSection, {
    props: { title: 'Insights', sectionKey: 'insights', ...props },
    slots: { default: '<p>body content</p>' },
    attachTo: document.body,
  })
}

beforeEach(() => {
  localStorage.clear()
})

describe('CollapsibleSection', () => {
  it('is OPEN when the user has never touched it', () => {
    // Collapsing by default would hide the episode's substance behind a tap nobody asked for. The
    // panel's job is to show it; folding is an escape hatch, not the resting state.
    expect(make().get('details').attributes('open')).toBeDefined()
  })

  it('shows the count in the header, so a folded section still says what it holds', () => {
    // Folding must not cost you the knowledge that something is there. "Insights · 8" is legible
    // closed; "Insights" alone is a locked door.
    expect(make({ count: 8 }).get('summary').text()).toContain('8')
  })

  it('omits the separator when there is no count', () => {
    expect(make().get('summary').text().trim()).toBe('Insights')
  })

  it('remembers being closed, and reopens closed next time', async () => {
    const w = make()
    const details = w.get('details').element as HTMLDetailsElement
    details.open = false
    details.dispatchEvent(new Event('toggle'))
    await w.vm.$nextTick() // the persist runs in a watcher, which flushes on the next tick
    expect(localStorage.getItem('lp.kp.insights')).toBe('closed')

    // A fresh mount — the same user, a later visit.
    expect(make().get('details').attributes('open')).toBeUndefined()
  })

  it('remembers being reopened', async () => {
    localStorage.setItem('lp.kp.insights', 'closed')
    const w = make()
    const details = w.get('details').element as HTMLDetailsElement
    details.open = true
    details.dispatchEvent(new Event('toggle'))
    await w.vm.$nextTick()
    expect(localStorage.getItem('lp.kp.insights')).toBe('open')
  })

  it('keeps state per SECTION, not shared between them', () => {
    localStorage.setItem('lp.kp.insights', 'closed')
    expect(make({ sectionKey: 'related' }).get('details').attributes('open')).toBeDefined()
  })

  it('falls back to OPEN when storage is unavailable, never to hidden', () => {
    // Private mode. A preference we cannot persist is not a reason to hide content.
    const original = Object.getOwnPropertyDescriptor(window, 'localStorage')
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get() {
        throw new Error('denied')
      },
    })
    try {
      expect(make().get('details').attributes('open')).toBeDefined()
    } finally {
      if (original) Object.defineProperty(window, 'localStorage', original)
    }
  })

  it('uses a native <details>, so keyboard and AT support are not reimplemented', () => {
    // Enter/Space, the disclosure role and the expanded-state announcement all come from the
    // element. Rebuilding those with a div and a ref is where a11y bugs live.
    const w = make()
    expect(w.element.tagName.toLowerCase()).toBe('details')
    expect(w.find('summary').exists()).toBe(true)
  })
})
