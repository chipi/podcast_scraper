import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import Tabs from './Tabs.vue'
import { panelAttrs, panelId, tabId } from './tabs'

type K = 'a' | 'b' | 'c'

const TABS = [
  { key: 'a' as const, label: 'Alpha', testid: 'tab-a' },
  { key: 'b' as const, label: 'Beta' },
  { key: 'c' as const, label: 'Gamma' },
]

function make(selected: K = 'a', extra: Record<string, unknown> = {}) {
  return mount(Tabs, {
    props: {
      modelValue: selected,
      tabs: TABS,
      label: 'Sections',
      idPrefix: 'demo',
      'onUpdate:modelValue': (v: K) => void v,
      ...extra,
    },
    attachTo: document.body,
  })
}

describe('Tabs (#1594 item 7)', () => {
  it('names the tablist — an unnamed one announces as a bare "tab list"', () => {
    const w = make()
    expect(w.get('[role="tablist"]').attributes('aria-label')).toBe('Sections')
  })

  it('marks exactly one tab selected', () => {
    const w = make('b')
    const sel = w.findAll('[role="tab"]').filter((t) => t.attributes('aria-selected') === 'true')
    expect(sel).toHaveLength(1)
    expect(sel[0].text()).toBe('Beta')
  })

  describe('roving tabindex', () => {
    it('puts exactly ONE tab in the page tab order', () => {
      // The rule everybody skips, because plain buttons are still *usable*: you can Tab to each
      // one, so nothing looks broken. It just costs a five-tab strip five Tab presses instead of
      // one, and it is invisible to a mouse test and to review.
      const w = make('b')
      const tabbable = w.findAll('[role="tab"]').filter((t) => t.attributes('tabindex') === '0')
      expect(tabbable).toHaveLength(1)
      expect(tabbable[0].text()).toBe('Beta')
    })

    it('takes the unselected tabs OUT of the tab order', () => {
      const w = make('a')
      const rest = w.findAll('[role="tab"]').filter((t) => t.attributes('aria-selected') !== 'true')
      expect(rest.every((t) => t.attributes('tabindex') === '-1')).toBe(true)
    })
  })

  describe('tab ↔ panel linkage', () => {
    it('every tab points at its panel, and the ids are the ones panels generate', () => {
      // The other universally-missed rule. Without the pair, "tab" and "tabpanel" are two unrelated
      // announcements. It needs matching ids on BOTH sides, which is what gets dropped when markup
      // is copy-pasted — so both sides come from the same helpers, and this asserts they agree.
      const w = make()
      for (const t of TABS) {
        const btn = w.findAll('[role="tab"]').find((b) => b.text() === t.label)!
        expect(btn.attributes('id')).toBe(tabId('demo', t.key))
        expect(btn.attributes('aria-controls')).toBe(panelId('demo', t.key))
        expect(panelAttrs('demo', t.key)['aria-labelledby']).toBe(btn.attributes('id'))
      }
    })

    it('a panel is focusable, so an arrow-key user has somewhere to land', () => {
      // A panel holding no focusable element of its own is a dead end for keyboard navigation.
      expect(panelAttrs('demo', 'a').tabindex).toBe('0')
      expect(panelAttrs('demo', 'a').role).toBe('tabpanel')
    })

    it('different prefixes do not collide, so two strips can share a page', () => {
      expect(tabId('library', 'a')).not.toBe(tabId('browse', 'a'))
    })

    it('a tab and its panel get DIFFERENT ids', () => {
      // The linkage test above compares helper output to helper output, so it stays green if both
      // helpers return the same string — and duplicate ids in one document break the association
      // entirely. Asserted here because the circular version cannot see it.
      expect(tabId('demo', 'a')).not.toBe(panelId('demo', 'a'))
    })
  })

  describe('arrow keys', () => {
    it('Right moves to the next tab', async () => {
      const w = make('a')
      await w.get('[role="tablist"]').trigger('keydown', { key: 'ArrowRight' })
      expect(w.emitted('update:modelValue')?.at(-1)).toEqual(['b'])
    })

    it('Left moves to the previous tab', async () => {
      const w = make('b')
      await w.get('[role="tablist"]').trigger('keydown', { key: 'ArrowLeft' })
      expect(w.emitted('update:modelValue')?.at(-1)).toEqual(['a'])
    })

    it('wraps at both ends rather than dead-ending', async () => {
      const last = make('c')
      await last.get('[role="tablist"]').trigger('keydown', { key: 'ArrowRight' })
      expect(last.emitted('update:modelValue')?.at(-1)).toEqual(['a'])

      const first = make('a')
      await first.get('[role="tablist"]').trigger('keydown', { key: 'ArrowLeft' })
      expect(first.emitted('update:modelValue')?.at(-1)).toEqual(['c'])
    })

    it('Home and End jump to the extremes', async () => {
      const w = make('b')
      await w.get('[role="tablist"]').trigger('keydown', { key: 'End' })
      expect(w.emitted('update:modelValue')?.at(-1)).toEqual(['c'])

      const w2 = make('b')
      await w2.get('[role="tablist"]').trigger('keydown', { key: 'Home' })
      expect(w2.emitted('update:modelValue')?.at(-1)).toEqual(['a'])
    })

    it('leaves other keys alone — and does not preventDefault them', async () => {
      // Notably Tab: swallowing it would trap focus inside the strip, turning an accessibility fix
      // into an accessibility bug.
      //
      // Checking only that no selection was emitted is NOT enough, and this test originally made
      // that mistake: a component that calls `preventDefault()` on every key emits nothing either,
      // and the focus trap it creates is exactly the failure the comment above claims to guard.
      // So the default has to be asserted directly.
      const w = make('a')
      for (const key of ['Tab', 'Enter', ' ', 'a', 'Escape']) {
        const e = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true })
        w.get('[role="tablist"]').element.dispatchEvent(e)
        expect(e.defaultPrevented, `"${key}" was swallowed by the tablist`).toBe(false)
      }
      expect(w.emitted('update:modelValue')).toBeUndefined()
    })

    it('DOES preventDefault the keys it handles', async () => {
      // The other half: an unprevented ArrowDown scrolls the page while moving the tab.
      const w = make('a')
      for (const key of ['ArrowRight', 'ArrowLeft', 'ArrowUp', 'ArrowDown', 'Home', 'End']) {
        const e = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true })
        w.get('[role="tablist"]').element.dispatchEvent(e)
        expect(e.defaultPrevented, `"${key}" moved the tab but also did its default`).toBe(true)
      }
    })

    it('moves FOCUS along with the selection', async () => {
      // Without this the roving tabindex strands focus on a tab that is no longer selected, and the
      // next arrow press starts from the wrong place — the strip appears to skip a tab.
      const w = make('a')
      await w.get('[role="tablist"]').trigger('keydown', { key: 'ArrowRight' })
      expect(document.activeElement).toBe(w.findAll('[role="tab"]')[1].element)
    })
  })

  it('clicking a tab selects it', async () => {
    const w = make('a')
    await w.findAll('[role="tab"]')[2].trigger('click')
    expect(w.emitted('update:modelValue')?.at(-1)).toEqual(['c'])
  })

  it('carries the caller data-testid through, so existing specs keep their selectors', () => {
    expect(make().find('[data-testid="tab-a"]').exists()).toBe(true)
  })

  describe('the radio pattern', () => {
    function radio(selected: K = 'a') {
      return mount(Tabs, {
        props: {
          modelValue: selected,
          tabs: TABS,
          label: 'Window',
          idPrefix: 'win',
          pattern: 'radio' as const,
          'onUpdate:modelValue': (v: K) => void v,
        },
        attachTo: document.body,
      })
    }

    it('announces as a radiogroup, not a tablist', () => {
      // "Tab" is the wrong announcement for "set this to one of four values", and these controls
      // switch no panel — the trend window re-queries a rail its PARENT owns.
      const w = radio()
      expect(w.get('[role="radiogroup"]').exists()).toBe(true)
      expect(w.find('[role="tablist"]').exists()).toBe(false)
      expect(w.findAll('[role="radio"]')).toHaveLength(3)
    })

    it('uses aria-checked, and emits no dangling aria-controls', () => {
      // A `role="tab"` whose `aria-controls` names nothing is worse than no linkage: a dangling
      // reference is a broken promise rather than an absent one.
      const w = radio('b')
      const opts = w.findAll('[role="radio"]')
      expect(opts[1].attributes('aria-checked')).toBe('true')
      expect(opts[0].attributes('aria-checked')).toBe('false')
      expect(opts.every((o) => o.attributes('aria-controls') === undefined)).toBe(true)
      expect(opts.every((o) => o.attributes('aria-selected') === undefined)).toBe(true)
    })

    it('keeps the same keyboard contract as the tabs pattern', () => {
      // Which is the whole reason both live in one component instead of two that drift.
      const w = radio('a')
      expect(w.findAll('[role="radio"]').filter((o) => o.attributes('tabindex') === '0')).toHaveLength(1)
      w.get('[role="radiogroup"]').trigger('keydown', { key: 'ArrowRight' })
      expect(w.emitted('update:modelValue')?.at(-1)).toEqual(['b'])
    })
  })

  describe('variants', () => {
    it('renders the three existing shapes rather than imposing one look', () => {
      // The strips look different on purpose: an underline is a page-level section switcher, a pill
      // is a compact in-card control. Item 7 asks for one COMPONENT, not one appearance.
      expect(make('a').get('[role="tablist"]').classes()).toContain('border-b')
      expect(make('a', { variant: 'segment' }).get('[role="tablist"]').classes()).toContain('lp-segment')
      expect(make('a', { variant: 'pill' }).get('[role="tablist"]').classes()).toContain('rounded-full')
    })

    it('equalWidth stretches the tabs, for the five-up phone strip', () => {
      const w = make('a', { equalWidth: true })
      expect(w.findAll('[role="tab"]')[0].classes()).toContain('flex-1')
    })
  })
})
