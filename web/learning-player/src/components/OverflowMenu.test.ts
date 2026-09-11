import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import OverflowMenu from './OverflowMenu.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

// Stub <Teleport> so the menu renders inline in the wrapper (it teleports to <body> in the app);
// attachTo keeps focus real for the focus/keyboard assertions.
function mountMenu() {
  return mount(OverflowMenu, {
    attachTo: document.body,
    global: { plugins: [i18n], stubs: { teleport: true } },
    slots: {
      default: `<button data-menuitem role="menuitem" data-testid="act-a">A</button>
                <button data-menuitem role="menuitem" data-testid="act-b">B</button>`,
    },
  })
}

afterEach(() => {
  document.body.innerHTML = ''
})

describe('OverflowMenu', () => {
  it('is closed until the trigger is tapped, then shows a role=menu panel', async () => {
    const w = mountMenu()
    expect(w.find('[data-testid="overflow-menu"]').exists()).toBe(false)
    expect(w.get('[data-testid="overflow-trigger"]').attributes('aria-expanded')).toBe('false')

    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    const menu = w.find('[data-testid="overflow-menu"]')
    expect(menu.exists()).toBe(true)
    expect(menu.attributes('role')).toBe('menu')
    expect(w.get('[data-testid="overflow-trigger"]').attributes('aria-expanded')).toBe('true')
    // Focus moves onto the first item on open.
    expect(document.activeElement?.getAttribute('data-testid')).toBe('act-a')
  })

  it('roams items with ArrowDown/ArrowUp and wraps', async () => {
    const w = mountMenu()
    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    const menu = w.get('[data-testid="overflow-menu"]')

    await menu.trigger('keydown', { key: 'ArrowDown' })
    expect(document.activeElement?.getAttribute('data-testid')).toBe('act-b')
    await menu.trigger('keydown', { key: 'ArrowDown' })
    expect(document.activeElement?.getAttribute('data-testid')).toBe('act-a') // wraps
  })

  it('Escape closes and restores focus to the trigger', async () => {
    const w = mountMenu()
    const trigger = w.get('[data-testid="overflow-trigger"]').element as HTMLElement
    await w.get('[data-testid="overflow-trigger"]').trigger('click')

    await w.get('[data-testid="overflow-menu"]').trigger('keydown', { key: 'Escape' })
    expect(w.find('[data-testid="overflow-menu"]').exists()).toBe(false)
    expect(document.activeElement).toBe(trigger)
  })

  it('an outside pointer closes the menu', async () => {
    const w = mountMenu()
    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    expect(w.find('[data-testid="overflow-menu"]').exists()).toBe(true)

    document.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    await new Promise((r) => queueMicrotask(() => r(null)))
    await w.vm.$nextTick()
    expect(w.find('[data-testid="overflow-menu"]').exists()).toBe(false)
  })

  it('exposes close() so a slot item dismisses the menu after acting', async () => {
    const w = mountMenu()
    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    expect(w.find('[data-testid="overflow-menu"]').exists()).toBe(true)
    ;(w.vm as unknown as { close: () => void }).close()
    await w.vm.$nextTick()
    expect(w.find('[data-testid="overflow-menu"]').exists()).toBe(false)
  })
})
