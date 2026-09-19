import { afterEach, describe, expect, it } from 'vitest'
import { defineComponent, h, ref } from 'vue'
import { mount } from '@vue/test-utils'
import { useAnchoredMenu } from './useAnchoredMenu'

/**
 * Regression guard for the top-layer trap (operator 2026-09-19).
 *
 * Collection and Share "did not work" on the topic sheet: their panels teleported to `<body>`,
 * which renders UNDERNEATH a `showModal()` dialog no matter the z-index, so the menu opened
 * invisibly. Every other signal — `open`, aria state, the click handler — said it had worked, which
 * is what made it read as a dead button rather than a layering bug.
 */
const Harness = defineComponent({
  setup() {
    const triggerEl = ref<HTMLElement | null>(null)
    const panelEl = ref<HTMLElement | null>(null)
    const menu = useAnchoredMenu(triggerEl, panelEl)
    return { ...menu, triggerEl, panelEl }
  },
  render: () => h('div'),
})

describe('useAnchoredMenu teleport target', () => {
  afterEach(() => {
    document.querySelectorAll('dialog').forEach((d) => d.remove())
  })

  it('teleports to <body> when no modal dialog is on screen', () => {
    const w = mount(Harness)
    w.vm.toggle()
    expect(w.vm.teleportTarget).toBe('body')
    w.unmount()
  })

  it('teleports INTO the open dialog, so the panel shares its top layer', () => {
    const dialog = document.createElement('dialog')
    dialog.setAttribute('open', '')
    document.body.appendChild(dialog)

    const w = mount(Harness)
    w.vm.toggle()
    expect(w.vm.teleportTarget).toBe(dialog)
    w.unmount()
  })

  it('re-resolves on every open, not once at setup', () => {
    const w = mount(Harness)
    // Opened with nothing layered: body is correct.
    w.vm.toggle()
    expect(w.vm.teleportTarget).toBe('body')
    w.vm.close(false)

    // The same trigger, tapped again once a sheet is up, must follow it into the top layer.
    const dialog = document.createElement('dialog')
    dialog.setAttribute('open', '')
    document.body.appendChild(dialog)
    w.vm.toggle()
    expect(w.vm.teleportTarget).toBe(dialog)
    w.unmount()
  })
})
