// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it } from 'vitest'
import { defineComponent, h } from 'vue'
import AppDialog from './AppDialog.vue'

// happy-dom implements <dialog> but not the modal show/close side effects we
// rely on (open property + close event). Patch the prototype so the component's
// showModal()/close() calls flip `open` and dispatch `close` like a browser.
beforeEach(() => {
  const proto = HTMLDialogElement.prototype as unknown as {
    showModal: () => void
    close: () => void
  }
  proto.showModal = function showModal(this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  proto.close = function close(this: HTMLDialogElement) {
    if (!this.open) return
    this.removeAttribute('open')
    this.dispatchEvent(new Event('close'))
  }
})

function mountDialog(open = true) {
  return mount(AppDialog, {
    props: { open, title: 'My Dialog', subtitle: 'a subtitle', testid: 'my-dialog' },
    slots: {
      default: '<p data-testid="body">Body content</p>',
      footer: '<span data-testid="footer">footer</span>',
    },
    attachTo: document.body,
  })
}

describe('AppDialog', () => {
  it('gives each dialog its own title id', () => {
    // Found by eslint (no-useless-assignment): the counter lived in <script setup>, so it restarted
    // per instance and every dialog's aria-labelledby named the same id.
    // Two dialogs in one app, as in the Workbench (useId is unique within an app).
    const w = mount(
      defineComponent({
        render: () => [
          h(AppDialog, { open: true, title: 'First', testid: 'dialog-a' }),
          h(AppDialog, { open: true, title: 'Second', testid: 'dialog-b' }),
        ],
      }),
      { attachTo: document.body },
    )
    const idA = w.get('[data-testid="dialog-a"]').attributes('aria-labelledby')
    const idB = w.get('[data-testid="dialog-b"]').attributes('aria-labelledby')
    expect(idA).toBeTruthy()
    expect(idA).not.toBe(idB)
    expect(w.get(`[id="${idA}"]`).text()).toContain('First')
  })

  it('renders title, subtitle, body and footer when open', () => {
    const w = mountDialog(true)
    expect(w.get('[data-testid="my-dialog"]').attributes('open')).toBeDefined()
    expect(w.text()).toContain('My Dialog')
    expect(w.text()).toContain('a subtitle')
    expect(w.get('[data-testid="body"]').exists()).toBe(true)
    expect(w.get('[data-testid="footer"]').exists()).toBe(true)
  })

  it('emits update:open=false when the Close button is clicked', async () => {
    const w = mountDialog(true)
    await w.get('[data-testid="app-dialog-close"]').trigger('click')
    expect(w.emitted('update:open')?.at(-1)).toEqual([false])
  })

  it('emits update:open=false on native close (Esc / programmatic)', async () => {
    const w = mountDialog(true)
    const dlg = w.get('[data-testid="my-dialog"]').element as HTMLDialogElement
    dlg.close()
    await w.vm.$nextTick()
    expect(w.emitted('update:open')?.at(-1)).toEqual([false])
  })

  it('closes when the backdrop (dialog element) is clicked', async () => {
    const w = mountDialog(true)
    await w.get('[data-testid="my-dialog"]').trigger('click')
    expect(w.emitted('update:open')?.at(-1)).toEqual([false])
  })

  it('does not open the native modal when open=false', () => {
    const w = mountDialog(false)
    expect(w.get('[data-testid="my-dialog"]').attributes('open')).toBeUndefined()
  })
})
