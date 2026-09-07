import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import { beforeAll, describe, expect, it, vi } from 'vitest'
import ConfirmDialog from './ConfirmDialog.vue'
import en from '../i18n/locales/en.json'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

/**
 * jsdom implements `<dialog>` as an element but not as a dialog: `showModal()` and `close()` are
 * missing entirely, so mounting this component would throw before any assertion ran. Stubbing them
 * to track `open` is enough for what these tests are about — WHICH method gets called, WHEN, and
 * what the component emits. Whether the browser then supplies a real focus trap is not something
 * jsdom could answer anyway; `design-invariants.spec.ts` and the e2e suite cover the real thing.
 */
beforeAll(() => {
  const proto = HTMLDialogElement?.prototype ?? (globalThis as never)
  if (!('showModal' in proto)) {
    Object.assign(HTMLDialogElement.prototype, {
      showModal(this: HTMLDialogElement) {
        this.open = true
      },
      show(this: HTMLDialogElement) {
        this.open = true
      },
      close(this: HTMLDialogElement) {
        this.open = false
        this.dispatchEvent(new Event('close'))
      },
    })
  }
})

function make(open = false) {
  return mount(ConfirmDialog, {
    props: { open, title: 'Delete this?', body: 'Gone for good.', confirmLabel: 'Delete it' },
    global: { plugins: [i18n] },
    attachTo: document.body,
  })
}

describe('ConfirmDialog (#1594)', () => {
  it('stays closed until asked', () => {
    const w = make(false)
    expect((w.element as HTMLDialogElement).open).toBe(false)
  })

  it('opens MODALLY, not as a non-modal popover', async () => {
    // `show()` renders the dialog without a focus trap, without the top layer and without making
    // the background inert — the destructive action would sit over a fully interactive page. The
    // distinction is invisible on screen and total in behaviour, so it is asserted directly.
    const w = make(false)
    const spy = vi.spyOn(w.element as HTMLDialogElement, 'showModal')
    await w.setProps({ open: true })
    expect(spy).toHaveBeenCalled()
  })

  it('focuses Cancel, not the destructive button', async () => {
    // A confirm that opens with Delete focused converts "tap, tap" muscle memory into a deletion:
    // it adds a step without adding a decision, which is worse than having no dialog, because now
    // the user believes they are protected.
    const w = make(false)
    await w.setProps({ open: true })
    await new Promise((r) => setTimeout(r, 0))
    expect(document.activeElement).toBe(w.get('[data-testid="confirm-cancel"]').element)
  })

  it('emits confirm only from the confirm button', async () => {
    const w = make(true)
    await w.get('[data-testid="confirm-accept"]').trigger('click')
    expect(w.emitted('confirm')).toHaveLength(1)
    expect(w.emitted('cancel')).toBeUndefined()
  })

  it('emits cancel from the cancel button', async () => {
    const w = make(true)
    await w.get('[data-testid="confirm-cancel"]').trigger('click')
    expect(w.emitted('cancel')).toHaveLength(1)
    expect(w.emitted('confirm')).toBeUndefined()
  })

  it('treats an Escape-driven close as a cancel', async () => {
    // The browser closes the dialog on Escape without telling the parent. Without this, the parent
    // keeps believing the dialog is open — and because `open` is what drives it, the NEXT delete
    // would show no confirmation at all. That failure appears one action later than its cause.
    const w = make(false)
    await w.setProps({ open: true })
    ;(w.element as HTMLDialogElement).close()
    await w.vm.$nextTick()
    expect(w.emitted('cancel')).toHaveLength(1)
  })

  it('does not emit cancel when the parent closes it after confirming', async () => {
    // The parent clears its pending id on confirm, which sets `open` false and closes the dialog.
    // If that path also emitted `cancel`, every confirmed delete would be followed by a spurious
    // cancel — harmless here, but the kind of thing a parent later hangs cleanup off.
    const w = make(true)
    await w.setProps({ open: false })
    expect(w.emitted('cancel')).toBeUndefined()
  })
})
