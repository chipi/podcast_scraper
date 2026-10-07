import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import * as outbox from '../services/outbox'
import en from '../i18n/locales/en.json'
import type { Collection } from '../services/types'
import { useAuthStore } from '../stores/auth'
import AddToCollectionButton from './AddToCollectionButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/login', name: 'login', component: stub },
  ],
})

function col(over: Partial<Collection> = {}): Collection {
  return { id: 'col_1', name: 'Research', created_at: 1, count: 0, ...over }
}

async function mountIt(signedIn = true) {
  setActivePinia(createPinia())
  await router.push('/')
  await router.isReady()
  const w = mount(AddToCollectionButton, {
    props: { item: { kind: 'episode', ref: 'ep-x' } },
    // The menu teleports to <body> via the shared popover shell; stub teleport so it renders inline
    // for `find`, and attach to the document so outside-pointer/Escape dismissal is real.
    attachTo: document.body,
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  if (signedIn) {
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
  } else {
    useAuthStore().loaded = true
  }
  await flushPromises()
  return w
}

afterEach(() => {
  vi.restoreAllMocks()
  document.body.innerHTML = ''
})

describe('AddToCollectionButton (#1839)', () => {
  it('opens the menu and pins the item to a chosen collection', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ count: 1 }))
    const w = await mountIt()
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="add-to-collection-menu"]').text()).toContain('Research')
    await w.get('[data-testid="add-to-collection-pick"]').trigger('click')
    expect(add).toHaveBeenCalledWith('col_1', { kind: 'episode', ref: 'ep-x' })
  })

  it('every board row states its own colour and its own width', async () => {
    /*
     * The board NAMES did not render on the operator's device (2026-09-27) while the "✓ Added"
     * beside them did. Never reproduced off-device — the data, the DOM and the compiled CSS in
     * Chromium were all measured correct — so what is pinned here is the two properties that make
     * the symptom impossible, rather than a reproduction of it.
     *
     * COLOUR: the name was the only text in this teleported panel inheriting its colour. A panel
     * teleported into an open `<dialog>` inherits the UA's `CanvasText`, so an inheriting control
     * can land black-on-black with a perfectly good theme. Both branches must be explicit.
     *
     * WIDTH: `min-w-0` with `flex-basis: auto` lets this span — and only this one, its sibling is
     * `shrink-0` — shrink to zero, and at zero width `truncate`'s `overflow: hidden` paints no
     * text and no ellipsis. `flex-1` gives it a definite basis.
     *
     * Asserted on the rendered class list rather than the source, so it also covers the branch
     * being picked correctly for held vs unheld rows.
     */
    vi.spyOn(api, 'getCollections').mockResolvedValue([
      col({ id: 'col_1', name: 'AI' }),
      col({ id: 'col_2', name: 'Investments' }),
    ])
    vi.spyOn(api, 'getCollectionsContaining').mockResolvedValue({ ids: ['col_2'], checked: true })
    const w = await mountIt()
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()

    const rows = w.findAll('[data-testid="add-to-collection-pick"]')
    expect(rows, 'expected one row per board').toHaveLength(2)

    const [plain, held] = rows
    // The names are on screen at all — the thing the operator could not see.
    expect(plain.text()).toContain('AI')
    expect(held.text()).toContain('Investments')

    // An explicit colour in BOTH states, never inherited through the teleport.
    expect(plain.classes(), 'an unheld row must state its colour').toContain('text-canvas-foreground')
    expect(held.classes(), 'a held row is grounded, and says so').toContain('text-grounded')

    /*
     * The name must not be CLIPPABLE. Proven on device 2026-09-27: from the second open onwards —
     * once the `shrink-0` "✓ Added" sibling exists — the flex distribution was computed against the
     * wrong container width (a forced sync layout while the panel is still `visibility:hidden`),
     * the name span collapsed to a sliver, and `truncate`'s `overflow:hidden` hid the text rather
     * than letting it spill. Three fixes chased it as a colour bug because clipped and invisible
     * look identical.
     *
     * So this asserts the ABSENCE of the clipping, not the presence of a width. A width can be
     * computed wrong; `overflow: visible` cannot hide anything whatever the width comes out as.
     */
    for (const row of rows) {
      const name = row.get('[data-testid="add-to-collection-board-name"]')
      expect(
        name.classes(),
        'the board name is truncatable again — a mis-measured flex row will clip it to nothing, ' +
          'which is the bug that took three attempts because it looks exactly like invisible text',
      ).not.toContain('truncate')
      expect(name.classes(), 'nor may it be clipped by hand').not.toContain('overflow-hidden')
      expect(name.classes(), 'it wraps instead').toContain('break-words')
    }
  })

  it('creates a new collection and pins into it', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const create = vi.spyOn(api, 'createCollection').mockResolvedValue(col({ id: 'col_2', name: 'New' }))
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ id: 'col_2', count: 1 }))
    const w = await mountIt()
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    await w.get('[data-testid="add-to-collection-new"]').trigger('click')
    await w.find('input[type="text"]').setValue('New')
    await w.find('form').trigger('submit')
    await flushPromises()
    expect(create).toHaveBeenCalledWith('New')
    expect(add).toHaveBeenCalledWith('col_2', { kind: 'episode', ref: 'ep-x' })
  })

  it('signed out: routes to sign-in instead of opening', async () => {
    const push = vi.spyOn(router, 'push')
    const w = await mountIt(false)
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(false)
    expect(push).toHaveBeenCalledWith(expect.objectContaining({ name: 'login' }))
  })
})

describe('failures are visible, not swallowed (#2004 item 13)', () => {
  it('says so when the collection list cannot be loaded', async () => {
    // Was `.catch(() => [])`: a failed load rendered as "you have no collections", which is how a
    // user comes to believe their collections were never saved.
    vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(401, 'nope'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="collection-error"]').text()).toBe(en.collections.loadFailed)
  })

  it('retries the load on the next open instead of latching an empty list', async () => {
    // `loaded` must not latch on failure, or the panel shows an empty list forever in a session.
    const spy = vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(500, 'boom'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.get('button').trigger('click') // close
    spy.mockResolvedValue([col()])
    await w.get('button').trigger('click') // re-open
    await flushPromises()
    expect(spy).toHaveBeenCalledTimes(2)
    expect(w.find('[data-testid="collection-error"]').exists()).toBe(false)
    expect(w.text()).toContain('Research')
  })

  it('keeps a previously-loaded list visible if a LATER open transiently fails (no blanking)', async () => {
    // The native bug (2026-09-15): after collections loaded once, reopening the menu showed an EMPTY
    // list — "second time I click collections it's empty, I have to go to another topic to reset".
    // The list must survive a later flaky fetch: replace on success, never blank on failure.
    const spy = vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    const w = await mountIt()
    await w.get('button').trigger('click') // open → loads Research
    await flushPromises()
    expect(w.text()).toContain('Research')
    await w.get('button').trigger('click') // close
    spy.mockRejectedValue(new api.ApiError(500, 'boom')) // the next fetch fails transiently
    await w.get('button').trigger('click') // reopen → refetch fails
    await flushPromises()
    expect(spy).toHaveBeenCalledTimes(2) // it DID refetch (no stale latch)
    expect(w.text()).toContain('Research') // …but the good list is still shown
    expect(w.find('[data-testid="collection-error"]').exists()).toBe(false) // no false "load failed"
  })

  it('says so when adding to a collection fails, and claims nothing', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    // 422 = a REFUSAL. A 500 is transient and is now queued for replay, not surfaced.
    vi.spyOn(api, 'addToCollection').mockRejectedValue(new api.ApiError(422, 'refused'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.findAll('button').filter((b) => b.text().includes('Research'))[0].trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="collection-error"]').text()).toBe(en.collections.addFailed)
    expect(w.text()).not.toContain('✓')
  })

  it('says so when creating fails, and KEEPS the typed name', async () => {
    // Retyping a name the app lost is the app's mistake charged to the user.
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const create = vi.spyOn(api, 'createCollection').mockRejectedValue(new api.ApiError(422, 'refused'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.get('[data-testid="add-to-collection-new"]').trigger('click')
    const input = w.get('input')
    await input.setValue('Tech')
    await w.get('form').trigger('submit')
    await flushPromises()
    expect(create).toHaveBeenCalledWith('Tech')
    expect(w.get('[data-testid="collection-error"]').text()).toBe(en.collections.createFailed)
    expect((input.element as HTMLInputElement).value).toBe('Tech')
  })
})

describe('menu a11y (#2004 #9)', () => {
  it('exposes aria-expanded and closes on Escape / outside-click', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    const w = await mountIt()
    const trigger = w.get('[data-testid="add-to-collection"]')
    expect(trigger.attributes('aria-haspopup')).toBe('true')
    expect(trigger.attributes('aria-expanded')).toBe('false')

    await trigger.trigger('click')
    await flushPromises()
    expect(trigger.attributes('aria-expanded')).toBe('true')
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(true)

    // Escape closes.
    await w.get('div.relative').trigger('keydown', { key: 'Escape' })
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(false)
    expect(trigger.attributes('aria-expanded')).toBe('false')

    // Reopen, then an outside pointerdown closes.
    await trigger.trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(true)
    document.dispatchEvent(new Event('pointerdown'))
    await flushPromises()
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(false)
  })
})

describe('a transient failure is QUEUED, not lost (#2004 item 13 — root cause)', () => {
  it('queues a create that failed transiently, and shows it immediately', async () => {
    // Collections was the only per-user write with no outbox: favourites, queue, highlights, notes
    // and follows all replay. That asymmetry is why a flaky moment was invisible everywhere else and
    // permanent here — not a collections-specific bug.
    const enqueued: unknown[] = []
    vi.spyOn(outbox, 'enqueue').mockImplementation((a) => void enqueued.push(a))
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    vi.spyOn(api, 'createCollection').mockRejectedValue(new api.ApiError(503, 'gateway'))

    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.get('[data-testid="add-to-collection-new"]').trigger('click')
    await w.get('input').setValue('Tech')
    await w.get('form').trigger('submit')
    await flushPromises()

    // The offline "create AND add" queues BOTH halves (#2004 #5): the create with a client-minted id,
    // and the pin targeting that same id — the server honours the id on replay so the pin lands.
    // Queuing only the create (as before) silently dropped the item the user was adding.
    expect(enqueued).toHaveLength(2)
    expect(enqueued[0]).toMatchObject({ op: 'collection.create', name: 'Tech' })
    const clientId = (enqueued[0] as { clientId: string }).clientId
    expect(clientId).toMatch(/^col_[a-z0-9]+$/)
    expect(enqueued[1]).toMatchObject({ op: 'collection.addItem', collectionId: clientId })
    expect(w.text()).toContain('Tech')
    expect(w.find('[data-testid="collection-error"]').exists()).toBe(false)
  })

  it('queues an add that failed transiently', async () => {
    const enqueued: unknown[] = []
    vi.spyOn(outbox, 'enqueue').mockImplementation((a) => void enqueued.push(a))
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    vi.spyOn(api, 'addToCollection').mockRejectedValue(new api.ApiError(503, 'gateway'))

    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.findAll('button').filter((b) => b.text().includes('Research'))[0].trigger('click')
    await flushPromises()

    expect(enqueued[0]).toMatchObject({ op: 'collection.addItem', collectionId: 'col_1' })
    expect(w.find('[data-testid="collection-error"]').exists()).toBe(false)
  })
})
