import { flushPromises, mount } from "@vue/test-utils"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"

import en from "../i18n/locales/en.json"
import ShareMenu from "./ShareMenu.vue"
import type { EntityCardModel } from "../composables/entityShareCard"

const shareEntityCard = vi.fn().mockResolvedValue(undefined)
vi.mock("../composables/entityShareCard", () => ({
  entityCopyText: () => "Risk\nhttps://closelistening.app/topic/risk",
  shareEntityCard: (m: unknown) => shareEntityCard(m),
}))
const copyText = vi.fn().mockResolvedValue(true)
vi.mock("../utils/clipboard", () => ({ copyText: (s: string) => copyText(s) }))

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const WITH_URL = { kicker: "Topic", title: "Risk", url: "https://closelistening.app/topic/risk" }
const NO_URL = { kicker: "Organization", title: "The Fed" }

function mountMenu(model: EntityCardModel) {
  // The menu teleports to <body> via the shared popover shell; stub teleport so it renders inline
  // for `find`, and attach to the document so the outside-pointer/Escape dismissal is real.
  return mount(ShareMenu, {
    props: { model, targetKind: 'topic' as const },
    attachTo: document.body,
    global: { plugins: [i18n], stubs: { teleport: true } },
  })
}

afterEach(() => {
  vi.clearAllMocks()
  document.body.innerHTML = ""
})

describe("ShareMenu (#2036)", () => {
  it("is closed until the affordance is clicked, then shows the three modes", async () => {
    const w = mountMenu(WITH_URL)
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(false)
    expect(w.get('[data-testid="share-menu"]').attributes("aria-expanded")).toBe("false")

    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(true)
    expect(w.find('[data-testid="share-card"]').exists()).toBe(true)
    expect(w.find('[data-testid="share-link"]').exists()).toBe(true)
    expect(w.find('[data-testid="share-text"]').exists()).toBe(true)
  })

  it("hides Copy link when the model has no url", async () => {
    const w = mountMenu(NO_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.find('[data-testid="share-link"]').exists()).toBe(false)
    expect(w.find('[data-testid="share-card"]').exists()).toBe(true)
  })

  it("Share card shares then closes the menu", async () => {
    const w = mountMenu(WITH_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-card"]').trigger("click")
    expect(shareEntityCard).toHaveBeenCalledWith(WITH_URL)
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(false)
  })

  it("names the two copy actions for what they do (operator 2026-10-05)", async () => {
    // "Share link" and "Share text" both opened the OS share sheet on a phone — beta testers could
    // not tell the three apart.
    const w = mountMenu(WITH_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.get('[data-testid="share-link"]').text()).toBe("Copy link")
    expect(w.get('[data-testid="share-text"]').text()).toBe("Copy text")
  })

  it("Copy link puts the link on the clipboard and says so", async () => {
    const w = mountMenu(WITH_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-link"]').trigger("click")
    await flushPromises()
    expect(copyText).toHaveBeenCalledWith(WITH_URL.url)
    expect(w.get('[role="status"]').text()).toBe("Link copied")
  })

  it("Copy text puts the name + link on the clipboard and says so", async () => {
    const w = mountMenu(WITH_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-text"]').trigger("click")
    await flushPromises()
    expect(copyText).toHaveBeenCalledWith("Risk\nhttps://closelistening.app/topic/risk")
    expect(w.get('[role="status"]').text()).toBe("Text copied")
  })

  it("claims nothing when the clipboard refused", async () => {
    copyText.mockResolvedValueOnce(false)
    const w = mountMenu(WITH_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-link"]').trigger("click")
    await flushPromises()
    expect(w.find('[role="status"]').exists()).toBe(false)
  })

  it("Escape closes the open menu", async () => {
    const w = mountMenu(WITH_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(true)
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await w.vm.$nextTick()
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(false)
  })
})
