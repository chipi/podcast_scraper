import { mount } from "@vue/test-utils"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"

import en from "../i18n/locales/en.json"
import ShareMenu from "./ShareMenu.vue"

const shareEntityCard = vi.fn().mockResolvedValue(undefined)
const shareEntityLink = vi.fn().mockResolvedValue("shared")
vi.mock("../composables/entityShareCard", () => ({
  entityCardText: () => "caption",
  shareEntityCard: (m: unknown) => shareEntityCard(m),
  shareEntityLink: (m: unknown) => shareEntityLink(m),
}))
vi.mock("../services/native", () => ({ isNative: () => false, saveAndShareText: vi.fn() }))

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const WITH_URL = { kicker: "Topic", title: "Risk", url: "https://closelistening.app/topic/risk" }
const NO_URL = { kicker: "Organization", title: "The Fed" }

function mountMenu(model: object) {
  return mount(ShareMenu, { props: { model }, global: { plugins: [i18n] } })
}

afterEach(() => vi.clearAllMocks())

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

  it("hides Share link when the model has no url", async () => {
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

  it("Escape closes the open menu", async () => {
    const w = mountMenu(WITH_URL)
    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(true)
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await w.vm.$nextTick()
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(false)
  })
})
