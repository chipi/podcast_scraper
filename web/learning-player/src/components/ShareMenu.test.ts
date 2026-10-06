import { flushPromises, mount } from "@vue/test-utils"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"

import en from "../i18n/locales/en.json"
import ShareMenu from "./ShareMenu.vue"

const shareCard = vi.fn().mockResolvedValue(undefined)
vi.mock("../composables/shareCard", () => ({
  shareCard: (...a: unknown[]) => shareCard(...a),
}))
const copyText = vi.fn().mockResolvedValue(true)
vi.mock("../utils/clipboard", () => ({ copyText: (s: string) => copyText(s) }))

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const TOPIC = { kind: "topic" as const, id: "topic:risk", title: "Risk", targetKind: "topic" as const }
const ORG = {
  kind: "organization" as const,
  id: "org:fed",
  title: "The Fed",
  targetKind: "organization" as const,
}

function mountMenu(props: Record<string, unknown>) {
  // The menu teleports to <body> via the shared popover shell; stub teleport so it renders inline
  // for `find`, and attach to the document so the outside-pointer/Escape dismissal is real.
  return mount(ShareMenu, {
    props: props as never,
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
    const w = mountMenu(TOPIC)
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(false)
    expect(w.get('[data-testid="share-menu"]').attributes("aria-expanded")).toBe("false")

    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(true)
    expect(w.find('[data-testid="share-card"]').exists()).toBe(true)
    expect(w.find('[data-testid="share-link"]').exists()).toBe(true)
    expect(w.find('[data-testid="share-text"]').exists()).toBe(true)
  })

  it("hides Copy link for an organization — it has no page of its own", async () => {
    const w = mountMenu(ORG)
    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.find('[data-testid="share-link"]').exists()).toBe(false)
    expect(w.find('[data-testid="share-card"]').exists()).toBe(true)
  })

  it("Share card shares the SERVER card for this kind + id, then closes the menu", async () => {
    // Operator 2026-10-05: one card design — the server's — for everything.
    const w = mountMenu(TOPIC)
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-card"]').trigger("click")
    await flushPromises()
    expect(shareCard).toHaveBeenCalledWith("topic", "topic:risk", "Risk")
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(false)
  })

  it("says so when the card could not be made, rather than doing nothing", async () => {
    shareCard.mockRejectedValueOnce(new Error("offline"))
    const w = mountMenu(TOPIC)
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-card"]').trigger("click")
    await flushPromises()
    expect(w.get('[role="status"]').text()).toBe("Couldn't make the card — try again")
  })

  it("names the two copy actions for what they do (operator 2026-10-05)", async () => {
    const w = mountMenu(TOPIC)
    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.get('[data-testid="share-link"]').text()).toBe("Copy link")
    expect(w.get('[data-testid="share-text"]').text()).toBe("Copy text")
  })

  it.each([
    ["episode", "ep-1", "/episode/ep-1"],
    ["show", "p05", "/podcast/p05"],
    ["theme", "tc:risk", "/theme/tc%3Arisk"],
    ["storyline", "topic:risk", "/storyline/topic%3Arisk"],
  ] as const)("Copy link for a %s opens its own page", async (kind, id, path) => {
    const w = mountMenu({ kind, id, title: "X", targetKind: "topic" })
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-link"]').trigger("click")
    await flushPromises()
    expect(copyText.mock.calls[0][0]).toMatch(new RegExp(`${path.replace(/[.%]/g, "\\$&")}$`))
    expect(w.get('[role="status"]').text()).toBe("Link copied")
  })

  it("Copy text puts the name, what it is, and the link on the clipboard", async () => {
    const w = mountMenu({ ...TOPIC, context: "a recurring topic" })
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-text"]').trigger("click")
    await flushPromises()
    expect(copyText.mock.calls[0][0]).toMatch(/^Risk — a recurring topic\n.*\/topic\/topic%3Arisk$/)
    expect(w.get('[role="status"]').text()).toBe("Text copied")
  })

  it("claims nothing when the clipboard refused", async () => {
    copyText.mockResolvedValueOnce(false)
    const w = mountMenu(TOPIC)
    await w.get('[data-testid="share-menu"]').trigger("click")
    await w.get('[data-testid="share-link"]').trigger("click")
    await flushPromises()
    expect(w.find('[role="status"]').exists()).toBe(false)
  })

  it("Escape closes the open menu", async () => {
    const w = mountMenu(TOPIC)
    await w.get('[data-testid="share-menu"]').trigger("click")
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(true)
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await w.vm.$nextTick()
    expect(w.find('[data-testid="share-menu-list"]').exists()).toBe(false)
  })
})
