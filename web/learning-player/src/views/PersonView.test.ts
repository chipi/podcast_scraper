import { flushPromises, mount } from "@vue/test-utils"
import { createPinia, setActivePinia } from "pinia"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import en from "../i18n/locales/en.json"
import type { PersonCard } from "../services/types"
import { useAuthStore } from "../stores/auth"
import PersonView from "./PersonView.vue"

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", name: "home", component: { template: "<div/>" } },
      { path: "/person/:id", name: "person", component: PersonView, props: true },
      // Cross-links from EntityCardBody push their own back-stack, not routes,
      // BUT searchLibrary / episode links / cross-open topic pushes do resolve
      // through the router — register stubs so tests don't blow up.
      { path: "/search", name: "search", component: { template: "<div/>" } },
      { path: "/episode/:slug", name: "player", component: { template: "<div/>" } },
      { path: "/podcast/:feedId", name: "podcast", component: { template: "<div/>" } },
      { path: "/topic/:id", name: "topic", component: { template: "<div/>" }, props: true },
    ],
  })
}

beforeEach(() => {
  vi.spyOn(api, "getEntitySignals").mockResolvedValue({})
  vi.spyOn(api, "getUserInterests").mockResolvedValue([])
  vi.spyOn(api, "getPersonCard").mockResolvedValue({
    id: "person:jane-doe",
    label: "Jane Doe",
    role: "host",
    episode_count: 5,
    episodes: [],
    related_people: [{ id: "person:bob-smith", name: "Bob Smith", kind: "person" }],
    related_topics: [
      { id: "topic:ai", label: "AI", cluster_id: null, cluster_label: null, cluster_size: 0 },
    ],
  })
})
afterEach(() => vi.restoreAllMocks())

async function mountPerson(id = "person:jane-doe") {
  setActivePinia(createPinia())
  const router = makeRouter()
  await router.push({ name: "person", params: { id } })
  await router.isReady()
  const w = mount(PersonView, {
    props: { id },
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  await flushPromises()
  return { w, router }
}

describe("PersonView (#1261-6)", () => {
  it("fetches the person card via the route param and renders the person label", async () => {
    const { w } = await mountPerson()
    expect(api.getPersonCard).toHaveBeenCalledWith("person:jane-doe")
    expect(w.find('[data-testid="person-view"]').exists()).toBe(true)
    expect(w.text()).toContain("Jane Doe")
  })

  it("the control at the ROOT of the page DISMISSES — an ✕, not a back arrow", async () => {
    // The page is the destination; nothing contains it, so the dismiss control is a Close (✕), not
    // a Back (‹). The control is now glyph-only in the header actions row — assert via its label.
    const { w } = await mountPerson()
    const dismiss = w.find('[data-testid="ec-dismiss"]')
    expect(dismiss.exists()).toBe(true)
    expect(dismiss.attributes("aria-label")).toBe("Close")
    expect(w.find('[data-testid="ec-open-in-page"]').exists()).toBe(false)
  })

  it("renders the person role badge for a signed-in view", async () => {
    const { w } = await mountPerson()
    // Role badge always renders (auth-independent — mirrors operator viewer).
    const roleBadge = w.find('[data-testid="ec-person-role"]')
    expect(roleBadge.exists()).toBe(true)
    expect(roleBadge.attributes("data-role")).toBe("host")
    expect(roleBadge.text().toLowerCase()).toContain("host")
  })

  it("related pills: five at most — a storyline, a theme, then topics — and \"+N more\" for the rest (operator 2026-10-08)", async () => {
    const topics = Array.from({ length: 8 }, (_, i) => ({
      id: `topic:t${i}`,
      label: `T${i}`,
      cluster_id: i < 3 ? `tc:c${i}` : null,
      cluster_label: i < 3 ? `C${i}` : null,
      cluster_size: 2,
      storyline_id: i === 0 ? "thc:s0" : null,
      storyline_label: i === 0 ? "S0" : null,
    }))
    vi.spyOn(api, "getPersonCard").mockResolvedValue({
      id: "person:jane-doe",
      label: "Jane Doe",
      role: "host",
      episode_count: 5,
      episodes: [],
      related_people: [],
      related_topics: topics,
    } as never)
    const { w } = await mountPerson()
    const count = (id: string) => w.findAll(`[data-testid="${id}"]`).length
    expect(count("ec-person-related-storyline")).toBe(1)
    expect(count("ec-person-related-theme")).toBe(1)
    expect(count("ec-person-related-topic")).toBe(3)
    // 1 storyline + 3 themes + 8 topics = 12, five shown.
    const more = w.get('[data-testid="ec-person-related-more"]')
    expect(more.text()).toContain("7")
    await more.trigger("click")
    expect(count("ec-person-related-theme")).toBe(3)
    expect(count("ec-person-related-topic")).toBe(8)
    expect(w.find('[data-testid="ec-person-related-more"]').exists()).toBe(false)
  })

  it("renders related-topic chips linking (via internal open() stack push) to a topic card", async () => {
    vi.spyOn(api, "getTopicCard").mockResolvedValue({
      id: "topic:ai",
      label: "AI",
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      sibling_topics: [],
      episode_count: 0,
      episodes: [],
      related_people: [],
    })
    const { w } = await mountPerson()
    // The pill names its kind ("Topic") — the related group mixes topics with their themes and
    // storylines (2026-10-05).
    const topicChip = w.findAll('[data-testid="ec-person-related-topic"]').find((b) => b.text().endsWith("AI"))
    expect(topicChip).toBeTruthy()
    await topicChip!.trigger("click")
    await flushPromises()
    // Internal stack push — the header title flips to the topic.
    expect(w.text()).toContain("AI")
    // Back (‹) returns to the person origin — glyph-only dismiss control (aria-label "Back" nested).
    const dismiss = w.find('[data-testid="ec-dismiss"]')
    expect(dismiss.attributes("aria-label")).toBe("Back")
    await dismiss.trigger("click")
    await flushPromises()
    expect(w.text()).toContain("Jane Doe")
  })

  it("follow button surfaces for a signed-in user and toggles interest via addInterest", async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: "u1", email: "a@b", name: "A" }
    const addInterest = vi.spyOn(api, "addInterest").mockResolvedValue(["person:jane-doe"])
    const router = makeRouter()
    await router.push({ name: "person", params: { id: "person:jane-doe" } })
    await router.isReady()
    const w = mount(PersonView, {
      props: { id: "person:jane-doe" },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    await flushPromises()
    const followBtn = w.findAll("button").find((b) => /^\+?\s*Follow/.test(b.text()))!
    expect(followBtn.text()).toContain("Follow")
    await followBtn.trigger("click")
    await flushPromises()
    expect(addInterest).toHaveBeenCalledWith("person:jane-doe")
  })

  it("failed load shows the notFound copy — does not crash", async () => {
    vi.spyOn(api, "getPersonCard").mockRejectedValueOnce(new Error("offline"))
    const { w } = await mountPerson()
    expect(w.text()).toContain("Nothing to show for this yet.")
  })

  it("keeps the PersonCard type import in scope for cross-file symmetry", () => {
    const _: PersonCard | null = null
    expect(_).toBeNull()
  })
})
