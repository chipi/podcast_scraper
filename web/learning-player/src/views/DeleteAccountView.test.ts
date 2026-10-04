import { flushPromises, mount } from "@vue/test-utils"
import { createPinia, setActivePinia } from "pinia"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import en from "../i18n/locales/en.json"
import { useAuthStore } from "../stores/auth"
import DeleteAccountView from "./DeleteAccountView.vue"

/** #2273 — delete account: typed confirmation, the right account named, local state cleared. */

const sequence: string[] = []
vi.mock("../services/contentCache", async (orig) => ({
  ...(await orig<typeof import("../services/contentCache")>()),
  clearCached: async () => void sequence.push("clearCached"),
}))
vi.mock("../services/deviceStore", async (orig) => ({
  ...(await orig<typeof import("../services/deviceStore")>()),
  getDeviceJson: async () => null,
  setDeviceJson: async () => {},
  removeDeviceKey: async () => {},
}))

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/account/delete", name: "account-delete", component: DeleteAccountView },
      { path: "/welcome", name: "landing", component: { template: "<div/>" } },
      { path: "/login", name: "login", component: { template: "<div/>" } },
    ],
  })
}

async function mountSignedIn(provider = "apple") {
  const auth = useAuthStore()
  auth.user = {
    user_id: "u1",
    email: "x@privaterelay.appleid.com",
    name: "Ada",
    provider,
  } as never
  vi.spyOn(auth, "logout").mockImplementation(async () => {
    sequence.push("logout")
  })
  const router = makeRouter()
  await router.push("/account/delete")
  const w = mount(DeleteAccountView, { global: { plugins: [i18n, router] } })
  await flushPromises()
  return { w, router }
}

describe("DeleteAccountView", () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    sequence.length = 0
    vi.restoreAllMocks()
  })

  it("names the account that will be deleted — one address can hold several", async () => {
    const { w } = await mountSignedIn("apple")
    expect(w.get('[data-testid="delete-account-who"]').text()).toBe(
      "You're signed in with Apple as x@privaterelay.appleid.com. This deletes that account.",
    )
  })

  it("will not delete until DELETE is typed exactly", async () => {
    const del = vi.spyOn(api, "deleteAccount").mockResolvedValue()
    const { w } = await mountSignedIn()
    const button = w.get('[data-testid="delete-account-submit"]')
    for (const attempt of ["", "delete", "DELET", "DELETE NOW"]) {
      await w.get('[data-testid="delete-account-confirm"]').setValue(attempt)
      expect(button.attributes("disabled"), attempt).toBeDefined()
    }
    await w.get("form").trigger("submit")
    expect(del).not.toHaveBeenCalled()
  })

  it("deletes, clears this device's content BEFORE the identity, and says so on the landing", async () => {
    const del = vi.spyOn(api, "deleteAccount").mockResolvedValue()
    const { w, router } = await mountSignedIn()
    await w.get('[data-testid="delete-account-confirm"]').setValue("DELETE")
    await w.get("form").trigger("submit")
    await flushPromises()
    expect(del).toHaveBeenCalledWith("DELETE")
    expect(sequence).toEqual(["clearCached", "logout"])
    expect(router.currentRoute.value.name).toBe("landing")
    expect(router.currentRoute.value.query.deleted).toBe("1")
  })

  it("on failure keeps the person here, signed in, with an error", async () => {
    vi.spyOn(api, "deleteAccount").mockRejectedValue(new api.ApiError(500, "boom"))
    const { w, router } = await mountSignedIn()
    await w.get('[data-testid="delete-account-confirm"]').setValue("DELETE")
    await w.get("form").trigger("submit")
    await flushPromises()
    expect(w.find('[data-testid="delete-account-error"]').exists()).toBe(true)
    expect(sequence).toEqual([])
    expect(router.currentRoute.value.name).toBe("account-delete")
  })

  it("signed out (the Play listing's link): explains how, and offers sign-in back to this page", async () => {
    const router = makeRouter()
    await router.push("/account/delete")
    const w = mount(DeleteAccountView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.get('[data-testid="delete-account-signed-out"]').text()).toContain("info@closelistening.app")
    expect(w.find('[data-testid="delete-account-submit"]').exists()).toBe(false)
    // Play's "Manage app data" link points here too: partial deletion must be explained.
    expect(w.get('[data-testid="delete-account-partial"]').text()).toContain("Clear listening history")
    expect(w.get("a").attributes("href")).toBe("/login?redirect=/account/delete")
  })
})
