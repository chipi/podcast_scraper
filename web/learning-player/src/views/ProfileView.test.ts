import { flushPromises, mount } from "@vue/test-utils"
import { createPinia, setActivePinia } from "pinia"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import * as push from "../composables/usePushSubscription"
import en from "../i18n/locales/en.json"
import type { CommsSettings, InterestCluster, UserStats } from "../services/types"
import { useAuthStore } from "../stores/auth"
import { useUserPreferencesStore } from "../stores/userPreferences"
import ProfileView from "./ProfileView.vue"
import AvatarCropModal from "../components/AvatarCropModal.vue"

// Native, so DeviceSettings would actually RENDER if it were still on this page. Without this the
// absence assertion below passes on a component that renders nothing off-native — it would hold
// whether or not Device had been moved, which is no assertion at all.
// `isNative: true` makes DeviceSettings render, which is what the placement test needs — but a
// partial mock of this module means every OTHER function it exports is undefined, and sign-out
// calls `storeAuthToken`. That surfaced as an unhandled error rather than a failure, so the suite
// stayed green while a test was throwing.
vi.mock("../services/native", async (orig) => ({
  ...(await orig<typeof import("../services/native")>()),
  isNative: () => true,
}))
vi.mock("../services/downloadScheduler", () => ({
  DEFAULT_POLICY: "wifi-only",
  applyDownloadCap: async () => {},
  getNetworkPolicy: async () => "wifi-only",
  setNetworkPolicy: async () => {},
}))
// Partial over the real module, for the same reason as the native mock above: sign-out calls
// `removeDeviceKey`, and a bare factory makes every unlisted export undefined.
vi.mock("../services/deviceStore", async (orig) => ({
  ...(await orig<typeof import("../services/deviceStore")>()),
  getDeviceJson: async () => null,
  setDeviceJson: async () => {},
}))

// ONE shared log, written by both the cache mock and the logout spy — two separate arrays could
// only show that both ran, never in which order, which is the whole claim being tested.
const sequence: string[] = []
vi.mock("../services/contentCache", async (orig) => {
  const actual = await orig<typeof import("../services/contentCache")>()
  return {
    ...actual,
    clearCached: async (...args: unknown[]) => {
      sequence.push("clearCached")
      void args
    },
  }
})

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/profile", name: "profile", component: ProfileView },
    { path: "/settings", name: "settings", component: { template: "<div/>" } },
    { path: "/", name: "home", component: { template: "<div/>" } },
    { path: "/catalog", name: "catalog", component: { template: "<div/>" } },
  ],
})

const clusters: InterestCluster[] = [{ id: "tc:ai", label: "AI", size: 12 }]

function stats(over: Partial<UserStats> = {}): UserStats {
  return {
    episodes: 8,
    shows: 3,
    listening_seconds: 7200,
    active_days: 5,
    day_streak: 4,
    daily: [
      { date: "2024-03-01", count: 2 },
      { date: "2024-03-02", count: 1 },
    ],
    ...over,
  }
}

function channels(over: Partial<Record<"email" | "push" | "in_app", boolean>> = {}) {
  return { email: false, push: false, in_app: true, ...over }
}
function comms(over: Partial<CommsSettings> = {}): CommsSettings {
  return {
    types: {
      digest: channels(),
      new_episodes: channels(),
      product: channels(),
    },
    digest_schedule: { cadence: "weekly", day_of_week: 6, hour: 13, paused: false },
    email_verified: true,
    unsubscribe_ref: null,
    ...over,
  }
}

/**
 * Every ProfileView mounted by this file, torn down after each test.
 *
 * Not tidiness — correctness, and the same trap `PlayerView.test.ts` documents. A wrapper that is
 * never unmounted keeps its watchers alive, so a LATER test re-mocking an endpoint to reject makes
 * the zombie re-run `load()` against it, with nobody awaiting the result. That surfaced as an
 * unhandled rejection attributed to the new test's mock, in a component whose own catch was fine —
 * it cost a long hunt for a consumer that did not exist.
 */
const mountedProfiles: Array<{ unmount: () => void }> = []
afterEach(() => {
  while (mountedProfiles.length) mountedProfiles.pop()!.unmount()
})

function mountProfile() {
  setActivePinia(createPinia())
  const auth = useAuthStore()
  auth.user = { user_id: "u_1", email: "dev@localhost", name: "Dev" }
  const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
  mountedProfiles.push(w)
  return w
}

beforeEach(() => {
  vi.spyOn(api, "getTopClusters").mockResolvedValue(clusters)
  vi.spyOn(api, "getMyStats").mockResolvedValue(stats())
  vi.spyOn(api, "getComms").mockResolvedValue(comms())
})
afterEach(() => vi.restoreAllMocks())

describe("ProfileView — Settings entry (#8)", () => {
  it("no longer carries the Device section — that belongs to Settings", async () => {
    // Profile is about me as a user. Download network policy and the size cap are about the
    // handset, and are shared by every account that signs in on it.
    const w = await mountProfile()
    await flushPromises()
    expect(w.find('[data-testid="device-settings"]').exists(), "Device is still on Profile").toBe(
      false
    )
  })

  // A picked file now opens the crop modal; the upload happens on the modal's `confirm` (with the
  // CROPPED blob), not on pick. jsdom has no canvas/objectURL, so we drive the modal's `confirm`
  // event directly with a blob rather than clicking through the canvas.
  async function pickFileAndOpenCrop(w: ReturnType<typeof mount>): Promise<void> {
    const input = w.get('[data-testid="avatar-file-input"]')
    const file = new File([new Uint8Array([1, 2, 3])], "a.png", { type: "image/png" })
    Object.defineProperty(input.element, "files", { value: [file], configurable: true })
    await input.trigger("change")
    await flushPromises()
  }

  it("opens the crop modal on pick, then uploads the cropped blob and refreshes /me (Area E)", async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: "u_1", email: "dev@localhost", name: "Dev" }
    const upload = vi.spyOn(api, "uploadAvatar").mockResolvedValue({ image: "/api/app/x/avatar" })
    const refresh = vi.spyOn(auth, "refresh").mockResolvedValue()
    const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
    mountedProfiles.push(w)
    await pickFileAndOpenCrop(w)
    const modal = w.findComponent(AvatarCropModal)
    expect(modal.exists(), "crop modal opens on pick").toBe(true)

    const blob = new Blob([new Uint8Array([9, 9])], { type: "image/png" })
    modal.vm.$emit("confirm", blob)
    await flushPromises()
    expect(upload).toHaveBeenCalledWith(blob) // the cropped blob, not the raw file
    expect(refresh).toHaveBeenCalled()
    expect(w.findComponent(AvatarCropModal).exists(), "modal closes after confirm").toBe(false)
    expect(w.find('[data-testid="avatar-error"]').exists()).toBe(false)
  })

  it("surfaces an error when the cropped avatar upload is rejected", async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: "u_1", email: "dev@localhost", name: "Dev" }
    vi.spyOn(api, "uploadAvatar").mockRejectedValue(new api.ApiError(413, "too big"))
    const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
    mountedProfiles.push(w)
    await pickFileAndOpenCrop(w)
    w.findComponent(AvatarCropModal).vm.$emit("confirm", new Blob([new Uint8Array([1])]))
    await flushPromises()
    expect(w.get('[data-testid="avatar-error"]').text()).toContain("Couldn")
  })

  it("closes the crop modal without uploading when cancelled", async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: "u_1", email: "dev@localhost", name: "Dev" }
    const upload = vi.spyOn(api, "uploadAvatar").mockResolvedValue({ image: "/x" })
    const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
    mountedProfiles.push(w)
    await pickFileAndOpenCrop(w)
    w.findComponent(AvatarCropModal).vm.$emit("cancel")
    await flushPromises()
    expect(w.findComponent(AvatarCropModal).exists()).toBe(false)
    expect(upload).not.toHaveBeenCalled()
  })

  it("shows the immutable @handle when present (Area E)", async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: "u_1", email: "dev@localhost", name: "Dev", username: "jane_doe" }
    const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
    mountedProfiles.push(w)
    expect(w.get('[data-testid="profile-handle"]').text()).toBe("@jane_doe")
  })

  it("links to the Settings screen via the gear", async () => {
    vi.spyOn(api, "getUserInterests").mockResolvedValue([])
    const w = mountProfile()
    const gear = w.find('[data-testid="profile-settings-link"]')
    expect(gear.exists()).toBe(true)
    expect(gear.attributes("href")).toBe("/settings")
  })
})

describe("ProfileView — Your Week layout", () => {
  it("reflects the saved layout preference and persists a change", async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: "u_1", email: "dev@localhost", name: "Dev" }
    const prefs = useUserPreferencesStore()
    vi.spyOn(prefs, "hydrate").mockResolvedValue()
    vi.spyOn(prefs, "get").mockReturnValue("full") // a saved 'full' preference
    const setSpy = vi.spyOn(prefs, "set").mockResolvedValue()
    vi.spyOn(api, "getUserInterests").mockResolvedValue([])
    const w = mount(ProfileView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    // Initial state reflects the saved pref: the Full button is the active one.
    //
    // Asserted on the ARIA state, not on a background class (#1959): the selected pill is styled
    // FROM that attribute, so the accessible state and the visible state cannot drift apart.
    //
    // The attribute is `aria-checked` now, not `aria-selected` (#1594 item 7). This control sets a
    // saved preference and switches no region, so it is a radiogroup rather than a tablist — "tab"
    // was the wrong announcement. This assertion is what caught the conversion breaking the
    // VISIBLE state: the CSS keyed the fill off `aria-selected` alone, so the option kept working
    // and quietly stopped looking selected. Exactly the drift the note above was written about.
    const fullBtn = w.findAll("button").find((b) => b.text() === "Full")!
    expect(fullBtn.attributes("aria-checked")).toBe("true")
    expect(fullBtn.attributes("role")).toBe("radio")
    const compactInitially = w.findAll("button").find((b) => b.text() === "Compact")!
    expect(compactInitially.attributes("aria-checked")).toBe("false")

    // Switching to Compact persists the change under the shared key.
    const compactBtn = w.findAll("button").find((b) => b.text() === "Compact")!
    await compactBtn.trigger("click")
    expect(setSpy).toHaveBeenCalledWith("lp.yourweek.layout", "compact")
  })
})

describe("ProfileView — interest chips", () => {
  it("renders chips hued by kind: person → text-person, topic/cluster → text-topic", async () => {
    vi.spyOn(api, "getUserInterests").mockResolvedValue([
      "tc:ai",
      "topic:personal-growth",
      "person:brian-chesky",
    ])
    const w = mountProfile()
    await flushPromises()

    const chips = w
      .findAll("span")
      .filter((s) => s.classes().includes("text-person") || s.classes().includes("text-topic"))
    const personChip = chips.find((c) => c.classes().includes("text-person"))!
    const topicChips = chips.filter((c) => c.classes().includes("text-topic"))

    // person:brian-chesky → person hue, de-slugged label
    expect(personChip.classes()).toContain("text-person")
    expect(personChip.text()).toBe("brian chesky")

    // topic:personal-growth → topic hue, de-slugged
    expect(topicChips.some((c) => c.text() === "personal growth")).toBe(true)
    // tc:ai resolves to its cluster label via the clusters map (not de-slugged "ai")
    expect(topicChips.some((c) => c.text() === "AI")).toBe(true)
  })

  it("shows the no-interests message when the list is empty", async () => {
    vi.spyOn(api, "getUserInterests").mockResolvedValue([])
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain("No interests chosen yet.")
  })
})

describe("ProfileView — Your listening panel", () => {
  it("renders streak / episodes / shows when episodes > 0", async () => {
    vi.spyOn(api, "getUserInterests").mockResolvedValue([])
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain("Your activity")
    expect(w.text()).toContain("Day streak")
    expect(w.text()).toContain("Episodes")
    expect(w.text()).toContain("Shows")
    // NO "Hours" tile here any more (#1914). It rendered `sum(position_seconds)` — a lifetime
    // snapshot of furthest position reached, which rises on a forward seek and does not move on a
    // re-listen. Time actually listened lives in ListeningRecap, with its coverage stated.
    expect(w.text()).not.toContain("Hours")
    // 4-day streak + 8 episodes + 3 shows surface their numbers.
    expect(w.text()).toContain("4")
    expect(w.text()).toContain("8")
    expect(w.text()).toContain("3")
  })

  it("shows the stats empty state when the user has no episodes", async () => {
    vi.spyOn(api, "getUserInterests").mockResolvedValue([])
    vi.spyOn(api, "getMyStats").mockResolvedValue(stats({ episodes: 0 }))
    const w = mountProfile()
    await flushPromises()
    expect(w.text()).toContain("Start listening to build your stats.")
    expect(w.text()).not.toContain("Day streak")
  })
})

describe("ProfileView — notifications", () => {
  beforeEach(() => vi.spyOn(api, "getUserInterests").mockResolvedValue([]))

  it("renders the layout switch + the type×channel matrix; cadence hidden until digest email on", async () => {
    const w = mountProfile()
    await flushPromises()
    // The in-app view is primary: the layout switch shows first.
    expect(w.text()).toContain("On your home")
    expect(w.text()).toContain("Compact")
    expect(w.text()).toContain("Full")
    // The matrix: every type row + every channel header.
    expect(w.text()).toContain("Your Week")
    expect(w.text()).toContain("New episodes")
    expect(w.text()).toContain("App updates")
    expect(w.text()).toContain("Email")
    expect(w.text()).toContain("Push")
    expect(w.text()).toContain("In-app")
    // A cell for every type × channel.
    expect(w.find('[data-testid="notif-digest-email"]').exists()).toBe(true)
    expect(w.find('[data-testid="notif-new_episodes-push"]').exists()).toBe(true)
    expect(w.find('[data-testid="notif-product-in_app"]').exists()).toBe(true)
    // Cadence is hidden until the digest email cell is on.
    expect(w.text()).not.toContain("Frequency")
  })

  it("enabling the digest email PUTs the whole matrix and reveals the cadence control", async () => {
    const put = vi
      .spyOn(api, "putComms")
      .mockResolvedValue(
        comms({
          types: {
            digest: channels({ email: true }),
            new_episodes: channels(),
            product: channels(),
          },
        })
      )
    const w = mountProfile()
    await flushPromises()

    await w.get('[data-testid="notif-digest-email"]').setValue(true)
    await flushPromises()

    expect(put).toHaveBeenCalledWith({
      types: expect.objectContaining({ digest: expect.objectContaining({ email: true }) }),
    })
    expect(w.text()).toContain("Frequency")
  })

  it("enabling a push cell registers a browser subscription via the composable", async () => {
    const enable = vi.spyOn(push, "enablePush").mockResolvedValue(true)
    vi.spyOn(api, "putComms").mockResolvedValue(
      comms({
        types: { digest: channels(), new_episodes: channels({ push: true }), product: channels() },
      })
    )
    const w = mountProfile()
    await flushPromises()

    await w.get('[data-testid="notif-new_episodes-push"]').setValue(true)
    await flushPromises()

    expect(enable).toHaveBeenCalled()
  })

  it("reverts the push cell when the browser cannot subscribe (no PUT)", async () => {
    vi.spyOn(push, "enablePush").mockResolvedValue(false)
    const put = vi.spyOn(api, "putComms").mockResolvedValue(comms())
    const w = mountProfile()
    await flushPromises()

    const cell = w.get('[data-testid="notif-new_episodes-push"]')
    await cell.setValue(true)
    await flushPromises()

    // The browser refused → the cell reverts and NOTHING is persisted (matrix push stays off).
    expect(put).not.toHaveBeenCalled()
    expect((cell.element as HTMLInputElement).checked).toBe(false)
  })

  it("reverts the push cell when the subscribe POST throws (no desync)", async () => {
    vi.spyOn(push, "enablePush").mockRejectedValue(new Error("network"))
    const put = vi.spyOn(api, "putComms").mockResolvedValue(comms())
    const w = mountProfile()
    await flushPromises()

    const cell = w.get('[data-testid="notif-new_episodes-push"]')
    await cell.setValue(true)
    await flushPromises()

    expect(put).not.toHaveBeenCalled()
    expect((cell.element as HTMLInputElement).checked).toBe(false)
  })

  describe("sign out (#1594)", () => {
    it("lands on Home, not the flat Catalog index", async () => {
      // Catalog is every episode in the corpus in one list. It is a fine place to browse TO and
      // the wrong place to be dropped: Home renders the signed-out hero that explains what the app
      // is, which is the only thing someone who just signed out might want. A bare list reads like
      // a session that half-broke rather than one they ended on purpose.
      vi.spyOn(api, "logout").mockResolvedValue(undefined as never)
      const w = mountProfile()
      await flushPromises()

      const btn = w.findAll("button").find((b) => b.text().includes("Sign out"))
      expect(btn, "no sign-out button rendered — this assertion would be vacuous").toBeTruthy()
      await btn!.trigger("click")
      await flushPromises()

      expect(router.currentRoute.value.name).toBe("home")
    })

    it("clears the cached content BEFORE dropping the identity", async () => {
      // Order matters and is invisible in the UI: a signed-out device must not keep the previous
      // account's library readable on disk (#1909). Swapping the two lines looks harmless in
      // review, so the SEQUENCE is what is asserted — not merely that both ran.
      sequence.length = 0
      vi.spyOn(api, "logout").mockImplementation(async () => {
        sequence.push("logout")
        return undefined as never
      })
      const w = mountProfile()
      await flushPromises()
      await w
        .findAll("button")
        .find((b) => b.text().includes("Sign out"))!
        .trigger("click")
      await flushPromises()

      expect(
        sequence,
        "the cached library must be wiped BEFORE the identity goes, or a signed-out device keeps " +
          "the previous account's content readable"
      ).toEqual(["clearCached", "logout"])
    })
  })

  /**
   * #1591's defect, recurring where nothing was watching: with no network the page told a user
   * with stats and interests that they had neither.
   */
  it("says a STATS load failed rather than claiming you have nothing", async () => {
    vi.spyOn(api, "getMyStats").mockImplementation(() => Promise.reject(new Error("offline")))
    const w = await mountProfile()
    await flushPromises()

    expect(w.find('[data-testid="stats-unavailable"]').exists(), "stats claimed emptiness").toBe(
      true
    )
    expect(w.text()).not.toContain("Start listening to build your stats")
  })

  it("a genuinely empty account still reads as empty, not as broken", async () => {
    // The distinction has to cut both ways or it is just a different lie.
    vi.spyOn(api, "getUserInterests").mockResolvedValue([])
    const w = await mountProfile()
    await flushPromises()
    expect(w.find('[data-testid="interests-unavailable"]').exists()).toBe(false)
    expect(w.text()).toContain("No interests chosen yet")
  })
})

/**
 * Its own describe, deliberately — this is the fix for a bug that was in the TEST FILE.
 *
 * `describe('ProfileView — notifications')` installs `vi.spyOn(api, 'getUserInterests')
 * .mockResolvedValue([])` in a beforeEach. Re-spying that same method inside a test there to make
 * it REJECT produced an unhandled rejection that no consumer of ours owned: the component's own
 * `.catch` demonstrably ran (the unavailable state rendered), and the loose promise came from the
 * two spy configurations overlapping. A long hunt went into looking for a consumer that did not
 * exist — the giveaway, in hindsight, was that the same mock with nothing mounted did not leak,
 * and that stubbing the children made it stop.
 *
 * Outside that describe there is no first spy, and the test is clean.
 */
describe("ProfileView — a failed load is not an empty account", () => {
  it("says an INTERESTS load failed rather than claiming you chose none", async () => {
    vi.spyOn(api, "getUserInterests").mockImplementation(() => Promise.reject(new Error("offline")))
    const w = mountProfile()
    await flushPromises()

    expect(
      w.find('[data-testid="interests-unavailable"]').exists(),
      "interests claimed emptiness"
    ).toBe(true)
    expect(w.text()).not.toContain("No interests chosen yet")
  })

  it("a genuinely empty interests list still reads as empty", async () => {
    // The distinction has to cut both ways or it is just a different lie.
    vi.spyOn(api, "getUserInterests").mockResolvedValue([])
    const w = mountProfile()
    await flushPromises()

    expect(w.find('[data-testid="interests-unavailable"]').exists()).toBe(false)
    expect(w.text()).toContain("No interests chosen yet")
  })
})
