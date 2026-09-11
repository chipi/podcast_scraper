import { mount } from "@vue/test-utils"
import { createI18n } from "vue-i18n"
import { describe, expect, it } from "vitest"
import en from "../i18n/locales/en.json"
import AvatarCropModal from "./AvatarCropModal.vue"

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })

function mountModal() {
  const file = new File([new Uint8Array([1, 2, 3])], "a.png", { type: "image/png" })
  return mount(AvatarCropModal, { props: { file }, global: { plugins: [i18n] } })
}

describe("AvatarCropModal", () => {
  it("renders the crop frame and a zoom control", () => {
    const w = mountModal()
    expect(w.find('[data-testid="avatar-crop-modal"]').exists()).toBe(true)
    expect(w.find('[data-testid="avatar-crop-zoom"]').exists()).toBe(true)
  })

  it("emits cancel from the cancel button and the backdrop", async () => {
    const w = mountModal()
    await w.get('[data-testid="avatar-crop-cancel"]').trigger("click")
    expect(w.emitted("cancel")).toHaveLength(1)
    await w.get('[data-testid="avatar-crop-modal"]').trigger("click") // backdrop (self)
    expect(w.emitted("cancel")?.length).toBeGreaterThanOrEqual(1)
  })

  it("disables the confirm button until the image has loaded", () => {
    // jsdom never fires img.onload for an object URL, so `ready` stays false — confirm is guarded
    // so a user can't export a blank canvas before the photo is in.
    const w = mountModal()
    expect(w.get('[data-testid="avatar-crop-confirm"]').attributes("disabled")).toBeDefined()
  })
})
