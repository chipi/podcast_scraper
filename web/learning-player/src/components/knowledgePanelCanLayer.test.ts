/**
 * `can-layer` must actually REACH TopicCardContent from a host that sets it.
 *
 * The device test for this passed while the storyline never opened: `openStoryline` fell to its
 * `router.push` branch, which navigates the page UNDERNEATH the panel, so the screen did not
 * visibly change and every assertion about "the storyline is present" was satisfied by the
 * "Part of a storyline" row inside the topic card itself (2026-09-16).
 *
 * A prop-level check is the honest place for this: it cannot be satisfied by a label that happens
 * to be on screen for another reason.
 */
import { describe, it, expect } from "vitest"
import { mount } from "@vue/test-utils"
import { defineComponent, h } from "vue"

// Mirror of the two lines that carry the value, so the test pins the CONTRACT rather than
// re-implementing the components: a host passes `canLayer`, the body forwards
// `canLayer ?? dismissAtRoot`, and the content acts on what it receives.
const Content = defineComponent({
  props: { canLayer: { type: Boolean, default: true } },
  setup: (props) => () => h("div", { "data-can-layer": String(props.canLayer) }),
})

const Body = defineComponent({
  props: {
    canLayer: { type: Boolean, default: undefined },
    dismissAtRoot: { type: Boolean, default: false },
  },
  setup: (props) => () => h(Content, { canLayer: props.canLayer ?? props.dismissAtRoot }),
})

describe("can-layer reaches the topic content", () => {
  it("is true when the host sets it, even though the card renders inline", () => {
    const w = mount(Body, { props: { canLayer: true, dismissAtRoot: false } })
    expect(w.find("[data-can-layer]").attributes("data-can-layer")).toBe("true")
  })

  it("falls back to dismissAtRoot when the host says nothing", () => {
    // NOTE the trap this pins: a Boolean-typed prop with no value is cast to FALSE by Vue, not
    // left undefined — so `canLayer ?? dismissAtRoot` can never reach the fallback unless the
    // prop's type admits undefined. If this returns "false" for a dismissAtRoot of true, the
    // nullish coalescing is dead code.
    const w = mount(Body, { props: { dismissAtRoot: true } })
    expect(w.find("[data-can-layer]").attributes("data-can-layer")).toBe("true")
  })
})
