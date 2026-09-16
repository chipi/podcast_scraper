<script setup lang="ts">
/**
 * Entity card — MODAL presentation of {@link EntityCardBody} (UXS-014: a modal is only opened from a
 * page-level surface, e.g. the Search results). Inside a panel we instead render EntityCardBody
 * INLINE (replace-in-panel), never stacking a second backdrop. Teleported to <body> so it covers the
 * viewport (escapes any clipped/transformed ancestor); modal a11y = role/aria-modal + focus trap +
 * restore focus + ESC/backdrop dismiss.
 *
 * ## Back closes the card, it does not navigate the page underneath (#1594)
 *
 * The card had no URL at all, so it was invisible to history. On Android, Capacitor maps the
 * hardware Back button to a history navigation — with an open modal that meant the page BEHIND the
 * card navigated away while the card sat over the result. The issue flagged this as needing
 * verification in the shell; it is a certainty rather than a risk, because nothing was listening.
 *
 * The fix gives the open card a history entry, via a `?card=kind:id` query. Back is then ordinary
 * navigation: it pops the entry, the query disappears, the card closes. As a side effect the open
 * card becomes linkable, which is also the answer to this issue's "same entity, two navigation
 * models" complaint — the modal now has a URL like the `/topic/:id` page does.
 *
 * ## Why the ROUTER and not `history.pushState`
 *
 * Raw `pushState` is shorter and desynchronises vue-router: the router keeps its own object in
 * `history.state` for scroll restoration and back-detection, and overwriting it leaves the router
 * describing a position the browser is not in. Going through the router keeps one owner of the
 * history stack.
 *
 * ## The three ways this closes, and why the bookkeeping differs
 *
 * 1. **Back** — the query is already gone; the watcher closes the card and there is nothing to undo.
 * 2. **Escape / backdrop / the card's own close** — the entry we pushed is still on the stack. It
 *    has to be popped, or the user's next Back press only undoes our bookkeeping and looks like a
 *    button that did nothing.
 * 3. **Navigating away from inside the card** (tapping through to a topic page) — the query is gone
 *    because the route changed, and popping would undo the navigation the user just asked for.
 *
 * `closedByNavigation` is what separates 1 and 3 from 2.
 */
import { onUnmounted, ref } from "vue"
import EntityCardBody from "./EntityCardBody.vue"
import { useModalSheet } from "../composables/useModalSheet"
import { registerStackedSheet } from "../composables/sheetStack"

const props = withDefaults(
  defineProps<{
    kind: "person" | "topic" | "organization"
    id: string
    /**
     * Which `?<key>=` history entry this sheet owns, and how far it is offset from the top.
     *
     * Every sheet records its own entry so hardware Back closes it rather than navigating the page
     * underneath. Two sheets sharing a key fight over one entry: the inner overwrites the outer's
     * value on open, and closing then leaves the outer pointing at the wrong entity. `StorylineCard`
     * already layers over an entity sheet and uses `storyline` for precisely this reason; a second
     * ENTITY sheet needs its own key too (2026-09-16).
     */
    historyKey?: string
    /**
     * How many sheets this one is stacked ON TOP of. 0 = the bottom card.
     *
     * A boolean `stacked` could only ever express one level, and stacks go arbitrarily deep
     * (topic → storyline → person → …). The depth drives `--lp-depth`, which sizes the card one
     * peek shorter per level so each card below keeps its title visible — a deck, not a pile
     * (operator 2026-09-16).
     */
    depth?: number
  }>(),
  { historyKey: "card", depth: 0 }
)
const emit = defineEmits<{ (e: "close"): void }>()

const dialogEl = ref<HTMLElement | null>(null)

/**
 * The `?card=` history marker. Entity ids are ALREADY kind-namespaced (`person:jane-doe`,
 * `topic:ai`), so composing `${kind}:${id}` would produce `person:person:jane-doe`; a caller that
 * passes a bare id still gets a qualified key, because two kinds could otherwise collide on the
 * same bare id. useModalSheet owns the focus trap + the three close paths (see its docblock).
 */
function cardKey(): string {
  return props.id.includes(":") ? props.id : `${props.kind}:${props.id}`
}
useModalSheet(dialogEl, () => emit("close"), { key: props.historyKey, value: cardKey })

// A layered card pins the whole stack's geometry for as long as it is on screen.
if (props.depth > 0) {
  const release = registerStackedSheet()
  onUnmounted(release)
}
</script>

<template>
  <Teleport to="body">
    <div class="lp-sheet-scrim" role="dialog" aria-modal="true" @click.self="emit('close')">
      <div
        ref="dialogEl"
        tabindex="-1"
        class="lp-sheet w-full max-w-lg overflow-hidden rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
        :class="depth > 0 ? 'lp-sheet--stacked' : undefined"
        :style="{ '--lp-depth': depth }"
      >
        <EntityCardBody variant="overlay" :kind="kind" :id="id" :depth="depth" @close="emit('close')" />
      </div>
    </div>
  </Teleport>
</template>
