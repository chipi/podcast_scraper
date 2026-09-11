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
import { ref } from "vue"
import EntityCardBody from "./EntityCardBody.vue"
import { useModalSheet } from "../composables/useModalSheet"

const props = defineProps<{ kind: "person" | "topic" | "organization"; id: string }>()
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
useModalSheet(dialogEl, () => emit("close"), { key: "card", value: cardKey })
</script>

<template>
  <Teleport to="body">
    <div class="lp-sheet-scrim" role="dialog" aria-modal="true" @click.self="emit('close')">
      <div
        ref="dialogEl"
        tabindex="-1"
        class="lp-sheet w-full max-w-lg overflow-hidden rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
      >
        <EntityCardBody variant="overlay" :kind="kind" :id="id" @close="emit('close')" />
      </div>
    </div>
  </Teleport>
</template>
