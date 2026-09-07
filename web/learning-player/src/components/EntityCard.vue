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
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import EntityCardBody from './EntityCardBody.vue'

const props = defineProps<{ kind: 'person' | 'topic'; id: string }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const dialogEl = ref<HTMLElement | null>(null)
let restoreFocus: HTMLElement | null = null

function focusables(): HTMLElement[] {
  if (!dialogEl.value) return []
  const sel = 'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])'
  return Array.from(dialogEl.value.querySelectorAll<HTMLElement>(sel))
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape') {
    emit('close')
    return
  }
  if (e.key !== 'Tab') return
  const items = focusables()
  if (items.length === 0) return
  const first = items[0]
  const last = items[items.length - 1]
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault()
    last.focus()
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault()
    first.focus()
  }
}

const router = useRouter()
const route = useRoute()

/** True when the query went away on its own — a Back press, or a route change from inside. */
let closedByNavigation = false

/**
 * The history marker for this card.
 *
 * Entity ids are ALREADY kind-namespaced (`person:jane-doe`, `topic:ai`), so composing
 * `${kind}:${id}` produced `person:person:jane-doe`. Callers that pass a bare id still get a
 * qualified key, because two kinds could otherwise collide on the same bare id.
 */
function cardKey(): string {
  return props.id.includes(':') ? props.id : `${props.kind}:${props.id}`
}

watch(
  () => route.query.card,
  (card) => {
    if (!card) {
      closedByNavigation = true
      emit('close')
    }
  },
)

onMounted(() => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener('keydown', onKeydown)
  void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
  void router.push({ query: { ...route.query, card: cardKey() } })
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
  restoreFocus?.focus?.()
  // Only when WE closed it. See the three cases in the header comment.
  if (!closedByNavigation && route.query.card) void router.back()
})
</script>

<template>
  <Teleport to="body">
    <div
      class="fixed inset-0 z-50 flex items-end justify-center bg-black/40 sm:items-center"
      role="dialog"
      aria-modal="true"
      @click.self="emit('close')"
    >
      <div
        ref="dialogEl"
        tabindex="-1"
        class="flex max-h-[92dvh] w-full max-w-lg flex-col overflow-hidden rounded-t-2xl bg-surface outline-none sm:max-h-[85dvh] sm:rounded-2xl"
      >
        <EntityCardBody variant="overlay" :kind="kind" :id="id" @close="emit('close')" />
      </div>
    </div>
  </Teleport>
</template>
