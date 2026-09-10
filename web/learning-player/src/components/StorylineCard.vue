<script setup lang="ts">
/**
 * Storyline card — MODAL presentation of {@link StorylineView} (the theme-cluster page), opened
 * "on top" from a topic card's storyline link instead of navigating away. Mirrors
 * {@link EntityCard} exactly: teleported to <body> so it escapes any clipped/transformed ancestor,
 * modal a11y (role/aria-modal + focus trap + restore focus + ESC/backdrop dismiss), and a
 * `?storyline=<anchorTopicId>` history entry so hardware/browser Back closes the card rather than
 * navigating the page underneath (see EntityCard's header for the full rationale of the three close
 * paths and why we go through the router, not raw pushState).
 *
 * The route param IS the anchor topic id; a storyline has no dedicated endpoint, so StorylineView
 * reconstructs the whole theme cluster from any member topic's card. The topic card passes its own
 * id, which is a member, so the same storyline resolves.
 */
import { nextTick, onMounted, onUnmounted, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { useRoute, useRouter } from "vue-router"
import StorylineView from "../views/StorylineView.vue"

const props = defineProps<{ id: string }>()
const emit = defineEmits<{ (e: "close"): void }>()

const { t } = useI18n()
const dialogEl = ref<HTMLElement | null>(null)
let restoreFocus: HTMLElement | null = null

function focusables(): HTMLElement[] {
  if (!dialogEl.value) return []
  const sel = 'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])'
  return Array.from(dialogEl.value.querySelectorAll<HTMLElement>(sel))
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === "Escape") {
    emit("close")
    return
  }
  if (e.key !== "Tab") return
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

watch(
  () => route.query.storyline,
  (v) => {
    if (!v) {
      closedByNavigation = true
      emit("close")
    }
  }
)

onMounted(() => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener("keydown", onKeydown)
  void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
  void router.push({ query: { ...route.query, storyline: props.id } })
})

onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown)
  restoreFocus?.focus?.()
  if (!closedByNavigation && route.query.storyline) void router.back()
})
</script>

<template>
  <Teleport to="body">
    <div class="lp-sheet-scrim" role="dialog" aria-modal="true" @click.self="emit('close')">
      <div
        ref="dialogEl"
        tabindex="-1"
        class="lp-sheet relative w-full max-w-lg overflow-hidden rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
        data-testid="storyline-card"
      >
        <!-- The sheet owns the ✕ (StorylineView drops its own back button when embedded). -->
        <button
          type="button"
          class="lp-nav absolute right-3 top-3 z-10 shrink-0"
          :aria-label="t('ec.close')"
          data-testid="storyline-card-close"
          @click="emit('close')"
        >
          <span aria-hidden="true" class="text-base leading-none">✕</span>
        </button>
        <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <StorylineView :id="id" embedded />
        </div>
      </div>
    </div>
  </Teleport>
</template>
