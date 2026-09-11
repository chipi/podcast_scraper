<script setup lang="ts">
/**
 * Share menu (#2036) — one affordance, three modes: **Share card** (the editorial PNG),
 * **Share link** (the canonical URL), **Share text** (the caption). Given an {@link EntityCardModel}
 * it drives `entityShareCard`; the card is the star, the link unfurls AS the card once OG-images
 * land, and text is the graceful fallback.
 */
import { ref, watch } from "vue"
import { useI18n } from "vue-i18n"

import {
  type EntityCardModel,
  entityCardText,
  shareEntityCard,
  shareEntityLink,
} from "../composables/entityShareCard"
import { isNative, saveAndShareText } from "../services/native"

const props = defineProps<{ model: EntityCardModel }>()
const { t } = useI18n()

const open = ref(false)
const note = ref("") // transient confirmation ("Link copied")
const rootEl = ref<HTMLElement | null>(null)

function close(): void {
  open.value = false
}
function toggle(): void {
  open.value = !open.value
}

// Escape + outside-click close while open; torn down via Vue's onCleanup when it closes/unmounts.
watch(open, (isOpen, _prev, onCleanup) => {
  if (!isOpen || typeof document === "undefined") return
  const onKey = (e: KeyboardEvent): void => {
    if (e.key === "Escape") close()
  }
  const onDown = (e: MouseEvent): void => {
    if (rootEl.value && !rootEl.value.contains(e.target as Node)) close()
  }
  document.addEventListener("keydown", onKey)
  document.addEventListener("mousedown", onDown)
  onCleanup(() => {
    document.removeEventListener("keydown", onKey)
    document.removeEventListener("mousedown", onDown)
  })
})

async function onCard(): Promise<void> {
  close()
  await shareEntityCard(props.model)
}
async function onLink(): Promise<void> {
  const r = await shareEntityLink(props.model)
  close()
  if (r === "copied") flash(t("share.linkCopied"))
}
async function onText(): Promise<void> {
  close()
  const text = entityCardText(props.model)
  if (typeof navigator !== "undefined" && "share" in navigator) {
    try {
      await navigator.share({ text })
      return
    } catch {
      /* fall through */
    }
  }
  if (isNative()) {
    await saveAndShareText("closelistening.txt", text, "text/plain")
    return
  }
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    flash(t("share.textCopied"))
  }
}

let flashTimer: ReturnType<typeof setTimeout> | undefined
function flash(msg: string): void {
  note.value = msg
  if (flashTimer) clearTimeout(flashTimer)
  flashTimer = setTimeout(() => (note.value = ""), 2000)
}
</script>

<template>
  <div ref="rootEl" class="relative inline-block">
    <button
      type="button"
      class="lp-nav shrink-0"
      :aria-label="t('share.open')"
      :aria-expanded="open"
      aria-haspopup="menu"
      data-testid="share-menu"
      @click="toggle"
    >
      <span aria-hidden="true" class="text-base leading-none">↗</span>
    </button>
    <div
      v-if="open"
      role="menu"
      class="absolute right-0 z-20 mt-2 w-48 overflow-hidden rounded border border-border bg-surface py-1 shadow-lg"
      data-testid="share-menu-list"
    >
      <button
        type="button"
        role="menuitem"
        class="block w-full px-3 py-2 text-left text-sm text-canvas-foreground hover:bg-overlay"
        data-testid="share-card"
        @click="onCard"
      >
        {{ t("share.card") }}
      </button>
      <button
        v-if="model.url"
        type="button"
        role="menuitem"
        class="block w-full px-3 py-2 text-left text-sm text-canvas-foreground hover:bg-overlay"
        data-testid="share-link"
        @click="onLink"
      >
        {{ t("share.link") }}
      </button>
      <button
        type="button"
        role="menuitem"
        class="block w-full px-3 py-2 text-left text-sm text-canvas-foreground hover:bg-overlay"
        data-testid="share-text"
        @click="onText"
      >
        {{ t("share.text") }}
      </button>
    </div>
    <span v-if="note" class="sr-only" role="status">{{ note }}</span>
  </div>
</template>
