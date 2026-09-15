<script setup lang="ts">
/**
 * Share menu (#2036) — one affordance, three modes: **Share card** (the editorial PNG),
 * **Share link** (the canonical URL), **Share text** (the caption). Given an {@link EntityCardModel}
 * it drives `entityShareCard`; the card is the star, the link unfurls AS the card once OG-images
 * land, and text is the graceful fallback.
 */
import { ref } from "vue"
import { useI18n } from "vue-i18n"

import {
  type EntityCardModel,
  entityCardText,
  shareEntityCard,
  shareEntityLink,
} from "../composables/entityShareCard"
import { isNative, saveAndShareText } from "../services/native"
import { useAnchoredMenu } from "../composables/useAnchoredMenu"

const props = defineProps<{ model: EntityCardModel }>()
const { t } = useI18n()

const note = ref("") // transient confirmation ("Link copied")
const triggerEl = ref<HTMLElement | null>(null)
const panelEl = ref<HTMLElement | null>(null)
// Shared popover shell — teleported, viewport-clamped placement, outside-pointer/Escape dismissal.
// This is why the menu no longer runs off the left edge when the trigger sits near it (a storyline
// share opened from Home): `anchorPanel` clamps it on screen (operator 2026-09-13).
const { open, toggle, close } = useAnchoredMenu(triggerEl, panelEl, { align: "end" })

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
    try {
      await navigator.clipboard.writeText(text)
      flash(t("share.textCopied"))
    } catch {
      /* clipboard permission denied — nothing copied, no crash */
    }
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
  <div class="relative inline-block">
    <!-- Same ghost circle as Favourite / Download / ⋯ (operator 2026-09-13): a bare glyph between
         two circled controls read as "soft" and out of place. One geometry for the whole row. -->
    <button
      ref="triggerEl"
      type="button"
      class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground"
      :class="{ 'text-canvas-foreground': open }"
      :aria-label="t('share.open')"
      :aria-expanded="open"
      aria-haspopup="menu"
      data-testid="share-menu"
      @click="toggle"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="h-4 w-4"
        aria-hidden="true"
      >
        <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
        <path d="m8.6 10.5 6.8-4M8.6 13.5l6.8 4" />
      </svg>
    </button>
    <!-- Teleported + viewport-clamped via the shared shell (was `absolute right-0`, which ran off the
         left edge when the trigger sat near it). -->
    <Teleport to="body">
      <div
        v-if="open"
        ref="panelEl"
        role="menu"
        class="invisible fixed left-0 top-0 z-50 w-48 max-w-[calc(100vw-1rem)] overflow-hidden rounded border border-border bg-surface py-1 shadow-lg"
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
    </Teleport>
    <!-- Visible transient confirmation (was sr-only → sighted users got no feedback on copy). Stays
         anchored to the trigger (not teleported): it appears AFTER the menu closes. -->
    <span
      v-if="note"
      role="status"
      class="absolute right-0 top-full mt-1 whitespace-nowrap rounded bg-overlay px-2 py-1 text-xs text-muted"
      >{{ note }}</span
    >
  </div>
</template>
