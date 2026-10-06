<script setup lang="ts">
/**
 * Share menu (#2036) — **Share card**, **Copy link** and **Copy text** (operator 2026-10-05).
 *
 * **Share card** is the SERVER's card (`server/og/card.py`, via composables/shareCard) — the same
 * image a shared link unfurls as, so the card you send and the preview a link shows are one design.
 * The menu therefore needs only WHAT is shared (`kind` + `id`) and its name; it no longer builds a
 * card model of its own.
 *
 * Copy link / Copy text put the public https link (utils/shareLink), or a line of text ending in
 * it, on the clipboard and confirm it. They were "Share link" and "Share text", and beta testers
 * could not tell them apart from the card: on a phone all three opened the same system sheet.
 */
import { computed, ref } from "vue"
import { useI18n } from "vue-i18n"
import { track } from "../services/analytics"

import type { ShareCardKind } from "../services/api"
import { shareCard } from "../composables/shareCard"
import { copyText } from "../utils/clipboard"
import { shareUrl, type ShareTarget } from "../utils/shareLink"
import { useAnchoredMenu } from "../composables/useAnchoredMenu"

/**
 * `targetKind` is the ANALYTICS enum and stays required (#2267) — `kind` is what the card and the
 * link are about. They differ where analytics has no bucket of its own (a show reports `episode`).
 */
const props = defineProps<{
  kind: ShareCardKind
  id: string
  /** The thing's name — the file name, and the head of Copy text. */
  title: string
  /** What it is, in a few words, for Copy text: the show an episode is from, a person's one-liner. */
  context?: string | null
  targetKind: 'episode' | 'moment' | 'topic' | 'person' | 'storyline' | 'organization'
}>()
const { t } = useI18n()

/** The page a link opens. An organization has none of its own (overlay-only), so no link. */
const LINK_TARGET: Partial<Record<ShareCardKind, ShareTarget>> = {
  episode: 'episode',
  show: 'podcast',
  topic: 'topic',
  person: 'person',
  storyline: 'storyline',
  theme: 'theme',
}
const url = computed(() => {
  const target = LINK_TARGET[props.kind]
  return target ? shareUrl(target, props.id) : null
})

const note = ref("") // transient confirmation ("Link copied")
const triggerEl = ref<HTMLElement | null>(null)
const panelEl = ref<HTMLElement | null>(null)
// Shared popover shell — teleported, viewport-clamped placement, outside-pointer/Escape dismissal.
// This is why the menu no longer runs off the left edge when the trigger sits near it (a storyline
// share opened from Home): `anchorPanel` clamps it on screen (operator 2026-09-13).
const { open, toggle, close, teleportTarget } = useAnchoredMenu(triggerEl, panelEl, { align: "end" })

const making = ref(false)
async function onCard(): Promise<void> {
  close()
  if (making.value) return
  // An image card always goes through the platform sheet — there is nothing to copy.
  track('share', { target_kind: props.targetKind, method: 'native_sheet' })
  making.value = true
  try {
    await shareCard(props.kind, props.id, props.title)
  } catch {
    // Offline, or the server could not draw it — say so rather than doing nothing.
    flash(t("share.cardFailed"))
  } finally {
    making.value = false
  }
}
async function onLink(): Promise<void> {
  close()
  if (!url.value || !(await copyText(url.value))) return
  track('share', { target_kind: props.targetKind, method: 'copy_link' })
  flash(t("share.linkCopied"))
}
/** "Name — what it is", then the link, so a pasted message leads back. */
function copyTextBody(): string {
  const head = props.context ? `${props.title} — ${props.context}` : props.title
  return [head, url.value].filter(Boolean).join("\n")
}
async function onText(): Promise<void> {
  close()
  if (!(await copyText(copyTextBody()))) return
  track('share', { target_kind: props.targetKind, method: 'copy_text' })
  flash(t("share.textCopied"))
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
      <!-- A non-hidden accessible name INSIDE the trigger (2026-09-24, Android device tier).
           `aria-haspopup` PLUS a fully hidden subtree leaves the button UNNAMED on Android System
           WebView 150 — the label string appears nowhere in the accessibility tree, so TalkBack
           announces only "Button". Neither condition alone does it: `Play` and `Skip back 15
           seconds` are icon-only with `aria-label` and named, because they open no popup. See
           OverflowMenu.vue for the measurement. -->
      <span class="sr-only">{{ t('share.open') }}</span>
    </button>
    <!-- Teleported + viewport-clamped via the shared shell (was `absolute right-0`, which ran off the
         left edge when the trigger sat near it). -->
    <Teleport :to="teleportTarget">
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
          :disabled="making"
          @click="onCard"
        >
          {{ t("share.card") }}
        </button>
        <button
          v-if="url"
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
