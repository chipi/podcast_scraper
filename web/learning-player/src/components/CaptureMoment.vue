<script setup lang="ts">
/**
 * "Mark this moment" — the cheapest entry to the learning loop, and its receipt (#1592).
 *
 * ## Why this is a component and not two buttons
 *
 * The affordance has to exist in two places at once: on mobile it belongs in the STICKY transport,
 * because the masthead scrolls away exactly when you want it (you mark a moment mid-listen, thumb
 * already on the controls); on desktop it stays in the masthead beside Favourite and Download,
 * where the transport is not sticky and there is room. Writing it twice would mean the state
 * machine below existed twice, and the two copies would drift the first time one of them changed.
 *
 * ## The three states, and why "failed" is one of them
 *
 * Capture used to have exactly one visible outcome: the icon filled with the accent for ~1.5s.
 * On failure it did nothing at all — `markMoment` announced `capture.saveFailed` into an `sr-only`
 * live region and returned before setting the flash, so a sighted user could not tell a failed
 * save from a tap that missed. The reasoning at the time was right (a false confirmation is worse
 * than silence, S8) and the fix was half-applied: the truth reached screen readers only.
 *
 * So: `idle` → tap → `saved` (a real receipt, see below) or `failed` (visible, in danger colour).
 *
 * ## Why the receipt is a link and not a toast
 *
 * A capture that vanishes into a 1.5s flash has no destination — a new user has no idea the thing
 * they just saved lives under Library → Saved. A toast was the obvious answer and the wrong one
 * here: Zone D owns the bottom of the player on mobile, and a bottom-anchored toast would land on
 * top of the live insight panel.
 *
 * Instead the control itself becomes the receipt: it expands into a labelled link to the place the
 * capture went, holds for `RECEIPT_MS`, then collapses back to an icon. The confirmation appears
 * where the action happened, and it reuses the linger rhythm Zone D already established rather
 * than inventing a second one.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { RouteLocationRaw } from 'vue-router'

const props = withDefaults(
  defineProps<{
    /** Outcome of the last capture. Owned by the player so both placements agree. */
    state?: 'idle' | 'saved' | 'failed'
    /** Signed out — the tap defers to sign-in rather than POSTing a 401 (#1590). */
    gated?: boolean
    /** `icon` sits among the masthead's icon row; `pill` matches the transport's corner pills. */
    variant?: 'icon' | 'pill'
  }>(),
  { state: 'idle', gated: false, variant: 'icon' },
)

const emit = defineEmits<{ (e: 'capture'): void }>()

const { t } = useI18n()

/** Where a capture actually goes: Library → Saved renders the Highlights section (#1141). */
const HIGHLIGHTS: RouteLocationRaw = { name: 'library', query: { tab: 'saved' } }

const label = computed(() => {
  if (props.gated) return t('auth.signInToCapture')
  if (props.state === 'saved') return t('capture.savedGoToHighlights')
  if (props.state === 'failed') return t('capture.saveFailed')
  return t('capture.markMoment')
})

/** Expanded (labelled) whenever there is an outcome to report; an icon the rest of the time. */
const expanded = computed(() => props.state !== 'idle')

const tone = computed(() => {
  if (props.state === 'saved') return 'bg-accent text-accent-foreground'
  if (props.state === 'failed') return 'bg-danger/15 text-danger'
  return props.variant === 'pill' ? 'bg-overlay text-muted' : 'text-muted hover:text-accent'
})

const shape = computed(() =>
  props.variant === 'pill' || expanded.value
    ? 'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-bold'
    : 'rounded-full p-1 text-xl',
)
</script>

<template>
  <!--
    Saved is a LINK, not a button: the receipt's whole job is to be followable. The other two
    states are buttons, because tapping them captures (or retries).
  -->
  <RouterLink
    v-if="state === 'saved'"
    :to="HIGHLIGHTS"
    :class="[shape, tone, 'no-underline transition']"
    :title="label"
    data-testid="capture-receipt"
  >
    <svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" class="h-4 w-4 shrink-0" aria-hidden="true">
      <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
    </svg>
    <span>{{ label }}</span>
  </RouterLink>

  <button
    v-else
    type="button"
    :class="[shape, tone, 'transition']"
    :aria-label="label"
    :title="label"
    data-testid="capture-moment"
    @click="emit('capture')"
  >
    <svg
      viewBox="0 0 24 24"
      :fill="state === 'failed' ? 'none' : 'none'"
      stroke="currentColor"
      stroke-width="2"
      :class="variant === 'pill' || expanded ? 'h-4 w-4 shrink-0' : 'h-5 w-5'"
      aria-hidden="true"
    >
      <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
      <!-- Failure gets its own mark, so the state is legible without relying on colour alone. -->
      <path v-if="state === 'failed'" d="M12 8v4" />
      <path v-if="state === 'failed'" d="M12 15h.01" />
    </svg>
    <span v-if="expanded">{{ label }}</span>
  </button>
</template>
