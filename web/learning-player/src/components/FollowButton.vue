<script setup lang="ts">
/**
 * FollowButton (F2.4) — the ONE show-follow pill, so Follow looks and behaves identically wherever a
 * show can be followed: the show-page header (`inline`) and the artwork overlay on `ShowTile`
 * (`overlay`). Presentational — the host owns the follow state and the gated toggle; this renders
 * the pill and emits `toggle`. Mirrors the FavoriteButton / AddToCollectionButton "one affordance,
 * a variant for context" pattern (UXS-014). Save ≠ Follow: this is the pill, the heart is
 * FavoriteButton.
 *
 * Kept SHOW-only with a STATIC `data-testid="follow-show"`: the entity-card / storyline follow pills
 * are hand-rolled with their own static testids, because the surface-map guard (and the e2e
 * reconstruction it enforces) extracts testids from SOURCE — a `:data-testid` bound to a prop is
 * invisible to it, and `follow-show` is e2e-critical.
 */
import { useI18n } from "vue-i18n"

withDefaults(
  defineProps<{
    /** Whether the show is currently followed. */
    following: boolean
    /** A toggle is in flight — disables the control. */
    busy?: boolean
    /** Signed out: the tap routes to sign-in (label says so, no pressed state asserted). */
    gated?: boolean
    /** `inline` = header pill; `overlay` = smaller pill floated over artwork. */
    variant?: "inline" | "overlay"
  }>(),
  { busy: false, gated: false, variant: "inline" }
)
defineEmits<{ (e: "toggle"): void }>()
const { t } = useI18n()
</script>

<template>
  <button
    type="button"
    data-testid="follow-show"
    class="inline-flex shrink-0 items-center gap-1 rounded-full font-bold transition disabled:opacity-50"
    :class="[
      variant === 'overlay'
        ? 'absolute right-1.5 top-1.5 h-7 px-2 text-[0.65rem] shadow-lg backdrop-blur'
        : 'px-3 py-1 text-xs',
      following
        ? 'bg-accent text-accent-foreground'
        : variant === 'overlay'
        ? 'bg-canvas/80 text-canvas-foreground hover:bg-canvas'
        : 'bg-overlay text-canvas-foreground hover:bg-elevated',
    ]"
    :aria-pressed="gated ? undefined : following"
    :aria-label="
      gated ? t('auth.signInToFollow') : following ? t('podcast.following') : t('podcast.follow')
    "
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    {{ following ? t("podcast.following") : t("podcast.follow") }}
  </button>
</template>
