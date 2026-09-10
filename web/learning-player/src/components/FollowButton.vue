<script setup lang="ts">
/**
 * FollowButton (F2.4) — the ONE follow pill, so Follow looks and behaves identically wherever
 * something can be followed: shows (show-page header `inline`, `ShowTile` artwork `overlay`), and —
 * via the label/testid props — topics, people and storylines on the entity card / storyline sheet.
 * Presentational: the host owns the follow state, the store, and the gated toggle; this renders the
 * pill and emits `toggle`. Mirrors FavoriteButton / AddToCollectionButton (one affordance, a variant
 * per context). Save ≠ Follow: this is the pill, the heart is FavoriteButton.
 *
 * The labels default to the show copy; any other kind passes its own resolved strings (e.g.
 * `t('ec.follow')`), so the copy is host-chosen but the pill is one component.
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"

const props = withDefaults(
  defineProps<{
    /** Whether the entity is currently followed. */
    following: boolean
    /** A toggle is in flight — disables the control. */
    busy?: boolean
    /** Signed out: the tap routes to sign-in (label says so, no pressed state asserted). */
    gated?: boolean
    /** `inline` = header pill; `overlay` = smaller pill floated over artwork. */
    variant?: "inline" | "overlay"
    /** Resolved copy — default to the show pill; other kinds pass their own. */
    labelFollow?: string
    labelFollowing?: string
    labelGated?: string
    /** Optional native tooltip (the entity card used one to explain what follow does). */
    hint?: string
    /** e2e selector — defaults to the show pill's `follow-show`. */
    testid?: string
  }>(),
  { busy: false, gated: false, variant: "inline", testid: "follow-show" }
)
defineEmits<{ (e: "toggle"): void }>()
const { t } = useI18n()
const followLabel = computed(() => props.labelFollow ?? t("podcast.follow"))
const followingLabel = computed(() => props.labelFollowing ?? t("podcast.following"))
const gatedLabel = computed(() => props.labelGated ?? t("auth.signInToFollow"))
</script>

<template>
  <button
    type="button"
    :data-testid="testid"
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
    :aria-label="gated ? gatedLabel : following ? followingLabel : followLabel"
    :title="hint || undefined"
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    {{ following ? followingLabel : followLabel }}
  </button>
</template>
