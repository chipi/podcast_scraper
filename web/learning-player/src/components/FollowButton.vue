<script setup lang="ts">
/**
 * FollowButton (F2.4) — the ONE follow pill/glyph, so Follow looks and behaves identically
 * wherever anything can be followed. Presentational — the host owns follow state and the
 * gated toggle; this renders and emits `toggle`. Mirrors the FavoriteButton /
 * AddToCollectionButton "one affordance, a variant for context" pattern (UXS-014).
 * Save ≠ Follow: this is the pill, the heart is FavoriteButton.
 *
 * Variants:
 *   `inline`      — header pill for show pages; always `data-testid="follow-show"`.
 *   `overlay`     — smaller, plated pill that reads over artwork (`ShowTile`). HOST positions it
 *                   (the button used to hard-code `absolute right-1.5 top-1.5`, which prevented
 *                   stacking — operator 2026-09-17). Still `data-testid="follow-show"`.
 *   `icon`        — 32px circle carrying only ✓/+, same shape as FavoriteButton. Exists for
 *                   action rows under a 128px artwork column where a labelled pill overflows.
 *                   Still `data-testid="follow-show"`.
 *   `ec`          — inline pill for topic/person/org cards (`EntityCardBody`). Same class-shape as
 *                   `inline` but testid `ec-follow`, `ec.*` i18n keys, and a sr-only that appends
 *                   the entity's label so "Follow" is never announced four-identical-times in a row.
 *   `storyline`   — inline pill for StorylineView. Testid `storyline-follow`,
 *                   `ec.followStoryline`/`ec.followingStoryline` keys, sr-only includes `label`.
 *   `discovery`   — glyph button for DiscoveryList rows. Testid `discovery-follow`, `ec.*` keys,
 *                   sr-only includes `label`.
 *   `trend-spark` — glyph button for TrendingSparkChips rows. Testid `trend-spark-follow`, same
 *                   shape as `discovery`, sr-only includes `label`.
 *
 * Accessibility (2026-09-25, Android device tier): EVERY variant uses `aria-hidden` on the glyph
 * and a real text node (either the inline label text or a `sr-only` span) for the accessible name.
 * `aria-label` alone is NOT sufficient — on Android System WebView 150 the glyph overrides it and
 * the `aria-label` never applies. Only real text inside the control produces a usable name. See
 * `OverflowMenu` and `FavoriteButton` for the same fix applied to the other icon-only controls.
 *
 * The four non-show testids (`ec-follow`, `storyline-follow`, `discovery-follow`,
 * `trend-spark-follow`) are STATIC strings in this file so the surface-map guard
 * (`src/__checks__/surface-map.test.ts`) can extract them. A `:data-testid` bound to a prop
 * would be invisible to that guard — that is why each variant has a separate `<button>` rather than
 * a single parameterised element.
 */
import { useI18n } from "vue-i18n"

const props = withDefaults(
  defineProps<{
    /** Whether the thing is currently followed. */
    following: boolean
    /** A toggle is in flight — disables the control. */
    busy?: boolean
    /**
     * Signed out: the tap routes to sign-in (label says so, no pressed state asserted).
     * Applies to show follows only — entity/storyline/discovery/trend-spark callers gate
     * rendering with `v-if="auth.isAuthenticated"`, so `gated` is unused there.
     */
    gated?: boolean
    /**
     * The name of the thing being followed (a topic label, a person's name, a storyline title).
     * Included in the `sr-only` accessible name for `ec`, `storyline`, `discovery`, and
     * `trend-spark` variants — so each control says "Follow — systems thinking" rather than
     * four identical "Follow"s on the same page (the original Android audit finding).
     * Optional for `inline`/`overlay`/`icon` — those variants show the label as inline text.
     */
    label?: string
    /**
     * See the component-level doc for the full list. Defaults to `inline`.
     */
    variant?:
      | "inline"
      | "overlay"
      | "icon"
      | "ec"
      | "storyline"
      | "theme"
      | "discovery"
      | "trend-spark"
  }>(),
  { busy: false, gated: false, variant: "inline" }
)
defineEmits<{ (e: "toggle"): void }>()
const { t } = useI18n()

/** The action label used in sr-only text, including the thing's name when available. */
function srLabel(followKey: string, followingKey: string): string {
  const action = props.following ? t(followingKey) : t(followKey)
  return props.label ? `${action} — ${props.label}` : action
}
</script>

<template>
  <!-- ── show-page variants (`inline` / `overlay` / `icon`) ─────────────────── -->
  <!-- Static testid `follow-show`: e2e-critical, extracted from source by the surface-map guard. -->
  <button
    v-if="variant === 'inline' || variant === 'overlay' || variant === 'icon'"
    type="button"
    data-testid="follow-show"
    class="inline-flex shrink-0 items-center gap-1 rounded-full font-bold transition disabled:opacity-50"
    :class="[
      variant === 'overlay'
        ? 'h-7 px-2 text-[0.65rem] shadow-lg backdrop-blur'
        : variant === 'icon'
        ? 'lp-tap h-8 w-8 justify-center border border-border text-base'
        : 'px-3 py-1 text-xs',
      following
        ? 'bg-accent text-accent-foreground'
        : variant === 'overlay'
        ? 'bg-canvas/80 text-canvas-foreground hover:bg-canvas'
        : variant === 'icon'
        ? 'text-canvas-foreground hover:bg-overlay'
        : 'bg-overlay text-canvas-foreground hover:bg-elevated',
    ]"
    :aria-pressed="gated ? undefined : following"
    :aria-label="
      gated ? t('auth.signInToFollow') : following ? t('podcast.following') : t('podcast.follow')
    "
    :title="following ? t('podcast.following') : t('podcast.follow')"
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    <!-- The label is the whole control in the labelled variants. -->
    <template v-if="variant !== 'icon'">{{
      following ? t("podcast.following") : t("podcast.follow")
    }}</template>
    <!-- ...and in `icon` it is an `sr-only` NAME, not "the glyph plus the accessible name"
         (2026-09-25, Android device tier). That reasoning was wrong on Chromium: the glyph above is
         `aria-hidden` and Chromium STILL used it as the button's name, so the control announced as
         "+" and the `aria-label` never applied. Correctly hiding the glyph is not enough — only
         real text inside the control produces a usable name. Fourteen instances of this one
         component were in the first audit run; see `AccessibleNameAuditTests`. -->
    <span v-else class="sr-only">{{
      gated ? t("auth.signInToFollow") : following ? t("podcast.following") : t("podcast.follow")
    }}</span>
  </button>

  <!-- ── topic / person / org card variant (`ec`) ───────────────────────────── -->
  <!-- Static testid `ec-follow`: e2e-critical (surface map row 376; `entity-and-rails-invariants.spec.ts`). -->
  <button
    v-else-if="variant === 'ec'"
    type="button"
    data-testid="ec-follow"
    class="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition disabled:opacity-50"
    :class="
      following
        ? 'bg-accent text-accent-foreground'
        : 'bg-overlay text-canvas-foreground hover:bg-elevated'
    "
    :aria-pressed="following"
    :title="t('ec.followHint')"
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    <!-- Visible label PLUS sr-only that includes the entity name — "Follow — systems thinking"
         rather than "Follow" four times in a row (Android audit, 2026-09-25). The visible text
         stays so sighted users also see which entity the button belongs to in context; the sr-only
         adds the label for users reading by control in isolation. -->
    {{ following ? t("ec.following") : t("ec.follow") }}
    <span v-if="label" class="sr-only"> — {{ label }}</span>
  </button>

  <!-- ── storyline variant ───────────────────────────────────────────────────── -->
  <!-- Static testid `storyline-follow`: e2e-critical (`storyline.spec.ts`,
       `entity-and-rails-invariants.spec.ts`). -->
  <button
    v-else-if="variant === 'storyline'"
    type="button"
    data-testid="storyline-follow"
    class="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition disabled:opacity-50"
    :class="
      following
        ? 'bg-accent text-accent-foreground'
        : 'bg-overlay text-canvas-foreground hover:bg-elevated'
    "
    :aria-pressed="following"
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    {{ following ? t("ec.followingStoryline") : t("ec.followStoryline") }}
    <span v-if="label" class="sr-only"> — {{ label }}</span>
  </button>

  <!-- ── theme variant ──────────────────────────────────────────────────────── -->
  <!-- The storyline pill with the theme's words. Reusing `storyline` outright put "Follow
       storyline" on the theme page, which says the wrong thing about the one distinction the page
       exists to make. Its own testid so an e2e can tell the two apart. -->
  <button
    v-else-if="variant === 'theme'"
    type="button"
    data-testid="theme-follow"
    class="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition disabled:opacity-50"
    :class="
      following
        ? 'bg-accent text-accent-foreground'
        : 'bg-overlay text-canvas-foreground hover:bg-elevated'
    "
    :aria-pressed="following"
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    {{ following ? t("ec.followingTheme") : t("ec.followTheme") }}
    <span v-if="label" class="sr-only"> — {{ label }}</span>
  </button>

  <!-- ── discovery list glyph variant (`discovery`) ─────────────────────────── -->
  <!-- Static testid `discovery-follow`: e2e-critical (`home-rails.spec.ts`, `trending.spec.ts`). -->
  <button
    v-else-if="variant === 'discovery'"
    type="button"
    data-testid="discovery-follow"
    class="shrink-0 rounded-full px-2 py-1 text-base leading-none transition disabled:opacity-50"
    :class="following ? 'text-accent' : 'text-muted hover:text-accent'"
    :aria-pressed="following"
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <!-- Glyph DECORATIVE, name in `sr-only` (2026-09-25, Android device tier). The row's label is
         included because "Follow" four times in a row tells a screen-reader user nothing about which
         topic they are following (same fix as the inline `ec` variant above). -->
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    <span class="sr-only">{{ srLabel("ec.follow", "ec.following") }}</span>
  </button>

  <!-- ── trending spark chips glyph variant (`trend-spark`) ────────────────── -->
  <!-- Static testid `trend-spark-follow`: documented in E2E_SURFACE_MAP.md. -->
  <button
    v-else-if="variant === 'trend-spark'"
    type="button"
    data-testid="trend-spark-follow"
    class="shrink-0 rounded-full px-2 py-1 text-base leading-none transition disabled:opacity-50"
    :class="following ? 'text-accent' : 'text-muted hover:text-accent'"
    :aria-pressed="following"
    :disabled="busy"
    @click.prevent.stop="$emit('toggle')"
  >
    <!-- Same pattern as `discovery`: glyph is decorative, sr-only carries the full name. -->
    <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
    <span class="sr-only">{{ srLabel("ec.follow", "ec.following") }}</span>
  </button>
</template>
