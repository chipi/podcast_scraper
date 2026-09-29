<script setup lang="ts">
/**
 * The ONE save-a-fragment control — the bookmark, everywhere a HIGHLIGHT is made (UXS-014: define
 * once, use everywhere).
 *
 * ## Why this exists (operator 2026-09-27)
 *
 * Three controls wrote to the same place — the capture store, `/api/app/highlights`, surfacing in
 * Library → Saved → Highlights — and drew two different glyphs between them:
 *
 *   - `CaptureMoment` (the transport)      bookmark
 *   - the transcript paragraph save         bookmark
 *   - the Knowledge panel's insight save    HEART, via `FavoriteButton variant="controlled"`
 *
 * So the heart meant two different things depending on where it was — a FAVOURITE on an episode or
 * show header (the favorites store, whole objects) and a HIGHLIGHT on an insight (this store, a
 * fragment). `services/types.ts` already said as much in a comment: *"Saveable favorite kinds.
 * `insight` is NOT one — an insight is a capture"* (#1593 banned favouriting insights). The
 * component borrowed for it still said "Save to favorites" out loud to a screen reader.
 *
 * The rule the app now keeps, and the reason there is one component rather than three call sites:
 *
 *   - **heart** = favourite — a WHOLE object (episode, show, topic, person)
 *   - **bookmark** = highlight — a FRAGMENT (a transcript span, an insight, a timestamp)
 *   - **collection** = filing into a named board, and it gets its own glyph (it used to draw a
 *     bookmark too, which is what made the player's mark-a-moment read as add-to-collection)
 *
 * ## Accessibility, learned the hard way on the Android tier
 *
 * `BookmarkIcon` is a COMPONENT, not a text node, so an `aria-label` over it alone leaves the
 * button announced as a bare "Button" by Android System WebView — found by the class-level guard
 * in `__checks__/accessible-names.test.ts` rather than by any device run, because no suite reaches
 * these controls. Hence the `sr-only` span, whose text matches `aria-label` EXACTLY: Android reads
 * `getText()` before `getContentDescription()`, so a shorter string would shadow the label and the
 * two would drift.
 *
 * `label` appends context to that name. A Knowledge panel carries many of these at once, and
 * without it every one of them announces identically — the same defect the heart's `label` prop
 * was added for on 2026-09-25.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import BookmarkIcon from './BookmarkIcon.vue'

const props = withDefaults(
  defineProps<{
    /** Whether this fragment is already in the user's highlights. */
    saved: boolean
    /**
     * Signed out. Rendered rather than hidden (#1590) — hiding the control hid the capability, and
     * this is the cheapest entry to the learning loop. Tapping routes to sign-in and returns here.
     */
    gated?: boolean
    /** Which fragment this saves, so the two contexts can name themselves accurately. */
    context?: 'line' | 'insight'
    /** Extra context for the accessible name, when several sit on one screen (e.g. insight text). */
    label?: string
    /** Glyph size in px. The transcript's rows are denser than the panel's. */
    size?: number
  }>(),
  { gated: false, context: 'line', size: 16 },
)

const emit = defineEmits<{ (e: 'toggle'): void }>()

const { t } = useI18n()

const name = computed(() => {
  if (props.gated) return t('auth.signInToCapture')
  const base = props.saved
    ? t(props.context === 'insight' ? 'capture.savedInsight' : 'capture.savedLine')
    : t(props.context === 'insight' ? 'capture.saveInsight' : 'capture.saveLine')
  return props.label ? `${base} — ${props.label}` : base
})
</script>

<template>
  <button
    type="button"
    class="lp-tap rounded-full p-1 transition"
    :class="saved ? 'text-accent' : 'text-muted hover:text-accent'"
    :aria-pressed="gated ? undefined : saved"
    :aria-label="name"
    :title="name"
    data-testid="highlight-toggle"
    :data-saved="saved ? 'true' : undefined"
    @click.stop.prevent="emit('toggle')"
  >
    <BookmarkIcon :size="size" :filled="saved" />
    <!-- A real text node, matching `aria-label` exactly — see the note in the docblock above. -->
    <span class="sr-only">{{ name }}</span>
  </button>
</template>
