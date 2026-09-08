<script setup lang="ts">
/**
 * A section of the Knowledge Panel that can be folded away, remembering how you left it.
 *
 * ## Why the panel needed this
 *
 * The spine grew: Summary, Key points, Topics & People, Insights, Related. On a real episode the
 * key points alone are ~8 sentences of ~200 characters, and the insight list can hold 36 rows. Every
 * visit meant scrolling past everything to reach the part you came for.
 *
 * ## Open by default, always
 *
 * Collapsing by default would hide content behind a tap nobody asked for — the panel's job is to
 * show the episode's substance, and a screen of closed headers shows none of it. So sections start
 * open and stay however YOU last left them.
 *
 * ## The count lives in the header
 *
 * A collapsed section still has to say what is inside it, otherwise folding it costs you the
 * knowledge that it exists. "Key points · 8" is legible closed; "Key points" is a locked door.
 *
 * ## Native `<details>`, not a div and a ref
 *
 * Keyboard operation, the disclosure role, Enter/Space, and the AT announcement of expanded state
 * all come from the element. Rebuilding that in userland is where a11y bugs live, and this panel
 * already has a modal contract to honour (S9).
 *
 * ## State is per USER, not per episode
 *
 * "I don't want to see related episodes" is a preference about the panel, not about one episode.
 * Keying it per episode would make the same choice again on every episode you open.
 */
import { ref, watch } from 'vue'

const props = defineProps<{
  /** Visible heading. */
  title: string
  /** Shown beside the title, so a folded section still says what it holds. Omit when not a count. */
  count?: number
  /**
   * Stable id for the remembered state, e.g. `insights`. Prefixed and stored under `lp.kp.<key>`,
   * alongside the app's other local UI preferences.
   */
  sectionKey: string
}>()

const STORAGE_PREFIX = 'lp.kp.'

function initialOpen(): boolean {
  try {
    // Absent means never touched, which is OPEN — see the note above about defaults.
    return localStorage.getItem(STORAGE_PREFIX + props.sectionKey) !== 'closed'
  } catch {
    // Private mode / storage disabled: a preference we cannot persist is not a reason to hide
    // content, so fall back to the default rather than to closed.
    return true
  }
}

const open = ref(initialOpen())

watch(open, (isOpen) => {
  try {
    localStorage.setItem(STORAGE_PREFIX + props.sectionKey, isOpen ? 'open' : 'closed')
  } catch {
    /* not persisting is survivable; the session still behaves */
  }
})
</script>

<template>
  <details
    :open="open"
    class="group"
    :data-testid="`kp-section-${sectionKey}`"
    @toggle="open = ($event.target as HTMLDetailsElement).open"
  >
    <summary
      class="flex cursor-pointer list-none items-center gap-2 py-1 marker:content-none [&::-webkit-details-marker]:hidden"
      :data-testid="`kp-section-toggle-${sectionKey}`"
    >
      <!-- The chevron is the only moving part: it rotates with the section's state and is
           `aria-hidden` because <details> already announces expanded/collapsed. -->
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="h-3 w-3 shrink-0 text-muted transition-transform group-open:rotate-90"
        aria-hidden="true"
      >
        <path d="m9 6 6 6-6 6" />
      </svg>
      <h3 class="lp-section">
        {{ title }}<template v-if="count !== undefined"> · {{ count }}</template>
      </h3>
    </summary>
    <div class="pt-1">
      <slot />
    </div>
  </details>
</template>
