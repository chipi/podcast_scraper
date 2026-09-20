<script setup lang="ts">
/**
 * The "discussed in N episodes" list shared by every entity surface — topic, storyline, person, org
 * (operator 2026-09-19).
 *
 * ## Why this is a component and not four lists
 *
 * It already WAS four lists. Each of Topic / Person / Org / Storyline rendered its own
 * `<ul><li v-for><EpisodeRow>`, uncapped, and the four had already drifted: three stated
 * "newest first" under the heading and the storyline did not. Uncapped is the real problem — a
 * topic with sixty episodes rendered sixty rows and buried everything below it, which is why the
 * conversation arc and the perspectives block were unreachable in practice on a busy topic.
 *
 * So: ten rows, then a control that adds ten more. The cap lives here rather than in each caller so
 * the next surface that needs an episode list inherits the behaviour instead of re-deciding it.
 *
 * ## Why the count in the heading is not this component's job
 *
 * Callers word their own heading ("Discussed in N episodes", "In N episodes", and the person card
 * switches between two depending on whether it is showing host episodes). They pass it in; this
 * owns the rows, the cap and the paging.
 *
 * The "newest first" kicker is NOT owned here, and that is worth stating because the drift which
 * motivated this extraction was precisely that kicker — three callers said it and the storyline
 * did not. It sits in the heading, the heading belongs to the caller, and a fifth caller can
 * forget it exactly as StorylineView did. Moving it would mean owning the whole heading row,
 * which is the thing the callers legitimately differ on.
 *
 * Nor does this SORT. "Newest first" is the server's guarantee (`_sorted_episode_cards`); nothing
 * here verifies it, so the kicker is true by contract rather than by construction.
 */
import { computed, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import EpisodeRow from "./EpisodeRow.vue"
import type { EpisodeSummary } from "../services/types"

const PAGE = 10

const props = defineProps<{ episodes: EpisodeSummary[] }>()

const { t } = useI18n()

const shown = ref(PAGE)
// Re-collapse when the list itself changes. These surfaces drill in place — open a sibling topic
// from a chip and the same component instance is handed a different entity's episodes — so without
// this you would land on the new topic already scrolled twenty rows deep.
watch(
  () => props.episodes,
  () => {
    shown.value = PAGE
    announcement.value = ""
  }
)

const visible = computed(() => props.episodes.slice(0, shown.value))
// Empty until the user presses, so nothing is announced on mount.
const announcement = ref("")

function reveal(): void {
  const before = visible.value.length
  shown.value += PAGE
  announcement.value = t("ec.moreEpisodesShown", { count: visible.value.length - before })
}
const remaining = computed(() => Math.max(0, props.episodes.length - visible.value.length))
</script>

<template>
  <!-- The announcement lives on a STATUS element, not on the list.
       A live region wrapped around the `<ul>` fires on MOUNT too, so a screen reader read all ten
       initial episodes aloud before the user had done anything — worse than the silence it was
       meant to fix. This is empty until a press, so it announces the delta and nothing else. -->
  <p aria-live="polite" class="sr-only">{{ announcement }}</p>
  <ul class="flex flex-col">
    <li v-for="e in visible" :key="e.slug">
      <EpisodeRow :episode="e" />
    </li>
  </ul>
  <!-- Same full-width treatment as the Shows and Episodes load-more controls (operator 2026-09-19):
       one shape for "there is more of this list below". -->
  <button
    v-if="remaining > 0"
    type="button"
    class="mt-4 w-full rounded-xl border border-border py-2.5 text-sm font-bold text-accent transition hover:bg-overlay"
    data-testid="entity-episodes-more"
    @click="reveal"
  >
    {{ t("ec.moreEpisodes", { count: Math.min(remaining, PAGE) }) }}
  </button>
</template>
