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
 * switches between two depending on whether it is showing host episodes). They pass it in; the
 * list only owns the rows and the paging. What IS owned here is the "newest first" kicker, because
 * that was the thing that drifted.
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
  }
)

const visible = computed(() => props.episodes.slice(0, shown.value))
const remaining = computed(() => Math.max(0, props.episodes.length - visible.value.length))
</script>

<template>
  <!-- `aria-live`: pressing "show more" appends rows silently otherwise — a screen-reader user
       activates the control and hears nothing at all, which is indistinguishable from a dead
       button. `atomic=false` so only the added rows are announced, not the whole list again. -->
  <ul class="flex flex-col" aria-live="polite" aria-atomic="false">
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
    @click="shown += PAGE"
  >
    {{ t("ec.moreEpisodes", { count: Math.min(remaining, PAGE) }) }}
  </button>
</template>
