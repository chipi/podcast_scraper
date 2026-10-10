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
import MomentsLink from "./MomentsLink.vue"
import type { EpisodeSummary } from "../services/types"

// Five, then five more per press (operator 2026-10-08: every entity episode list, topic / person /
// org / storyline / theme). Ten filled a phone screen before the next section could be seen.
const PAGE = 5

const props = defineProps<{
  episodes: EpisodeSummary[]
  /**
   * PAGED ON THE SERVER (2026-10-08): the full list's length when `episodes` is only its first
   * page, with `loadMore` fetching the rest a page at a time. The cards used to return every
   * episode — a storyline 99, 1.29 MB on prod — to show five. Omit both for a list that is whole.
   */
  total?: number
  loadMore?: (offset: number, limit: number) => Promise<EpisodeSummary[]>
}>()

const { t } = useI18n()

const shown = ref(PAGE)
const loaded = ref<EpisodeSummary[]>([...props.episodes])
const loading = ref(false)
// Re-collapse when the list itself changes. These surfaces drill in place — open a sibling topic
// from a chip and the same component instance is handed a different entity's episodes — so without
// this you would land on the new topic already scrolled twenty rows deep.
watch(
  () => props.episodes,
  (eps) => {
    loaded.value = [...eps]
    shown.value = PAGE
    announcement.value = ""
  }
)

const total = computed(() => Math.max(props.total ?? 0, loaded.value.length))
const visible = computed(() => loaded.value.slice(0, shown.value))
// Empty until the user presses, so nothing is announced on mount.
const announcement = ref("")

async function reveal(): Promise<void> {
  const before = visible.value.length
  const want = shown.value + PAGE
  if (props.loadMore && loaded.value.length < Math.min(want, total.value)) {
    loading.value = true
    try {
      const next = await props.loadMore(loaded.value.length, PAGE)
      const seen = new Set(loaded.value.map((e) => e.slug))
      loaded.value = [...loaded.value, ...next.filter((e) => !seen.has(e.slug))]
    } catch {
      // Nothing more arrived; the button stays so the listener can try again.
    } finally {
      loading.value = false
    }
  }
  shown.value = want
  announcement.value = t("ec.moreEpisodesShown", { count: visible.value.length - before })
}
const remaining = computed(() => Math.max(0, total.value - visible.value.length))
</script>

<template>
  <!-- The announcement lives on a STATUS element, not on the list.
       A live region wrapped around the `<ul>` fires on MOUNT too, so a screen reader read all ten
       initial episodes aloud before the user had done anything — worse than the silence it was
       meant to fix. This is empty until a press, so it announces the delta and nothing else. -->
  <p aria-live="polite" class="sr-only">{{ announcement }}</p>
  <ul class="flex flex-col">
    <li v-for="e in visible" :key="e.slug">
      <EpisodeRow :episode="e">
        <!-- Topic, person, storyline, theme and org pages are where a listener explores, so each
             episode offers its reel (operator 2026-10-10). -->
        <template v-if="e.has_gi" #trailing><MomentsLink :slug="e.slug" class="self-center" /></template>
      </EpisodeRow>
    </li>
  </ul>
  <!-- Same full-width treatment as the Shows and Episodes load-more controls (operator 2026-09-19):
       one shape for "there is more of this list below". -->
  <button
    v-if="remaining > 0"
    type="button"
    class="mt-4 w-full rounded-xl border border-border py-2.5 text-sm font-bold text-accent transition hover:bg-overlay"
    data-testid="entity-episodes-more"
    :disabled="loading"
    @click="reveal"
  >
    {{ t("ec.moreEpisodes", { count: Math.min(remaining, PAGE) }) }}
  </button>
</template>
