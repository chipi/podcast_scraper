<script setup lang="ts">
/**
 * An episode-grouped result block: the shared `EpisodeCard` as the header, its items collapsible
 * beneath it.
 *
 * Search already rendered its episode groups this way — the real `EpisodeCard` (artwork, show,
 * title, date) with the per-match rows as a sibling list below — and Revisit grouped by episode with
 * a bare text heading and no artwork. Same object, two presentations. This is the one block both
 * use, so an episode that carries matches looks like an episode that carries moments (operator
 * 2026-09-17: "add episode artwork to episode groups on revisit as we have on search", and make the
 * section collapsible "on both search and revisit").
 *
 * Groups start EXPANDED: collapsing is an affordance for a long page, not a new default that hides
 * results a listener has already asked for.
 */
import { ref } from "vue"
import { useI18n } from "vue-i18n"
import EpisodeCard from "./EpisodeCard.vue"
import type { EpisodeSummary } from "../services/types"

const props = defineProps<{
  episode: EpisodeSummary
  /**
   * The localised plural noun for what is inside — "matches", "moments". Used for the toggle's
   * wording; the COUNT is not repeated here because it already sits under the artwork in `#aside`.
   */
  noun: string
  /**
   * How many rows are in the slot — the toggle is suppressed for an empty group, where there is
   * nothing to collapse.
   *
   * A count rather than a `hasItems` boolean on purpose: Vue casts an ABSENT boolean prop to
   * `false`, so an optional `hasItems` silently meant "no items" for every caller that omitted it,
   * and the toggle never rendered.
   */
  itemCount: number
  /**
   * Render the header as a SLIM identity strip: 80px artwork instead of 128px, no action cluster,
   * tighter padding (operator 2026-09-17: "artwork is too big, use the smaller one", "episode
   * section is too thick, remove actions", "can we get it shorter").
   *
   * One prop rather than three because it is one decision — "this card labels the group, it is not
   * the thing being acted on". Each part is forwarded to `EpisodeCard`'s own variants rather than
   * restyled here, so the sizes cannot drift apart. Search leaves it off: its groups ARE the result,
   * and its header keeps the full card with actions.
   */
  slim?: boolean
  testid?: string
}>()

const { t } = useI18n()
const expanded = ref(true)
</script>

<template>
  <li
    class="overflow-hidden rounded-xl border border-border bg-surface"
    :data-testid="props.testid ?? 'episode-group'"
  >
    <div class="px-4 pt-1">
      <EpisodeCard
        :episode="episode"
        :compact="slim"
        :hide-actions="slim"
        :dense="slim"
      >
        <template v-if="$slots.aside" #aside><slot name="aside" /></template>
        <template v-if="$slots.meta" #meta><slot name="meta" /></template>
      </EpisodeCard>
    </div>
    <!-- The collapse control sits BETWEEN the episode and its items, so it reads as governing the
         list below it rather than the card above it. Muted label + accent chevron, matching the
         folded-cluster rows already inside Search's groups. -->
    <button
      v-if="itemCount > 0"
      type="button"
      class="flex w-full items-center gap-2 border-t border-border px-4 py-2.5 text-left"
      :aria-expanded="expanded"
      data-testid="episode-group-toggle"
      @click="expanded = !expanded"
    >
      <span class="text-xs font-semibold text-muted">
        {{ expanded ? t("episodeGroup.hide", { noun }) : t("episodeGroup.show", { noun }) }}
      </span>
      <span class="ml-auto text-xs font-bold text-accent" aria-hidden="true">
        {{ expanded ? "▲" : "▼" }}
      </span>
    </button>
    <!-- `v-show`, not `v-if`: collapsing must not throw away the state of what is inside (Search's
         folded clusters expand in place), and re-opening should be instant. -->
    <div v-show="expanded" data-testid="episode-group-body">
      <slot />
    </div>
  </li>
</template>
