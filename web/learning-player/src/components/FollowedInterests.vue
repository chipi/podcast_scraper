<script setup lang="ts">
/**
 * Followed interests — the topics, people and storylines you follow (the + on trending topics /
 * storylines / entity cards), grouped by type like the Saved tab's sections. Following these was
 * previously invisible: the tokens went into your interests profile but nothing surfaced them. This
 * makes them visible, navigable and unfollow-able. Complements the followed-shows grid above it.
 *
 * Governed by the Following filter bar (#2042 follow-up): `search` type-to-filters by label,
 * `sort` orders (recent / A–Z), `visibleTypes` gates which kinds show, and each section caps to the
 * top N with "Show all" so 100+ follows stay scannable. A non-empty search lifts the caps.
 *
 * Labels: clusters resolve via the top-cluster set + storylines list; topics/people de-slug from
 * their id (`topic:personal-growth` → "personal growth"), matching ProfileView.
 */
import { computed, onMounted, ref } from 'vue'
import { INTEREST_PREFIX, interestKind } from '../utils/interests'
import CloseIcon from "./CloseIcon.vue"
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useInterestsStore } from '../stores/interests'
import { getStorylines, getTopClusters } from '../services/api'
import { matchesQuery } from '../utils/textFilter'
import { useCappedSections } from '../composables/useCappedSections'
import ShowAllToggle from './ShowAllToggle.vue'
import type { Storyline } from '../services/types'

const props = defineProps<{ search?: string; sort?: string; visibleTypes?: string[] }>()

const { t } = useI18n()
const router = useRouter()
const interests = useInterestsStore()
const { ids } = storeToRefs(interests)

const clusterLabels = ref<Map<string, string>>(new Map())
const storylines = ref<Storyline[]>([])

onMounted(async () => {
  await interests.ensureLoaded().catch(() => {})
  const [clusters, stories] = await Promise.all([
    getTopClusters(50).catch(() => []),
    getStorylines(50).catch(() => []),
  ])
  clusterLabels.value = new Map(clusters.map((c) => [c.id, c.label]))
  storylines.value = stories
})

const storylineById = computed(() => new Map(storylines.value.map((s) => [s.id, s])))
const deslug = (id: string) => id.replace(/^(tc|thc|topic|person):/, '').replace(/[-_]+/g, ' ')
function labelOf(id: string): string {
  return clusterLabels.value.get(id) ?? storylineById.value.get(id)?.label ?? deslug(id)
}

const caps = useCappedSections()
const searchActive = computed(() => (props.search ?? '').trim() !== '')
function typeVisible(key: string): boolean {
  return !props.visibleTypes || props.visibleTypes.length === 0 || props.visibleTypes.includes(key)
}
/** Filter by label, then order: A–Z by label, or 'recent' (interests append newest-last → reverse). */
function arrange(list: string[]): string[] {
  const filtered = list.filter((id) => matchesQuery(labelOf(id), props.search ?? ''))
  if (props.sort === 'title') return [...filtered].sort((a, b) => labelOf(a).localeCompare(labelOf(b)))
  return [...filtered].reverse()
}

// Grouped by KIND, not by prefix. `interestKind` is the one place that maps a token to what a
// reader calls it, and asking it here means the wire prefixes can be renamed without touching this
// file — which matters, because `utils/interests` records that the prefixes are inverted against
// the product names (`thc:` is a storyline, `tc:` is a theme) and that the rename is deferred.
// Hand-rolled prefix lists are exactly what that file warns go stale.
const topics = computed(() => arrange(ids.value.filter((i) => interestKind(i) === 'topic')))
const persons = computed(() => arrange(ids.value.filter((i) => interestKind(i) === 'person')))
// Themes and storylines are still shown TOGETHER under "storylines" — unchanged behaviour, now
// said out loud. They are different things (a theme groups by meaning, a storyline by
// co-occurrence) and giving the theme its own group is the next step, not this one.
const storylineTokens = computed(() =>
  arrange(ids.value.filter((i) => ['storyline', 'theme'].includes(interestKind(i)))),
)
// "Genuinely follows nothing" is about the UNFILTERED set — otherwise a search that matches none of
// your follows showed "you're not following anything" (Fable-5 review S3). When you DO follow things
// but a search hid them all, say that instead; a type-chip exclusion just renders nothing (no lie).
const followsAnything = computed(() => ids.value.some((i) => INTEREST_PREFIX.test(i)))
const isEmpty = computed(() => !followsAnything.value)
const noSearchMatch = computed(
  () =>
    followsAnything.value &&
    searchActive.value &&
    !topics.value.length &&
    !persons.value.length &&
    !storylineTokens.value.length,
)

function unfollow(id: string): void {
  void interests.toggle(id)
}

function openStoryline(id: string): void {
  // A THEME goes straight to its own page — its id IS the route param, no lookup needed. Until the
  // theme page existed this branch did not, so a followed theme sat in this group as a DEAD chip:
  // `storylineById` only resolves `thc:` storylines, so `anchor_topic_id` was always undefined for
  // a `tc:` token and the tap did nothing.
  if (interestKind(id) === 'theme') {
    void router.push({ name: 'theme', params: { id } })
    return
  }
  // A STORYLINE has no endpoint of its own, so it opens keyed by its anchor topic (F4.5).
  // Unresolvable → the chip stays display-only.
  const s = storylineById.value.get(id)
  if (s?.anchor_topic_id) void router.push({ name: 'storyline', params: { id: s.anchor_topic_id } })
}
</script>

<template>
  <div data-testid="followed-interests">
    <p v-if="isEmpty" class="text-sm text-muted">{{ t('library.followingEmpty') }}</p>
    <p v-else-if="noSearchMatch" class="text-sm text-muted" data-testid="following-no-match">{{ t('library.followingEmptyFiltered') }}</p>

    <section v-if="typeVisible('topics') && topics.length" class="mb-5">
      <h2 class="lp-section mb-2">
        {{ t('library.followingTopics') }}
        <span class="lp-kicker ml-1 font-normal">{{ topics.length }}</span>
      </h2>
      <ul class="flex flex-wrap gap-1.5">
        <li v-for="id in caps.visible('topics', topics, searchActive)" :key="id" class="inline-flex items-center rounded-full bg-overlay">
          <button
            type="button"
            class="max-w-[12rem] truncate py-1 pl-3 pr-1.5 text-sm font-semibold text-topic transition hover:opacity-80"
            @click="router.push({ name: 'topic', params: { id } })"
          >
            {{ labelOf(id) }}
          </button>
          <button
            type="button"
            class="rounded-r-full py-1 pl-1 pr-2.5 text-xs text-muted transition hover:text-danger"
            :aria-label="t('library.unfollow', { label: labelOf(id) })"
            data-testid="unfollow"
            @click="unfollow(id)"
          >
            <CloseIcon />
          </button>
        </li>
      </ul>
      <ShowAllToggle
        v-if="caps.overflows(topics.length, searchActive)"
        :expanded="caps.expanded.has('topics')"
        :count="topics.length"
        @toggle="caps.toggle('topics')"
      />
    </section>

    <section v-if="typeVisible('people') && persons.length" class="mb-5">
      <h2 class="lp-section mb-2">
        {{ t('library.followingPeople') }}
        <span class="lp-kicker ml-1 font-normal">{{ persons.length }}</span>
      </h2>
      <ul class="flex flex-wrap gap-1.5">
        <li
          v-for="id in caps.visible('people', persons, searchActive)"
          :key="id"
          class="inline-flex items-center rounded-full bg-overlay"
        >
          <button
            type="button"
            class="max-w-[12rem] truncate py-1 pl-3 pr-1.5 text-sm font-semibold text-person transition hover:opacity-80"
            @click="router.push({ name: 'person', params: { id } })"
          >
            {{ labelOf(id) }}
          </button>
          <button
            type="button"
            class="rounded-r-full py-1 pl-1 pr-2.5 text-xs text-muted transition hover:text-danger"
            :aria-label="t('library.unfollow', { label: labelOf(id) })"
            data-testid="unfollow"
            @click="unfollow(id)"
          >
            <CloseIcon />
          </button>
        </li>
      </ul>
      <ShowAllToggle
        v-if="caps.overflows(persons.length, searchActive)"
        :expanded="caps.expanded.has('people')"
        :count="persons.length"
        @toggle="caps.toggle('people')"
      />
    </section>

    <section v-if="typeVisible('storylines') && storylineTokens.length">
      <h2 class="lp-section mb-2">
        {{ t('library.followingStorylines') }}
        <span class="lp-kicker ml-1 font-normal">{{ storylineTokens.length }}</span>
      </h2>
      <ul class="flex flex-wrap gap-1.5">
        <li
          v-for="id in caps.visible('storylines', storylineTokens, searchActive)"
          :key="id"
          class="inline-flex items-center rounded-full bg-overlay"
        >
          <button
            type="button"
            class="max-w-[14rem] truncate py-1 pl-3 pr-1.5 text-sm font-semibold text-canvas-foreground transition hover:opacity-80"
            @click="openStoryline(id)"
          >
            {{ labelOf(id) }}
          </button>
          <button
            type="button"
            class="rounded-r-full py-1 pl-1 pr-2.5 text-xs text-muted transition hover:text-danger"
            :aria-label="t('library.unfollow', { label: labelOf(id) })"
            data-testid="unfollow"
            @click="unfollow(id)"
          >
            <CloseIcon />
          </button>
        </li>
      </ul>
      <ShowAllToggle
        v-if="caps.overflows(storylineTokens.length, searchActive)"
        :expanded="caps.expanded.has('storylines')"
        :count="storylineTokens.length"
        @toggle="caps.toggle('storylines')"
      />
    </section>
  </div>
</template>
