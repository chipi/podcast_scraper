<script setup lang="ts">
/**
 * "Your Week" — the in-app personal digest (#1412), the highlight of the home page.
 *
 * The SAME rollup the email sends (revisit + new-in-follows + trending-in-your-corpus), served
 * live and decoupled from email consent — so turning the email off never loses the capability;
 * the email is just the edge for when you don't visit.
 *
 * Two layouts, a per-user preference (synced via userPreferences, so it follows the user across
 * devices): `compact` = one rail of the week's top highlights; `full` = a labelled rail per
 * section. Flip it inline with "Show more / Show less" (same preference the Your Week setting
 * writes). Hidden entirely when signed-out or nothing is due yet.
 */
import { computed, ref, watch } from 'vue'
import { useSectionState } from '../composables/useSectionState'
import SectionStatus from './SectionStatus.vue'
import { RouterLink } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { getYourWeek } from '../services/api'
import type { YourWeekItem, YourWeekResponse, YourWeekSectionKind } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { useUserPreferencesStore } from '../stores/userPreferences'
import CardRail from './CardRail.vue'
import YourWeekCard from './YourWeekCard.vue'

const LAYOUT_PREF_KEY = 'lp.yourweek.layout'
const COMPACT_MAX = 6

const { t } = useI18n()
const auth = useAuthStore()
const userPrefs = useUserPreferencesStore()

const section = useSectionState<YourWeekResponse | null>(null)
const data = computed(() => section.data.value)
const layout = ref<'compact' | 'full'>('compact')

const sections = computed(() => data.value?.sections ?? [])
const nonEmptySections = computed(() => sections.value.filter((s) => s.items.length > 0))
/**
 * Renders for ANY signed-in user (#1591), not only when something is due.
 *
 * It used to self-hide whenever every section was empty, which meant a brand-new user — the person
 * most in need of learning that a weekly digest exists — got no hint of it, and an API outage was
 * indistinguishable from a quiet week. Now the section persists and explains, per row, what will
 * appear there and how to earn it.
 */
const show = computed(() => auth.isAuthenticated)
const hasContent = computed(() => nonEmptySections.value.length > 0)

/**
 * The first-run rows. `new_in_follows` is USER-empty — it is blank because you follow nothing, an
 * action you can take — so it carries a link to do it. The other two are SYSTEM-empty: they fill as
 * you listen, with nothing to click today, so they explain rather than prompt.
 *
 * The copy has to describe what the system ACTUALLY does, because this is the screen a user reads
 * when nothing has appeared yet — the moment they are deciding whether the feature is broken (#38):
 *
 *   - revisit said "Moments you capture come back here" with no hint of the 2-day first rung, so a
 *     user who captured something and looked immediately was told a promise the ladder had not
 *     broken yet;
 *   - trending said "what's heating up across them shows here", which is unconditionally false on a
 *     corpus that never ships the temporal-velocity enrichment — no amount of listening fills it.
 */
const FIRST_RUN: { kind: YourWeekSectionKind; actionable: boolean }[] = [
  { kind: 'new_in_follows', actionable: true },
  { kind: 'new_in_interests', actionable: true },
  { kind: 'revisit', actionable: false },
  { kind: 'trending_in_your_corpus', actionable: false },
]

// Compact = the best few items across all sections, kept in section order.
const compactItems = computed<YourWeekItem[]>(() =>
  nonEmptySections.value.flatMap((s) => s.items).slice(0, COMPACT_MAX),
)

const sectionLabel = (kind: YourWeekSectionKind): string => t(`home.yourWeekSection.${kind}`)

function toggleLayout(): void {
  layout.value = layout.value === 'compact' ? 'full' : 'compact'
  void userPrefs.set(LAYOUT_PREF_KEY, layout.value)
}

async function load(): Promise<void> {
  if (!auth.isAuthenticated) return
  await userPrefs.hydrate() // idempotent; ensures the synced layout pref is loaded before we read it
  const pref = userPrefs.get<string>(LAYOUT_PREF_KEY)
  if (pref === 'full' || pref === 'compact') layout.value = pref
  await section.load(() => getYourWeek())
}

// Load when already authed AND re-load when auth resolves later: a fresh page load hydrates auth
// asynchronously, so a one-shot onMounted fetch can race it and leave the section wrongly empty.
// On sign-out, drop the prior user's rollup so a same-tab re-sign-in can never flash their data.
watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (authed) load()
    else section.data.value = null
  },
  { immediate: true },
)
</script>

<template>
  <section v-if="show" data-testid="your-week" class="mt-7">
    <div class="mb-3 flex items-baseline justify-between gap-3">
      <div>
        <span class="lp-kicker text-accent">{{ t('home.yourWeekKicker') }}</span>
        <h2 class="lp-section">{{ t('home.yourWeek') }}</h2>
      </div>
      <button
        v-if="hasContent"
        type="button"
        data-testid="yourweek-toggle"
        class="shrink-0 text-sm font-bold text-accent"
        @click="toggleLayout"
      >
        {{ layout === 'compact' ? t('home.yourWeekShowMore') : t('home.yourWeekShowLess') }}
      </button>
    </div>

    <SectionStatus :phase="section.phase.value" :rows="2" @retry="load" />

    <!-- First run: the digest exists before it has anything in it. Each row says what will appear
         and how to earn it; the one that is empty because of an action YOU can take links to it. -->
    <ul
      v-if="section.isReady.value && !hasContent"
      class="divide-y divide-border rounded-xl border border-border"
      data-testid="yourweek-firstrun"
    >
      <li v-for="row in FIRST_RUN" :key="row.kind" class="px-4 py-3">
        <p class="text-sm font-bold text-canvas-foreground">{{ sectionLabel(row.kind) }}</p>
        <p class="mt-0.5 text-xs text-muted">
          {{ t(`home.yourWeekFirstRun.${row.kind}`) }}
          <RouterLink
            v-if="row.actionable"
            :to="{ name: 'catalog' }"
            class="font-bold text-accent no-underline"
          >{{ t('home.yourWeekFindShows') }}</RouterLink>
        </p>
      </li>
    </ul>

    <!-- Compact: a single rail of the week's highlights. -->
    <CardRail v-if="hasContent && layout === 'compact'">
      <li
        v-for="(item, i) in compactItems"
        :key="`${item.episode_slug}-${i}`"
        class="w-60 shrink-0"
      >
        <YourWeekCard :item="item" />
      </li>
    </CardRail>

    <!-- Full: one labelled rail per non-empty section. -->
    <div v-else-if="hasContent" class="flex flex-col gap-5">
      <div v-for="s in nonEmptySections" :key="s.kind">
        <h3 class="lp-kicker mb-2">{{ sectionLabel(s.kind) }}</h3>
        <CardRail>
          <li
            v-for="(item, i) in s.items"
            :key="`${item.episode_slug}-${i}`"
            class="w-60 shrink-0"
          >
            <YourWeekCard :item="item" />
          </li>
        </CardRail>
      </div>
    </div>
  </section>
</template>
