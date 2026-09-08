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

const section = useSectionState<YourWeekResponse | null>(null, { cacheKey: 'home.yourweek' })
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
        <span class="lp-kicker text-muted">{{ t('home.yourWeekKicker') }}</span>
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

    <!-- First run: the digest exists before it has anything in it (#1591 — it TEACHES rather than
         self-hiding, and that stands).

         ONE LINE, not four rows (#1978). Measured on a brand-new account, the four-row bordered
         list rendered 373px tall with zero episode links, sitting between the hero and "What's
         new" — roughly 44% of the first viewport spent promising future value to the only audience
         that has no history yet, which is every tester on their first run. The four rows also said
         the same thing four ways ("… will land here"), and two of them offered the identical
         "Find shows →" link.

         So the teaching survives and the real estate does not: what it is, and the one action that
         starts filling it. Same move the set-your-interests offer made in #1964 — an explanation is
         a line, not an announcement. Once there IS content this branch never renders and the full
         digest returns unchanged. -->
    <p
      v-if="section.isReady.value && !hasContent"
      class="text-sm text-muted"
      data-testid="yourweek-firstrun"
    >
      {{ t('home.yourWeekFirstRunLine') }}
      <RouterLink
        :to="{ name: 'browse', query: { tab: 'shows' } }"
        class="font-bold text-accent no-underline"
      >{{ t('home.yourWeekFindShows') }}</RouterLink>
    </p>

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
