<script setup lang="ts">
/**
 * "What this show's about" — show-level signals on the consumer show page (PodcastView):
 * dominant themes, the topics it covers, what's trending here, and who's on it. Reads
 * GET /api/app/podcasts/{feedId}/signals — a listener projection of the operator feed-signals
 * (the operator-only grounding/QA score is dropped server-side). Chips emit `open` so the page
 * opens the shared entity card. Best-effort: the whole band hides when the show has no signals.
 *
 * People shows one section (key people) — on a real show "recurring guests" is nearly the same
 * set (hosts dominate both), so we don't render it twice.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { kindPill } from '../utils/interests'
import CollapsibleSection from './CollapsibleSection.vue'
import SectionStatus from './SectionStatus.vue'
import { useSectionState } from '../composables/useSectionState'
import { getPodcastSignals } from '../services/api'
import type { PodcastSignals } from '../services/types'

const props = defineProps<{ feedId: string }>()
const emit = defineEmits<{
  (e: 'open', payload: { kind: 'topic' | 'person' | 'storyline' | 'theme'; id: string }): void
}>()
const { t } = useI18n()

/**
 * `useSectionState` so a failed signals call does not hide the whole band as if the show had
 * nothing to say about itself. This is #1591's defect — caught into null, hidden when empty — and
 * here it removed an entire titled section from the show page, silently.
 */
const section = useSectionState<PodcastSignals | null>(null)
const signals = computed(() => section.data.value)
/** Drops a slow reply for a show the reader has already navigated away from. */
const requestSeq = ref(0)

async function load(): Promise<void> {
  const mine = requestSeq.value + 1
  requestSeq.value = mine
  await section.load(async () => {
    const s = await getPodcastSignals(props.feedId, 10)
    if (mine !== requestSeq.value) throw new Error('superseded')
    return s
  })
}

watch(() => props.feedId, () => void load(), { immediate: true })


const topics = computed(() => signals.value?.top_topics ?? [])
const people = computed(() => signals.value?.key_people ?? [])

/**
 * Localise a speaker role for the people chips — the SAME map and keys `KnowledgePanel` uses, so
 * the show page and the episode card cannot drift on what a role is called.
 *
 * An unrecognised role falls through to its raw string rather than vanishing (the idiom
 * `TrendingSparkChips` and `KnowledgePanel` both follow), so a new server-side role still shows.
 * An ABSENT role returns "", and the caller renders no badge — older KGs carry no role, and an
 * unbadged chip is honest where a guessed "mentioned" would not be.
 */
const ROLE_LABEL_KEYS: Record<string, string> = {
  host: 'ec.roleHost',
  guest: 'ec.roleGuest',
  mentioned: 'ec.roleMentioned',
}
function roleLabel(role: string | null | undefined): string {
  if (!role) return ''
  const key = ROLE_LABEL_KEYS[role.toLowerCase()]
  return key ? t(key) : role
}

/**
 * Distinctiveness, which is what the band's title actually asks — now for every kind, not only
 * topics (operator 2026-10-12).
 *
 * Coverage alone can't answer "what this show's about": in the validation corpus every show
 * covers "expert interviews" in all four of its episodes, so it ties with the one topic that
 * identifies the show. `lift` (a subject's share of THIS show's episodes over its share of the
 * whole library's, computed server-side in `feed_signals`; for a theme or storyline, episodes
 * touching any of its topics) breaks that tie on meaning: wallpaper lands at 1.0, a signature
 * subject well above it.
 *
 * Distinctive = `lift >= MIN_LIFT` AND in ≥2 of the show's episodes (one passing mention on a
 * short show can score a high lift while saying nothing about it). An unknown lift is never
 * promoted. The distinctive ones lead their own kind's row with a "4×" badge — there used to be a
 * separate "Distinctive to this show" row of topics only, which read as a fifth kind, said
 * nothing about themes or storylines, and kept its "how much more" in a hover tooltip a phone
 * never shows.
 */
const MIN_LIFT = 1.5
const MIN_EPISODES_FOR_DISTINCTIVE = 2

function isDistinctive(x: { lift?: number | null; episode_count?: number | null }): boolean {
  return (x.lift ?? 0) >= MIN_LIFT && (x.episode_count ?? 0) >= MIN_EPISODES_FOR_DISTINCTIVE
}

/** Distinctive first, by lift; the rest keep the server's order (most-covered first). */
function distinctiveFirst<T extends { lift?: number | null; episode_count?: number | null }>(
  list: T[],
): T[] {
  const lead = list.filter(isDistinctive).sort((a, b) => (b.lift ?? 0) - (a.lift ?? 0))
  return [...lead, ...list.filter((x) => !isDistinctive(x))]
}

const storylines = computed(() => distinctiveFirst(signals.value?.dominant_storylines ?? []))
const themeList = computed(() => distinctiveFirst(signals.value?.dominant_themes ?? []))
const topicList = computed(() => distinctiveFirst(topics.value))

/** "9" not "9.0", "1.8" as-is. */
function formatLift(lift: number | null | undefined): string {
  if (lift == null) return ''
  return Number.isInteger(lift) ? String(lift) : lift.toFixed(1)
}
const hasAny = computed(
  () =>
    storylines.value.length > 0 ||
    themeList.value.length > 0 ||
    topics.value.length > 0 ||
    people.value.length > 0,
)
</script>

<template>
  <!-- A failed signals call keeps the band, titled, with a retry — rather than deleting a whole
       section of the show page and leaving the reader to assume the show has nothing to say about
       itself. A show that genuinely has no signals still renders nothing (#1591: hide when the
       SYSTEM is empty). -->
  <section
    v-if="section.isError.value"
    class="mb-6 rounded-2xl border border-border bg-surface p-4"
    data-testid="podcast-signals-error"
  >
    <h2 class="lp-section mb-3">{{ t('podcast.about') }}</h2>
    <SectionStatus :phase="section.phase.value" @retry="load()" />
  </section>

  <section
    v-else-if="hasAny"
    class="mb-6 rounded-2xl border border-border bg-surface p-4"
    data-testid="podcast-signals"
  >
    <!-- Foldable like the episode panel's sections (operator 2026-10-04): open by default, and the
         choice is remembered per listener across every show — "I don't need this band" is a
         preference about the page, not about one show. -->
    <CollapsibleSection :title="t('podcast.about')" section-key="show-about">
      <!-- One row per kind (operator 2026-10-12): Storylines, Themes, Topics, Key people. In each,
           the subjects DISTINCTIVE to this show come first with their "4×" — how many times more
           of this show's episodes they run through than the library's. -->
      <div v-if="storylines.length" class="mb-3">
        <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigStorylines') }}</h3>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="th in storylines"
            :key="th.storyline_id"
            type="button"
            data-testid="ps-theme"
            :data-distinctive="isDistinctive(th) || undefined"
            class="lp-storyline-chip rounded-full px-2.5 py-1 text-xs font-semibold text-storyline transition disabled:opacity-60"
            :disabled="!th.anchor_topic_id"
            @click="th.anchor_topic_id && emit('open', { kind: 'storyline', id: th.anchor_topic_id })"
          >
            {{ th.label }}<span
              v-if="isDistinctive(th)"
              data-testid="ps-lift"
              class="ml-1 rounded-full bg-canvas/50 px-1.5 py-0.5 text-[0.6rem] font-bold"
              :aria-label="t('podcast.sigLiftHint', { factor: formatLift(th.lift) })"
              :title="t('podcast.sigLiftHint', { factor: formatLift(th.lift) })"
            >{{ t('podcast.sigLift', { factor: formatLift(th.lift) }) }}</span>
          </button>
        </div>
      </div>

      <div v-if="themeList.length" class="mb-3">
        <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigThemes') }}</h3>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="th in themeList"
            :key="th.theme_id"
            type="button"
            data-testid="ps-theme-cluster"
            :data-distinctive="isDistinctive(th) || undefined"
            class="rounded-full px-2.5 py-1 text-xs transition hover:opacity-80"
            :class="kindPill('theme')"
            @click="emit('open', { kind: 'theme', id: th.theme_id })"
          >
            {{ th.label }}<span
              v-if="isDistinctive(th)"
              data-testid="ps-lift"
              class="ml-1 rounded-full bg-canvas/50 px-1.5 py-0.5 text-[0.6rem] font-bold"
              :aria-label="t('podcast.sigLiftHint', { factor: formatLift(th.lift) })"
              :title="t('podcast.sigLiftHint', { factor: formatLift(th.lift) })"
            >{{ t('podcast.sigLift', { factor: formatLift(th.lift) }) }}</span>
          </button>
        </div>
      </div>

      <div v-if="topicList.length" class="mb-3">
        <h3 class="lp-kicker mb-1.5" data-testid="ps-topics-heading">{{ t('podcast.sigTopics') }}</h3>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="tp in topicList"
            :key="tp.topic_id"
            type="button"
            data-testid="ps-topic"
            :data-distinctive="isDistinctive(tp) || undefined"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
            @click="emit('open', { kind: 'topic', id: tp.topic_id })"
          >
            {{ tp.label }}<span
              v-if="isDistinctive(tp)"
              data-testid="ps-lift"
              class="ml-1 rounded-full bg-canvas/50 px-1.5 py-0.5 text-[0.6rem] font-bold"
              :aria-label="t('podcast.sigLiftHint', { factor: formatLift(tp.lift) })"
              :title="t('podcast.sigLiftHint', { factor: formatLift(tp.lift) })"
            >{{ t('podcast.sigLift', { factor: formatLift(tp.lift) }) }}</span>
          </button>
        </div>
      </div>

      <!-- Corpus-wide "Trending here" row removed from the show page (operator): its velocity is
           corpus-wide, not this-show-specific, so it read as clutter beside the distinctive/topics. -->

      <div v-if="people.length">
        <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigPeople') }}</h3>
        <div class="flex flex-wrap gap-1.5">
          <!--
            Role badge + role ORDER (operator 2026-09-27): mark people host/guest/mentioned in the
            same style as the episode Insights panel, and order by it too.

            Same markup and the same i18n keys as `kp-person-role` (`ec.roleHost` / `ec.roleGuest` /
            `ec.roleMentioned`), so one idea has one appearance across the two surfaces that list
            people.

            The ORDER is server-side, in `feed_signals.key_people` — host > guest > mentioned, then
            footprint, then name. It has to be there rather than here, because `top_k` truncates:
            sorting on the client would only rearrange whichever names survived a count-based cut, and
            a show's own host could be missing from the list altogether. So this renders the array as
            given.

            A roleless person (older KGs carry none) renders unbadged rather than being labelled
            "mentioned" — the badge reports what the KG knows, and an absent role is not a claim.
          -->
          <button
            v-for="p in people"
            :key="p.person_id"
            type="button"
            data-testid="ps-person"
            :data-role="p.role?.toLowerCase()"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
            @click="emit('open', { kind: 'person', id: p.person_id })"
          >
            {{ p.name
            }}<span
              v-if="roleLabel(p.role)"
              data-testid="ps-person-role"
              class="ml-1 rounded-full bg-canvas/50 px-1.5 py-0.5 text-[0.6rem] font-bold uppercase tracking-wide"
              >{{ roleLabel(p.role) }}</span
            >
            <span class="text-muted">· {{ p.episode_count }}</span>
          </button>
        </div>
      </div>
    </CollapsibleSection>
  </section>
</template>
