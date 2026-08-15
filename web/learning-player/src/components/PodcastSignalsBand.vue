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
import SectionStatus from './SectionStatus.vue'
import { useSectionState } from '../composables/useSectionState'
import { getPodcastSignals } from '../services/api'
import type { PodcastSignals } from '../services/types'

const props = defineProps<{ feedId: string }>()
const emit = defineEmits<{ (e: 'open', payload: { kind: 'topic' | 'person'; id: string }): void }>()
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

const themes = computed(() => signals.value?.dominant_themes ?? [])
const topics = computed(() => signals.value?.top_topics ?? [])
const trending = computed(() => signals.value?.trending_topics ?? [])
const people = computed(() => signals.value?.key_people ?? [])

/**
 * Coverage, not momentum.
 *
 * This band used to open with a bubble cloud sized by `velocity`. That was removed because the
 * number was answering a different question than the band asks. `velocity` on this endpoint is
 * computed corpus-wide (keyed by topic id, with no feed filter), so a topic scores the same on
 * every show that mentions it. The measured effect: the three topics common to ALL shows drew the
 * biggest bubbles, while each show's distinguishing topic — the one that actually says what the
 * show is about — scored 0.0 and rendered as a tiny unlabelled dot. The encoding was inverted
 * against the band's own title. (It also drew an ↑ on values below 1.0, which `trending.ts`
 * classifies as *cooling*.)
 *
 * What the payload can honestly support is coverage: how many of the show's episodes a topic
 * appears in. When every top topic appears in every episode, the true claim is consistency —
 * "this show is reliably about X" — which is a strength, not a fake trend.
 */
const episodeTotal = computed(() => signals.value?.episode_count ?? 0)

/** True when every listed topic appears in every episode — then we can claim it outright. */
const coversEveryEpisode = computed(
  () =>
    episodeTotal.value > 0 &&
    topics.value.length > 0 &&
    topics.value.every((t) => t.episode_count === episodeTotal.value),
)

/** Episodes the most-covered topic appears in — used when coverage is uneven. */
const topicCoverageMax = computed(() =>
  topics.value.reduce((max, t) => Math.max(max, t.episode_count ?? 0), 0),
)

/**
 * Distinctiveness, which is what the band's title actually asks.
 *
 * Coverage alone can't answer "what this show's about": in the validation corpus every show
 * covers "expert interviews" in all four of its episodes, so it ties with the one topic that
 * identifies the show, and the tie breaks alphabetically — one show's signature topic sorted
 * dead last. `lift` (the topic's share of THIS show over its share of the whole corpus, computed
 * server-side in `feed_signals`) breaks that tie on meaning instead: corpus-wide wallpaper lands
 * at 1.0, a signature topic well above it.
 *
 * A topic must clear BOTH gates to be called distinctive:
 *  - `lift >= MIN_LIFT` — meaningfully above the corpus base rate, not noise. Same bar the
 *    trending gate uses for velocity (`_trending_topics(min_velocity=1.5)`).
 *  - present in ≥2 of the show's episodes — one passing mention on a short show can score a high
 *    lift while saying nothing about the show. Same "regulars, not one-offs" rule as
 *    `_recurring_guests`.
 *
 * Topics with an unknown lift (`null` — no corpus base rate) are never promoted; they fall
 * through to the plain list rather than being guessed at.
 */
const MIN_LIFT = 1.5
const MIN_EPISODES_FOR_DISTINCTIVE = 2

const byLiftDesc = (a: { lift: number | null }, b: { lift: number | null }) =>
  (b.lift ?? 0) - (a.lift ?? 0)

const distinctiveTopics = computed(() =>
  topics.value
    .filter(
      (t) =>
        (t.lift ?? 0) >= MIN_LIFT && (t.episode_count ?? 0) >= MIN_EPISODES_FOR_DISTINCTIVE,
    )
    .slice()
    .sort(byLiftDesc),
)

/** Everything else, still most-covered-first (the order the server returns). */
const otherTopics = computed(() => {
  const promoted = new Set(distinctiveTopics.value.map((t) => t.topic_id))
  return topics.value.filter((t) => !promoted.has(t.topic_id))
})

/** "9" not "9.0", "1.8" as-is — the multiplier in the chip's hover explanation. */
function formatLift(lift: number | null): string {
  if (lift === null) return ''
  return Number.isInteger(lift) ? String(lift) : lift.toFixed(1)
}
const hasAny = computed(
  () =>
    themes.value.length > 0 ||
    topics.value.length > 0 ||
    trending.value.length > 0 ||
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
    <h2 class="lp-section mb-3">{{ t('podcast.about') }}</h2>

    <div v-if="themes.length" class="mb-3">
      <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigThemes') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="th in themes"
          :key="th.theme_id"
          type="button"
          data-testid="ps-theme"
          class="lp-theme-chip rounded-full px-2.5 py-1 text-xs font-semibold text-surface-foreground transition disabled:opacity-60"
          :disabled="!th.anchor_topic_id"
          @click="th.anchor_topic_id && emit('open', { kind: 'topic', id: th.anchor_topic_id })"
        >
          {{ th.label }} <span class="opacity-70">· {{ th.topic_count }}</span>
        </button>
      </div>
    </div>

    <!--
      What sets this show apart, ahead of what it merely covers. These chips carry the accent
      because they are the answer to the section's title; the rest is context.
    -->
    <div v-if="distinctiveTopics.length" class="mb-3">
      <h3 class="lp-kicker mb-1.5" data-testid="ps-distinctive-heading">
        {{ t('podcast.sigDistinctive') }}
      </h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="tp in distinctiveTopics"
          :key="tp.topic_id"
          type="button"
          data-testid="ps-distinctive-topic"
          :title="t('podcast.sigDistinctiveHint', { factor: formatLift(tp.lift) })"
          class="rounded-full border border-accent/40 bg-accent/10 px-2.5 py-1 text-xs font-semibold text-surface-foreground transition hover:bg-accent/20"
          @click="emit('open', { kind: 'topic', id: tp.topic_id })"
        >
          {{ tp.label }}
        </button>
      </div>
    </div>

    <div v-if="otherTopics.length" class="mb-3">
      <!--
        The heading states what the data supports. Once the distinctive topics are split out this
        group is the remainder, so it says so plainly; otherwise the coverage claim stands — when
        every topic appears in every episode the claim is consistency, when coverage is uneven we
        say how far it reaches rather than implying uniformity.
      -->
      <h3 class="lp-kicker mb-1.5" data-testid="ps-topics-heading">
        <template v-if="distinctiveTopics.length">{{ t('podcast.sigAlsoCovers') }}</template>
        <template v-else-if="coversEveryEpisode">
          {{ t('podcast.sigCoverageAll') }}
        </template>
        <template v-else-if="topicCoverageMax > 1 && episodeTotal > 0">
          {{ t('podcast.sigCoverageMost', { count: topicCoverageMax, total: episodeTotal }) }}
        </template>
        <template v-else>{{ t('podcast.sigTopics') }}</template>
      </h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="tp in otherTopics"
          :key="tp.topic_id"
          type="button"
          data-testid="ps-topic"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
          @click="emit('open', { kind: 'topic', id: tp.topic_id })"
        >
          {{ tp.label }}
        </button>
      </div>
    </div>

    <div v-if="trending.length" class="mb-3">
      <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigTrending') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="tr in trending"
          :key="tr.topic_id"
          type="button"
          data-testid="ps-trending"
          class="inline-flex items-center gap-1 rounded-full bg-emerald-500/20 px-2.5 py-1 text-xs font-semibold text-emerald-300 transition hover:bg-emerald-500/30"
          @click="emit('open', { kind: 'topic', id: tr.topic_id })"
        >
          {{ tr.label }} <span class="opacity-80">↑ {{ tr.velocity }}×</span>
        </button>
      </div>
    </div>

    <div v-if="people.length">
      <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigPeople') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="p in people"
          :key="p.person_id"
          type="button"
          data-testid="ps-person"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
          @click="emit('open', { kind: 'person', id: p.person_id })"
        >
          {{ p.name }} <span class="text-muted">· {{ p.episode_count }}</span>
        </button>
      </div>
    </div>
  </section>
</template>
