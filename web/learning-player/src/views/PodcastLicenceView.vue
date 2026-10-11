<script setup lang="ts">
/**
 * The full text of what a show's feed declares about licence and copyright (#2325). Reached from
 * the one-line summary under the show artwork. Every declaration is shown exactly as the publisher
 * wrote it, with where it sits (whole show or one episode), what it applies to, the documents it
 * links, any conflict between declarations, when it was read, and earlier versions when the
 * publisher changed it. A record of the declaration, not a licence grant.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import BackIcon from '../components/BackIcon.vue'
import LicenceDeclarationCard from '../components/LicenceDeclarationCard.vue'
import { getEpisodeLicence, getPodcastLicence } from '../services/api'
import type { EpisodeLicence, PodcastLicence } from '../services/types'
import { formatPublishDate } from '../utils/format'

// `episode`: opened from that episode's About (#2325) -- the same page, plus the episode's own
// declarations at the top when it has any (Podcasting 2.0 lets an item declare its own licence).
const props = defineProps<{ feedId: string; episode?: string | null }>()
const { t, locale } = useI18n()
const router = useRouter()

const data = ref<PodcastLicence | null>(null)
const episodeData = ref<EpisodeLicence | null>(null)
const error = ref(false)

async function load(): Promise<void> {
  error.value = false
  data.value = null
  episodeData.value = null
  try {
    const [show, episode] = await Promise.all([
      getPodcastLicence(props.feedId),
      props.episode ? getEpisodeLicence(props.episode).catch(() => null) : Promise.resolve(null),
    ])
    data.value = show
    episodeData.value = episode
  } catch {
    error.value = true
  }
}
onMounted(load)
watch(() => [props.feedId, props.episode], load)

/** The episode's own record, only when it declares something itself. */
const episodeOwn = computed(() =>
  episodeData.value?.episode?.status === 'declared' ? episodeData.value.episode : null,
)

const cameFromApp = typeof window !== 'undefined' && Boolean(window.history.state?.back)
function goBack(e: MouseEvent): void {
  if (!cameFromApp) return
  e.preventDefault()
  router.back()
}

const current = computed(() => data.value?.licence?.declarations ?? [])
// The newest version is the current one; the rest are what the feed declared before.
const earlier = computed(() => (data.value?.history ?? []).slice(1))

function date(iso: string | null | undefined): string {
  return (iso && formatPublishDate(iso, locale.value)) || ''
}
</script>

<template>
  <section class="lp-page lp-focus pb-8 pt-4" data-testid="podcast-licence-page">
    <RouterLink
      :to="{ name: 'podcast', params: { feedId } }"
      class="mb-4 inline-flex items-center gap-1 text-sm font-medium text-muted no-underline transition hover:text-canvas-foreground"
      data-testid="podcast-licence-back"
      @click="goBack"
    >
      <BackIcon /> {{ data?.title || t('licence.back') }}
    </RouterLink>
    <h1 class="mb-2 font-display text-3xl font-extrabold tracking-tight">{{ t('licence.title') }}</h1>
    <p class="mb-6 text-sm text-muted">{{ t('licence.disclaimer') }}</p>

    <p v-if="error" class="text-sm text-muted" data-testid="podcast-licence-error">
      {{ t('licence.loadError') }}
    </p>

    <template v-else-if="data">
      <div v-if="episodeOwn" class="mb-3" data-testid="podcast-licence-episode">
        <h2 class="mb-2 text-lg font-bold">
          {{ t('licence.thisEpisode', { title: episodeData?.title ?? '' }) }}
        </h2>
        <p class="mb-3 text-sm text-muted">{{ t('licence.episodeOwnApplies') }}</p>
        <LicenceDeclarationCard v-for="(d, i) in episodeOwn.declarations" :key="i" :declaration="d" />
        <ul v-if="episodeOwn.conflicts.length" class="mb-4 list-disc pl-5 text-sm">
          <li v-for="c in episodeOwn.conflicts" :key="c">{{ c }}</li>
        </ul>
        <h2 class="mt-6 text-lg font-bold">{{ t('licence.theShow') }}</h2>
      </div>
      <p v-if="data.status === 'not_checked'" class="mb-4 text-sm" data-testid="podcast-licence-not-checked">
        {{ t('licence.notChecked') }}
      </p>
      <p
        v-else-if="data.status === 'no_explicit_licence_found'"
        class="mb-4 text-sm"
        data-testid="podcast-licence-none"
      >
        {{ t('licence.none') }}
      </p>

      <template v-else>
        <h2 class="mb-3 text-lg font-bold">{{ t('licence.current') }}</h2>
        <LicenceDeclarationCard v-for="(d, i) in current" :key="i" :declaration="d" />

        <div v-if="data.licence?.conflicts.length" class="mb-4" data-testid="podcast-licence-conflicts">
          <h2 class="mb-2 text-lg font-bold">{{ t('licence.conflicts') }}</h2>
          <ul class="list-disc pl-5 text-sm">
            <li v-for="c in data.licence.conflicts" :key="c">{{ c }}</li>
          </ul>
        </div>
      </template>

      <div v-if="data.licence?.documents_for_review.length" class="mb-4" data-testid="podcast-licence-documents">
        <h2 class="mb-2 text-lg font-bold">{{ t('licence.documents') }}</h2>
        <ul class="list-disc pl-5 text-sm">
          <li v-for="u in data.licence.documents_for_review" :key="u">
            <a :href="u" target="_blank" rel="noopener noreferrer" class="break-all underline">{{ u }}</a>
          </li>
        </ul>
      </div>

      <p v-if="data.licence?.retrieved_at" class="mb-6 break-words text-xs text-muted" data-testid="podcast-licence-read">
        {{ t('licence.readOn', { date: date(data.licence.retrieved_at), source: data.licence.source_feed_url ?? '' }) }}
      </p>

      <div v-if="earlier.length" data-testid="podcast-licence-history">
        <h2 class="mb-3 text-lg font-bold">{{ t('licence.earlier') }}</h2>
        <article v-for="(v, i) in earlier" :key="i" class="mb-4 rounded-xl border border-elevated p-4">
          <p class="mb-2 text-xs text-muted">
            {{ t('licence.readRange', { first: date(v.first_read), last: date(v.last_read) }) }}
          </p>
          <p v-if="!v.declarations.length" class="text-sm">{{ t('licence.none') }}</p>
          <div v-for="(d, j) in v.declarations" :key="j" class="mb-2">
            <p class="text-xs font-semibold uppercase tracking-wide text-muted">
              {{ t(`licence.mechanism.${d.mechanism}`) }}
            </p>
            <p class="whitespace-pre-wrap break-words text-sm">{{ d.raw }}</p>
          </div>
        </article>
      </div>
    </template>
  </section>
</template>
