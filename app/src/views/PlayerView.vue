<script setup lang="ts">
/**
 * Player (PRD-039) — scaffold only. Resolves episode detail by slug and renders the masthead
 * + intelligence-zone placeholder. The transcript-sync engine, controls, and adaptive
 * artwork zone land in C4 (#1083); this proves routing + detail wiring.
 */
import { onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getEpisode } from '../services/api'
import type { EpisodeDetail } from '../services/types'

const props = defineProps<{ slug: string }>()
const { t } = useI18n()
const episode = ref<EpisodeDetail | null>(null)
const loading = ref(true)
const notFound = ref(false)

async function load(slug: string): Promise<void> {
  loading.value = true
  notFound.value = false
  try {
    episode.value = await getEpisode(slug)
  } catch {
    notFound.value = true
  } finally {
    loading.value = false
  }
}

onMounted(() => load(props.slug))
watch(() => props.slug, (s) => load(s))
</script>

<template>
  <section>
    <RouterLink :to="{ name: 'catalog' }" class="lp-kicker no-underline">
      ‹ {{ t('player.back') }}
    </RouterLink>

    <p v-if="loading" class="text-muted mt-4">{{ t('player.loading') }}</p>
    <p v-else-if="notFound" class="text-danger mt-4">{{ t('player.notFound') }}</p>

    <div v-else-if="episode" class="mt-3">
      <span class="lp-kicker">{{ episode.podcast_title }}</span>
      <h1 class="font-display text-3xl font-extrabold tracking-tight leading-tight mt-1">
        {{ episode.title }}
      </h1>
      <!-- Player surface (transcript-sync, controls, intelligent artwork zone) — C4 / #1083. -->
      <div class="mt-6 rounded-2xl border border-border bg-surface p-5 text-muted">
        Player surface coming in #1083.
      </div>
    </div>
  </section>
</template>
