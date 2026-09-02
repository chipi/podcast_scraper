<script setup lang="ts">
/**
 * The Downloaded surface (#1905), rendered from the device registry with ZERO API calls — this is
 * the one list that has to be correct when there is no network, so it cannot depend on one.
 *
 * That is why the registry carries title/show/duration/artwork: offline the API is unreachable
 * exactly when this renders.
 *
 * Native only. Lives inside the Saved tab rather than taking a sixth tab slot, following
 * LibraryView's own note that the strip is capped and new kinds fold in here.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import DownloadButton from './DownloadButton.vue'
import { localArtworkFor } from '../services/downloads'
import { isNative } from '../services/native'
import { useDownloadsStore } from '../stores/downloads'

const { t } = useI18n()
const downloads = useDownloadsStore()
const native = isNative()

/** Newest first — the thing you just downloaded is the thing you are looking for. */
const items = computed(() =>
  Object.values(downloads.entries).sort((a, b) => b.updatedAt - a.updatedAt),
)

function minutes(seconds?: number): string | null {
  if (!seconds) return null
  return `${Math.max(1, Math.round(seconds / 60))} min`
}

const usedMb = computed(() => (downloads.bytesOnDisk / (1024 * 1024)).toFixed(0))
</script>

<template>
  <!-- Hidden entirely until something is downloaded: an empty block on every visit to Saved is
       clutter for the majority who have none. The control that creates the first one lives on the
       episode cards themselves, so discovery does not depend on this section existing. -->
  <section v-if="native && items.length" data-testid="downloaded-section" class="mb-6">
    <h2 class="lp-section mb-2">{{ t('downloads.title') }}</h2>

    <ul class="flex flex-col">
      <li
        v-for="e in items"
        :key="e.slug"
        data-testid="downloaded-item"
        class="flex items-center gap-3 border-b border-border py-3"
      >
        <img
          v-if="localArtworkFor(e.slug)"
          :src="localArtworkFor(e.slug) ?? undefined"
          alt=""
          class="h-10 w-10 shrink-0 rounded object-cover"
        />
        <RouterLink
          :to="{ name: 'player', params: { slug: e.slug } }"
          class="min-w-0 flex-1 no-underline text-canvas-foreground"
        >
          <p class="truncate text-sm font-semibold leading-snug">{{ e.title ?? e.slug }}</p>
          <p class="lp-kicker mt-1 truncate">
            <span v-if="e.showTitle">{{ e.showTitle }}</span>
            <template v-if="e.showTitle && minutes(e.durationSeconds)"> · </template>
            <span v-if="minutes(e.durationSeconds)">{{ minutes(e.durationSeconds) }}</span>
            <!-- The non-downloaded states have to be legible here too, or a queued episode looks
                 like a downloaded one that will not play. -->
            <template v-if="e.state !== 'downloaded'"> · </template>
            <span v-if="e.state === 'queued'" class="text-muted">{{
              t('downloads.waitingWifi')
            }}</span>
            <span v-else-if="e.state === 'downloading'" class="text-accent">{{
              t('downloads.downloading')
            }}</span>
            <span v-else-if="e.state === 'failed'" class="text-muted">{{
              e.errorKind === 'permanent'
                ? t('downloads.unavailable')
                : e.errorKind === 'needs-space'
                  ? t('downloads.needsSpace')
                  : t('downloads.retry')
            }}</span>
          </p>
        </RouterLink>
        <DownloadButton :slug="e.slug" />
      </li>
    </ul>

    <p data-testid="downloaded-storage" class="lp-kicker mt-3">
      {{ t('downloads.storageUsed') }}: {{ usedMb }} MB ·
      {{ t('downloads.episodeCount', { n: downloads.downloadedCount }) }}
    </p>
  </section>
</template>
