<script setup lang="ts">
/**
 * "▶ Moments" — the ONE visible entrance to an episode's Moments reel on list rows (operator
 * 2026-10-10). The operator rejected a boxed chip under every title as heavy and repeated on every
 * episode: this is small amber monospace text on a row that already exists (no box, no count, no
 * extra height). Amber because it is an action — a different kind from the row's heart and queue.
 * The visible text is small; the tap area is padded to 44px with a negative margin, so it does not
 * push the row taller.
 *
 * Offline it works for a downloaded episode (its moments are saved with the download) and is
 * greyed for any other, like every offline action: inert, and saying why.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { useOnline } from '../composables/useOnline'
import { useDownloadsStore } from '../stores/downloads'

const props = defineProps<{ slug: string }>()

const { t } = useI18n()
const { isOnline } = useOnline()
const downloads = useDownloadsStore()
const available = computed(() => isOnline.value || downloads.isDownloaded(props.slug))
</script>

<template>
  <RouterLink
    v-if="available"
    :to="{ name: 'player', params: { slug }, query: { moments: '1' } }"
    class="relative z-30 -my-3 inline-flex min-h-[44px] shrink-0 items-center px-1 font-mono text-[11px] font-bold uppercase tracking-[0.12em] text-accent no-underline"
    :aria-label="t('moments.play_moments')"
    data-testid="moments-link"
    @click.stop
  >
    <span aria-hidden="true">▶ {{ t('moments.door') }}</span>
  </RouterLink>
  <span
    v-else
    class="relative z-30 -my-3 inline-flex min-h-[44px] shrink-0 items-center px-1 font-mono text-[11px] font-bold uppercase tracking-[0.12em] text-disabled"
    role="link"
    aria-disabled="true"
    :aria-label="t('moments.needsConnection')"
    :title="t('moments.needsConnection')"
    data-testid="moments-link-offline"
  >
    <span aria-hidden="true">▶ {{ t('moments.door') }}</span>
  </span>
</template>
