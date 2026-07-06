<script setup lang="ts">
/**
 * Episode insight-density (Plan B — RFC-088 `insight_density` on the player).
 * A compact early/mid/late bar strip at the head of the Insights list: "where
 * the substance sits." Fed by /api/app/episodes/{slug}/enrichment (cached).
 * Each third is tap-to-seek — jump to the dense part. Hides when the enricher
 * didn't run or the episode has no insights.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getEpisodeEnrichment } from '../services/api'

const props = defineProps<{ slug: string }>()
const emit = defineEmits<{ (e: 'seek', seconds: number): void }>()
const { t } = useI18n()

const SEGMENTS = ['early', 'mid', 'late'] as const
type Seg = (typeof SEGMENTS)[number]

const density = ref<{ early: number; mid: number; late: number; durationSeconds: number } | null>(null)

watch(
  () => props.slug,
  () => {
    density.value = null
    void getEpisodeEnrichment(props.slug)
      .then((s) => {
        const d = s.insight_density
        const c = d?.counts
        if (!c) return
        const early = c.early ?? 0
        const mid = c.mid ?? 0
        const late = c.late ?? 0
        if (early + mid + late > 0) {
          density.value = { early, mid, late, durationSeconds: d?.duration_seconds ?? 0 }
        }
      })
      .catch(() => {
        density.value = null
      })
  },
  { immediate: true },
)

const max = computed(() =>
  density.value ? Math.max(density.value.early, density.value.mid, density.value.late, 1) : 1,
)
const peak = computed<Seg | null>(() => {
  const d = density.value
  if (!d) return null
  return SEGMENTS.reduce((a, b) => (d[b] > d[a] ? b : a))
})
/** Bar height in px (max ~36), min 4 so an empty third still reads as a stub. */
function barPx(n: number): number {
  return Math.max(4, Math.round((n / max.value) * 36))
}
/** Seek to the start of a third when the episode duration is known. */
function seekTo(seg: Seg): void {
  const dur = density.value?.durationSeconds ?? 0
  if (!dur) return
  const frac = seg === 'early' ? 0 : seg === 'mid' ? 1 / 3 : 2 / 3
  emit('seek', Math.floor(dur * frac))
}
</script>

<template>
  <div v-if="density" class="mb-3 rounded-xl border border-border p-3" data-testid="episode-density">
    <p class="lp-kicker mb-2">{{ t('kp.density') }}</p>
    <div class="flex items-end gap-2">
      <button
        v-for="seg in SEGMENTS"
        :key="seg"
        type="button"
        class="flex flex-1 flex-col items-center gap-1 rounded transition hover:opacity-80 disabled:cursor-default disabled:hover:opacity-100"
        :disabled="!density.durationSeconds"
        :data-testid="`density-${seg}`"
        :aria-label="t('kp.densitySeek', { third: t(`kp.density_${seg}`), count: density[seg] })"
        @click="seekTo(seg)"
      >
        <span
          class="w-full rounded-t bg-accent/70"
          :class="peak === seg ? 'bg-accent' : ''"
          :style="{ height: `${barPx(density[seg])}px` }"
        />
        <span class="text-[10px] text-muted">{{ t(`kp.density_${seg}`) }}</span>
        <span class="text-[10px] font-semibold tabular-nums">{{ density[seg] }}</span>
      </button>
    </div>
    <p v-if="peak" class="mt-2 text-xs text-muted" data-testid="density-peak">
      {{ t('kp.densityPeak', { third: t(`kp.density_${peak}`) }) }}
    </p>
  </div>
</template>
