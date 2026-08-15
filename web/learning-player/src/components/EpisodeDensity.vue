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
import SectionStatus from './SectionStatus.vue'
import { useSectionState } from '../composables/useSectionState'
import { getEpisodeEnrichment } from '../services/api'

const props = defineProps<{ slug: string }>()
const emit = defineEmits<{ (e: 'seek', seconds: number): void }>()
const { t } = useI18n()

const SEGMENTS = ['early', 'mid', 'late'] as const
type Seg = (typeof SEGMENTS)[number]

type Density = { early: number; mid: number; late: number; durationSeconds: number }

/**
 * `useSectionState` so "the enrichment call failed" is not rendered identically to "this episode
 * has no insight density" — #1591's defect, which was fixed for the Home sections and never
 * carried to the Knowledge Panel's.
 */
const section = useSectionState<Density | null>(null)
const density = computed(() => section.data.value)
/** Drops a slow reply for an episode the reader has already left. */
const requestSeq = ref(0)

async function load(): Promise<void> {
  const mine = requestSeq.value + 1
  requestSeq.value = mine
  await section.load(async () => {
    const s = await getEpisodeEnrichment(props.slug)
    if (mine !== requestSeq.value) throw new Error('superseded')
    const d = s.insight_density
    const c = d?.counts
    if (!c) return null
    const early = c.early ?? 0
    const mid = c.mid ?? 0
    const late = c.late ?? 0
    // No marked insights is a real, successful answer — a ready empty, never an error.
    if (early + mid + late === 0) return null
    return { early, mid, late, durationSeconds: d?.duration_seconds ?? 0 }
  })
}

watch(() => props.slug, () => void load(), { immediate: true })

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
  <!-- A failed enrichment call says so; an episode with no marked insights still renders nothing. -->
  <SectionStatus v-if="section.isError.value" :phase="section.phase.value" @retry="load()" />

  <div
    v-else-if="density"
    class="mb-3 rounded-xl border border-border p-3"
    data-testid="episode-density"
  >
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
