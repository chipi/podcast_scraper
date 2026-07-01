<script setup lang="ts">
/**
 * RFC-088 chunk-9 follow-up — surface episode-scope enrichments in the
 * episode detail panel. Reads:
 *   - metadata/enrichments/{stem}.insight_density.json  → early/mid/late bars
 *
 * Per-episode topic co-occurrence was removed from this panel: on a single
 * episode it is just "every pair of this episode's tags" (trivial — the reader
 * already sees the tags). Co-occurrence only earns its keep at *corpus* scope,
 * ranked and surfaced on the Topic node card. This panel now carries only the
 * per-episode-meaningful insight-density signal.
 *
 * Self-loading on mount + on prop change. Hidden when the density envelope is
 * absent (panel stays uncluttered for fresh corpora).
 */
import { computed, onMounted, ref, watch } from 'vue'
import { getEpisodeEnrichmentEnvelope } from '../../api/enrichmentApi'

interface Props {
  corpusPath: string
  metadataRelpath: string
}
const props = defineProps<Props>()
// Reported up so the episode rail can hide the Enrichment tab when this episode
// carries no signals (keeps the tablist honest on fresh / un-enriched corpora).
const emit = defineEmits<{ 'has-content': [boolean] }>()

interface InsightDensityCounts {
  early: number
  mid: number
  late: number
  unknown?: number
}
interface InsightDensityData {
  episode_id?: string
  duration_seconds?: number
  has_timing?: boolean
  counts: InsightDensityCounts
  total_insights: number
}

const density = ref<InsightDensityData | null>(null)
const loaded = ref(false)

const hasAny = computed(() => Boolean(density.value && density.value.total_insights > 0))

const segmentMax = computed(() => {
  const c = density.value?.counts
  if (!c) return 0
  return Math.max(c.early, c.mid, c.late, c.unknown ?? 0)
})

async function load(): Promise<void> {
  const root = props.corpusPath?.trim()
  const rel = props.metadataRelpath?.trim()
  if (!root || !rel) {
    density.value = null
    loaded.value = true
    return
  }
  try {
    const d = await getEpisodeEnrichmentEnvelope<InsightDensityData>(
      root,
      'insight_density',
      rel,
    ).catch(() => null)
    density.value = d?.data ?? null
  } catch {
    /* best-effort */
  } finally {
    loaded.value = true
  }
}

onMounted(load)
watch(() => [props.corpusPath, props.metadataRelpath], () => void load(), { deep: false })
watch(
  () => loaded.value && Boolean(hasAny.value),
  (has) => emit('has-content', has),
  { immediate: true },
)

function bar(segment: 'early' | 'mid' | 'late' | 'unknown'): number {
  if (!density.value || segmentMax.value === 0) return 0
  return density.value.counts[segment] ?? 0
}

function widthPct(segment: 'early' | 'mid' | 'late' | 'unknown'): string {
  if (segmentMax.value === 0) return '0%'
  return `${Math.round((bar(segment) / segmentMax.value) * 100)}%`
}
</script>

<template>
  <section
    v-if="loaded && hasAny"
    class="mt-2 rounded border border-default bg-overlay/40 p-2"
    aria-label="Episode enrichment signals"
    data-testid="episode-enrichment-section"
  >
    <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
      Enrichment signals
    </h3>
    <!-- Insight density bars: early / mid / late count of insights -->
    <div
      v-if="density && density.total_insights > 0"
      class="mb-2"
      data-testid="episode-enrichment-density"
    >
      <p class="mb-1 text-[10px] text-muted">
        Insight density · {{ density.total_insights }} insights
        <span v-if="density.has_timing === false">(no timing — even split)</span>
      </p>
      <div class="flex flex-col gap-0.5 text-[10px]">
        <div
          v-for="seg in (['early', 'mid', 'late'] as const)"
          :key="seg"
          class="flex items-center gap-2"
        >
          <span class="w-10 text-muted">{{ seg }}</span>
          <div class="relative h-2 flex-1 rounded bg-overlay">
            <div
              class="absolute left-0 top-0 h-2 rounded bg-emerald-500/40"
              :style="{ width: widthPct(seg) }"
            />
          </div>
          <span class="w-6 text-right font-mono">{{ bar(seg) }}</span>
        </div>
      </div>
    </div>

  </section>
</template>
