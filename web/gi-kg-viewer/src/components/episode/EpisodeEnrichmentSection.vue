<script setup lang="ts">
/**
 * RFC-088 chunk-9 follow-up — surface episode-scope enrichments in the
 * episode detail panel. Reads:
 *   - metadata/enrichments/{stem}.insight_density.json  → early/mid/late bars
 *   - metadata/enrichments/{stem}.topic_cooccurrence.json → per-episode topic pairs
 *
 * Self-loading on mount + on prop change. Hidden when neither envelope
 * exists (panel stays uncluttered for fresh corpora).
 */
import { computed, onMounted, ref, watch } from 'vue'
import { getEpisodeEnrichmentEnvelope } from '../../api/enrichmentApi'
import { useSubjectStore } from '../../stores/subject'

interface Props {
  corpusPath: string
  metadataRelpath: string
}
const props = defineProps<Props>()

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

interface TopicCooccurrencePair {
  topic_a_id: string
  topic_b_id: string
  topic_a_label?: string
  topic_b_label?: string
  episode_count: number
}
interface TopicCooccurrenceData {
  episode_id?: string
  pairs: TopicCooccurrencePair[]
}

const density = ref<InsightDensityData | null>(null)
const cooccurrence = ref<TopicCooccurrenceData | null>(null)
const loaded = ref(false)

const subject = useSubjectStore()

const hasAny = computed(
  () =>
    (density.value && density.value.total_insights > 0) ||
    (cooccurrence.value && cooccurrence.value.pairs.length > 0),
)

const segmentMax = computed(() => {
  const c = density.value?.counts
  if (!c) return 0
  return Math.max(c.early, c.mid, c.late, c.unknown ?? 0)
})

const COOC_TOP_N = 8
const cooccurrenceChips = computed<TopicCooccurrencePair[]>(() => {
  const pairs = cooccurrence.value?.pairs ?? []
  // Per-episode pairs all have episode_count=1; sort by topic_a then b for stable order.
  return [...pairs]
    .sort((a, b) => a.topic_a_id.localeCompare(b.topic_a_id) || a.topic_b_id.localeCompare(b.topic_b_id))
    .slice(0, COOC_TOP_N)
})

async function load(): Promise<void> {
  const root = props.corpusPath?.trim()
  const rel = props.metadataRelpath?.trim()
  if (!root || !rel) {
    density.value = null
    cooccurrence.value = null
    loaded.value = true
    return
  }
  try {
    const [d, c] = await Promise.all([
      getEpisodeEnrichmentEnvelope<InsightDensityData>(root, 'insight_density', rel).catch(
        () => null,
      ),
      getEpisodeEnrichmentEnvelope<TopicCooccurrenceData>(
        root,
        'topic_cooccurrence',
        rel,
      ).catch(() => null),
    ])
    density.value = d?.data ?? null
    cooccurrence.value = c?.data ?? null
  } catch {
    /* best-effort */
  } finally {
    loaded.value = true
  }
}

onMounted(load)
watch(() => [props.corpusPath, props.metadataRelpath], () => void load(), { deep: false })

/**
 * Click on either endpoint button of a pair focuses the OTHER endpoint
 * (the partner). ``clicked`` names which endpoint the user clicked;
 * we return the partner's id.
 */
function focusPartner(p: TopicCooccurrencePair, clicked: 'a' | 'b'): void {
  const id = clicked === 'a' ? p.topic_b_id : p.topic_a_id
  if (id) subject.focusTopic(id)
}

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

    <!-- Per-episode topic co-occurrence chips -->
    <div v-if="cooccurrenceChips.length" data-testid="episode-enrichment-cooccurrence">
      <p class="mb-1 text-[10px] text-muted">Topic pairs in this episode</p>
      <div class="flex flex-wrap gap-1">
        <span
          v-for="p in cooccurrenceChips"
          :key="`${p.topic_a_id}::${p.topic_b_id}`"
          class="flex items-center gap-0.5 rounded border border-default bg-overlay px-2 py-0.5 text-[10px]"
          :data-testid="`episode-enrichment-cooccurrence-${p.topic_a_id}--${p.topic_b_id}`"
        >
          <button
            type="button"
            class="hover:underline"
            :title="`Focus partner ${p.topic_b_label || p.topic_b_id}`"
            @click="focusPartner(p, 'a')"
          >{{ p.topic_a_label || p.topic_a_id }}</button>
          <span class="text-muted">↔</span>
          <button
            type="button"
            class="hover:underline"
            :title="`Focus partner ${p.topic_a_label || p.topic_a_id}`"
            @click="focusPartner(p, 'b')"
          >{{ p.topic_b_label || p.topic_b_id }}</button>
        </span>
      </div>
    </div>
  </section>
</template>
