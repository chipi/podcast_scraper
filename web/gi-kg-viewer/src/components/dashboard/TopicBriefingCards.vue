<script setup lang="ts">
/**
 * PRD-033 FR6.1 — retrieval-grounded topic briefing cards (#888).
 *
 * Each card is grounded in a *direct retrieval query*, not a briefing pack
 * (the MCP briefing-pack form is orthogonal, RFC-093): the digest topic bands are
 * already built by running a semantic search per topic (`run_corpus_search`), so each
 * band's best-scored hit is the top segment + score. For topics that map to a KG node
 * we additionally pull the top cross-show insight from the relational layer
 * (`cross_show_synthesis`). Cards rank by retrieval signal and drill into the
 * Topic Entity View rail.
 */
import { computed, ref, watch } from 'vue'
import type { CorpusDigestResponse, CorpusDigestTopicBand } from '../../api/digestApi'
import { fetchCrossShow } from '../../api/relationalApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

const props = defineProps<{
  digest: CorpusDigestResponse | null
}>()

const shell = useShellStore()
const subject = useSubjectStore()

const CARD_COUNT = 4

/** Retrieval signal for a band: best hit score lifted by density + distinct-show breadth. */
function bandSignal(band: CorpusDigestTopicBand): number {
  const hits = band.hits ?? []
  if (!hits.length) return 0
  const top = Math.max(...hits.map((h) => h.score ?? 0))
  const distinctShows = new Set(
    hits.map((h) => h.feed_id).filter((f): f is string => Boolean(f)),
  ).size
  return top * (1 + 0.25 * Math.log2(1 + hits.length) + 0.5 * Math.log2(1 + distinctShows))
}

interface BriefingCard {
  topicId: string
  graphTopicId: string
  label: string
  episodeCount: number
  topSegment: string
  topScore: number
}

const cards = computed<BriefingCard[]>(() => {
  const bands = props.digest?.topics ?? []
  return bands
    .filter((b) => (b.hits?.length ?? 0) > 0)
    .map((b) => ({ b, s: bandSignal(b) }))
    .sort((x, y) => y.s - x.s)
    .slice(0, CARD_COUNT)
    .map(({ b }) => {
      const best = [...(b.hits ?? [])].sort((h1, h2) => (h2.score ?? 0) - (h1.score ?? 0))[0]
      return {
        topicId: b.topic_id,
        graphTopicId: b.graph_topic_id?.trim() ?? '',
        label: b.label,
        episodeCount: b.hits?.length ?? 0,
        topSegment: best?.summary_preview?.trim() || best?.episode_title?.trim() || '',
        topScore: best?.score ?? 0,
      }
    })
})

/** Lazily-fetched top cross-show insight per mapped topic (the relational differentiator). */
const crossShow = ref<Record<string, { shows: number; insight: string } | null>>({})

async function loadCrossShow(card: BriefingCard): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!card.graphTopicId || !root || !shell.healthStatus) return
  if (card.topicId in crossShow.value) return
  crossShow.value = { ...crossShow.value, [card.topicId]: null }
  try {
    const body = await fetchCrossShow(root, card.graphTopicId)
    const groups = Object.values(body.groups ?? {})
    const first = groups.find((g) => g.length > 0)?.[0]
    crossShow.value = {
      ...crossShow.value,
      [card.topicId]: first ? { shows: groups.length, insight: first.text } : null,
    }
  } catch {
    crossShow.value = { ...crossShow.value, [card.topicId]: null }
  }
}

watch(
  cards,
  (list) => {
    for (const card of list) void loadCrossShow(card)
  },
  { immediate: true },
)

function openTopic(card: BriefingCard): void {
  if (card.graphTopicId) subject.focusTopic(card.graphTopicId)
}
</script>

<template>
  <section
    v-if="cards.length"
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="topic-briefing-cards"
    aria-label="Topic briefing cards"
  >
    <h3 class="mb-2 text-sm font-semibold">
      Topic briefings
    </h3>
    <p class="mb-2 text-[11px] text-muted">
      Top topics by retrieval signal — each grounded in a live search; mapped topics add the
      cross-show synthesis.
    </p>
    <div class="grid gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,13rem),1fr))]">
      <article
        v-for="card in cards"
        :key="card.topicId"
        data-testid="topic-briefing-card"
        class="rounded border border-border bg-elevated/40 p-2 text-xs"
      >
        <div class="flex items-baseline justify-between gap-2">
          <button
            v-if="card.graphTopicId"
            type="button"
            data-testid="topic-briefing-card-link"
            class="min-w-0 flex-1 truncate rounded text-left font-semibold text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            :title="`Open Topic panel for ${card.label}`"
            @click="openTopic(card)"
          >{{ card.label }}</button>
          <span
            v-else
            class="min-w-0 flex-1 truncate font-semibold text-surface-foreground"
          >{{ card.label }}</span>
          <span class="shrink-0 text-[10px] text-muted">{{ card.episodeCount }} ep</span>
        </div>
        <p
          v-if="card.topSegment"
          class="mt-1 line-clamp-2 text-[11px] leading-snug text-muted"
          :title="card.topSegment"
        >
          <span
            class="mr-1 rounded bg-primary/15 px-1 py-px font-mono text-[9px] text-primary"
          >{{ card.topScore.toFixed(2) }}</span>
          {{ card.topSegment }}
        </p>
        <p
          v-if="crossShow[card.topicId]"
          data-testid="topic-briefing-card-cross-show"
          class="mt-1 border-l-2 border-primary/40 pl-1.5 text-[10px] leading-snug text-surface-foreground/80"
          :title="crossShow[card.topicId]?.insight"
        >
          <span class="font-medium text-primary">Across {{ crossShow[card.topicId]?.shows }} shows:</span>
          <span class="line-clamp-2">{{ crossShow[card.topicId]?.insight }}</span>
        </p>
      </article>
    </div>
  </section>
</template>
