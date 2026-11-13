<script setup lang="ts">
/**
 * PRD-033 FR6.1 — retrieval-grounded topic briefing cards (#888).
 *
 * Each card is grounded in a *direct retrieval query*, not a briefing pack
 * (the MCP briefing-pack form is orthogonal): the digest topic bands are
 * already built by running a semantic search per topic (`run_corpus_search`), so each
 * band's best-scored hit is the top segment + score. A downstream build adds sections to each
 * card (``topicBriefingSections``). Cards rank by retrieval signal and drill into the
 * Topic Entity View rail.
 */
import { computed } from 'vue'
import type { CorpusDigestResponse, CorpusDigestTopicBand } from '../../api/digestApi'
import { topicBriefingSections } from '../../extensions'
import { useSubjectStore } from '../../stores/subject'

const props = defineProps<{
  digest: CorpusDigestResponse | null
}>()

const subject = useSubjectStore()
const sections = topicBriefingSections.list()

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
      Top topics by retrieval signal — each grounded in a live search.
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
        <component
          :is="section.component"
          v-for="section in sections"
          :key="section.id"
          :topic-id="card.topicId"
          :graph-topic-id="card.graphTopicId"
          :label="card.label"
        />
      </article>
    </div>
  </section>
</template>
