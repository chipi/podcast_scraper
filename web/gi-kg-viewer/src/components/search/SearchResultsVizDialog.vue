<script setup lang="ts">
import { computed, ref } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import {
  computeScoreStats,
  docTypeDistribution,
  episodeDistribution,
  episodeRowTooltip,
  feedDistribution,
  feedRowTooltip,
  insightDominantDocType,
  insightTimeline,
  insightTopTerm,
  publishMonthTimeline,
  scoreBarsForHits,
  topTermsFromHits,
} from '../../utils/searchResultsViz'

const props = defineProps<{
  hits: SearchHit[]
}>()

const dialogRef = ref<HTMLDialogElement | null>(null)

const docTypes = computed(() => docTypeDistribution(props.hits))
const episodes = computed(() => episodeDistribution(props.hits))
const feeds = computed(() => feedDistribution(props.hits))
const scoreStats = computed(() => computeScoreStats(props.hits))
const scoreBars = computed(() => scoreBarsForHits(props.hits))
const timeline = computed(() => publishMonthTimeline(props.hits))
const terms = computed(() => topTermsFromHits(props.hits))
const maxTermCount = computed(() =>
  terms.value.length ? Math.max(...terms.value.map((t) => t.count), 1) : 1,
)
const maxMonthCount = computed(() =>
  timeline.value.buckets.length
    ? Math.max(...timeline.value.buckets.map((b) => b.count), 1)
    : 1,
)

const headlineInsight = computed(() =>
  insightDominantDocType(docTypes.value, props.hits.length),
)
const timeInsight = computed(() => insightTimeline(timeline.value, props.hits.length))
const termsInsight = computed(() => insightTopTerm(terms.value))

function onBackdropClick(e: MouseEvent): void {
  const el = dialogRef.value
  if (el && e.target === el) {
    el.close()
  }
}

function open(): void {
  dialogRef.value?.showModal()
}

function close(): void {
  dialogRef.value?.close()
}

defineExpose({ open, close })
</script>

<template>
  <dialog
    ref="dialogRef"
    class="w-[min(100%,38rem)] max-h-[min(90vh,42rem)] overflow-y-auto rounded-lg border border-border bg-surface p-4 text-surface-foreground shadow-xl [&::backdrop]:bg-black/40"
    aria-labelledby="search-results-viz-title"
    @click="onBackdropClick"
  >
    <div class="mb-3 flex items-start justify-between gap-3">
      <div class="min-w-0">
        <h2 id="search-results-viz-title" class="text-sm font-medium text-surface-foreground">
          Search result insights
        </h2>
        <p class="mt-0.5 text-[11px] text-muted">
          {{ hits.length }} {{ hits.length === 1 ? 'hit' : 'hits' }}
        </p>
        <p
          v-if="headlineInsight"
          class="mt-1.5 text-[11px] font-medium leading-snug text-elevated-foreground"
        >
          {{ headlineInsight }}
        </p>
      </div>
      <button
        type="button"
        class="shrink-0 rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
        @click="close"
      >
        Close
      </button>
    </div>

    <!-- Small multiples: doc types + time side-by-side (no tabs). -->
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <section
        aria-labelledby="sr-viz-doc-types-heading"
        role="region"
        class="text-xs"
      >
        <h3
          id="sr-viz-doc-types-heading"
          class="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted"
        >
          Doc types
        </h3>
        <ul class="space-y-1.5">
          <li
            v-for="row in docTypes"
            :key="row.key"
            class="flex items-center gap-2"
          >
            <span class="w-[7.5rem] shrink-0 truncate text-muted" :title="row.label">{{ row.label }}</span>
            <div class="h-2 min-w-0 flex-1 overflow-hidden bg-overlay">
              <div
                class="h-full bg-primary"
                :style="{ width: `${row.pct}%` }"
              />
            </div>
            <span class="w-8 shrink-0 text-right font-mono text-[10px] text-muted">{{ row.count }}</span>
          </li>
        </ul>
      </section>

      <section
        aria-labelledby="sr-viz-time-heading"
        role="region"
        class="text-xs"
      >
        <h3
          id="sr-viz-time-heading"
          class="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted"
        >
          Publish month
        </h3>
        <p
          v-if="timeInsight"
          class="mb-1.5 text-[11px] leading-snug text-elevated-foreground"
        >
          {{ timeInsight }}
        </p>
        <p class="mb-1.5 text-[10px] leading-snug text-muted">
          From index <code class="rounded bg-canvas px-0.5">publish_date</code> (monthly buckets).
        </p>
        <ul
          v-if="timeline.buckets.length"
          class="space-y-2"
        >
          <li
            v-for="b in timeline.buckets"
            :key="b.label"
            class="flex items-center gap-2"
          >
            <span class="w-16 shrink-0 font-mono text-[10px] text-muted">{{ b.label }}</span>
            <div class="h-2.5 min-w-0 flex-1 overflow-hidden bg-overlay">
              <div
                class="h-full bg-primary"
                :style="{ width: `${(b.count / maxMonthCount) * 100}%` }"
              />
            </div>
            <span class="w-6 shrink-0 text-right font-mono text-[10px] text-muted">{{ b.count }}</span>
          </li>
        </ul>
        <p
          v-else-if="!timeInsight"
          class="text-[11px] text-muted"
        >
          No month buckets.
        </p>
      </section>
    </div>

    <section
      aria-labelledby="sr-viz-episodes-heading"
      role="region"
      class="mt-4 text-xs"
    >
      <h3
        id="sr-viz-episodes-heading"
        class="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted"
      >
        Episodes (top)
      </h3>
      <ul class="space-y-1.5">
        <li
          v-for="row in episodes.rows"
          :key="row.key"
          class="flex items-center gap-2"
        >
          <span
            class="w-[11rem] shrink-0 truncate text-[10px] text-muted sm:w-[13rem]"
            :title="episodeRowTooltip(row)"
          >{{ row.label }}</span>
          <div class="h-2 min-w-0 flex-1 overflow-hidden bg-overlay">
            <div
              class="h-full bg-primary"
              :style="{ width: `${row.pct}%` }"
            />
          </div>
          <span class="w-8 shrink-0 text-right font-mono text-[10px] text-muted">{{ row.count }}</span>
        </li>
      </ul>
      <p
        v-if="episodes.tailDistinct > 0"
        class="mt-1.5 text-[10px] text-muted"
      >
        +{{ episodes.tailDistinct }} other episode(s), {{ episodes.tailHitCount }} hit(s) not listed above.
      </p>
    </section>

    <section
      aria-labelledby="sr-viz-feeds-heading"
      role="region"
      class="mt-4 text-xs"
    >
      <h3
        id="sr-viz-feeds-heading"
        class="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted"
      >
        Feeds (top)
      </h3>
      <ul class="space-y-1.5">
        <li
          v-for="row in feeds.rows"
          :key="row.key"
          class="flex items-center gap-2"
        >
          <span
            class="w-[11rem] shrink-0 truncate text-[10px] text-muted sm:w-[13rem]"
            :title="feedRowTooltip(row)"
          >{{ row.label }}</span>
          <div class="h-2 min-w-0 flex-1 overflow-hidden bg-overlay">
            <div
              class="h-full bg-primary"
              :style="{ width: `${row.pct}%` }"
            />
          </div>
          <span class="w-8 shrink-0 text-right font-mono text-[10px] text-muted">{{ row.count }}</span>
        </li>
      </ul>
      <p
        v-if="feeds.tailDistinct > 0"
        class="mt-1.5 text-[10px] text-muted"
      >
        +{{ feeds.tailDistinct }} other feed(s), {{ feeds.tailHitCount }} hit(s) not listed above.
      </p>
    </section>

    <section
      aria-labelledby="sr-viz-scores-heading"
      role="region"
      class="mt-4 text-xs"
    >
      <h3
        id="sr-viz-scores-heading"
        class="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted"
      >
        Similarity scores
      </h3>
      <p class="mb-2 text-[10px] leading-snug text-muted">
        Bar length is proportional to each score divided by the
        <span class="font-medium text-elevated-foreground">strongest hit in this list</span>
        (same numeric scale as the index; not min–max stretched).
      </p>
      <p
        v-if="scoreStats"
        class="mb-2 font-mono text-[10px] leading-relaxed text-muted"
      >
        min {{ scoreStats.min.toFixed(4) }} · max {{ scoreStats.max.toFixed(4) }} · mean
        {{ scoreStats.mean.toFixed(4) }} · spread {{ scoreStats.spread.toFixed(4) }}
      </p>
      <ul class="max-h-48 space-y-1 overflow-y-auto pr-1">
        <li
          v-for="b in scoreBars"
          :key="b.rank"
          class="flex items-center gap-2"
        >
          <span class="w-5 shrink-0 font-mono text-[10px] text-muted">#{{ b.rank }}</span>
          <span class="w-14 shrink-0 truncate font-mono text-[9px] text-primary">{{ b.docType }}</span>
          <div class="h-2 min-w-0 flex-1 overflow-hidden bg-overlay">
            <div
              class="h-full bg-primary"
              :style="{ width: `${b.widthPct}%` }"
            />
          </div>
          <span class="w-12 shrink-0 text-right font-mono text-[10px] text-muted">{{
            b.score.toFixed(3)
          }}</span>
        </li>
      </ul>
    </section>

    <section
      aria-labelledby="sr-viz-terms-heading"
      role="region"
      class="mt-4 text-xs"
    >
      <h3
        id="sr-viz-terms-heading"
        class="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted"
      >
        Terms
      </h3>
      <p
        v-if="termsInsight"
        class="mb-1.5 text-[11px] leading-snug text-elevated-foreground"
      >
        {{ termsInsight }}
      </p>
      <p class="mb-1.5 text-[10px] leading-snug text-muted">
        Word frequency across hit text (stopwords removed). Heuristic — not KG entities.
      </p>
      <ul
        v-if="terms.length"
        class="space-y-1.5"
      >
        <li
          v-for="t in terms"
          :key="t.term"
          class="flex items-center gap-2"
        >
          <span class="w-24 shrink-0 truncate font-mono text-[10px] text-elevated-foreground" :title="t.term">{{
            t.term
          }}</span>
          <div class="h-2 min-w-0 flex-1 overflow-hidden bg-overlay">
            <div
              class="h-full bg-primary"
              :style="{ width: `${(t.count / maxTermCount) * 100}%` }"
            />
          </div>
          <span class="w-6 shrink-0 text-right font-mono text-[10px] text-muted">{{ t.count }}</span>
        </li>
      </ul>
      <p
        v-else
        class="text-[11px] text-muted"
      >
        No terms extracted (empty text or only stopwords).
      </p>
    </section>
  </dialog>
</template>
