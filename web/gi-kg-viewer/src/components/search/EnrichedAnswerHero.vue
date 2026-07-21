<script setup lang="ts">
/**
 * Enriched-answer hero — Search v3 §S5 (RFC-107, UXS-016 + UXS-008).
 *
 * Sits directly above ``ResultSetOperatorBar`` on the Search main tab.
 * Renders an aggregated summary of the QueryEnricher chain output
 * (RFC-088 chunk 5) — today that's just
 * ``related_topics`` per hit (from ``query_topic_relatedness``); the
 * component's shape is deliberately open so a future
 * ``synthesized_answer`` field on ``SearchResponse`` can slot in without
 * a schema change.
 *
 * States (UXS-008):
 *   * Hidden — enrichment not requested OR no decorated hits AND no
 *     error signal. Renders nothing (returns null from the template).
 *   * Skeleton — enrichment requested (``filters.enrichResults=true``
 *     effectively on) AND a search is loading. Placeholder body so the
 *     hero's layout doesn't jump on the settled state.
 *   * Error — server sent ``enrichment_error`` (non-fatal); muted
 *     alert row explains that the QueryEnricher chain failed but the
 *     vector hits are still valid.
 *   * Rendered — one or more hits carried
 *     ``metadata.query_enrichments.related_topics``; we aggregate,
 *     de-duplicate, and rank by summed ``similarity``. Top-N (default
 *     6) surface as clickable topic chips that open the Topic subject
 *     rail via the shared subject store.
 *
 * The hero is intentionally silent when enrichment is off — this is a
 * bounded-cost surface (per the RFC: NOT rendered by the Cmd-K palette
 * or the LeftPanel launcher).
 */
import { computed } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import { useSearchStore } from '../../stores/search'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

interface RelatedTopicPayload {
  topic_id: string
  topic_label?: string
  similarity?: number
}

interface AggregatedTopic {
  topic_id: string
  label: string
  summedSimilarity: number
  hitCount: number
}

const search = useSearchStore()
const shell = useShellStore()
const subject = useSubjectStore()

/**
 * Effective enrichment-on: user's explicit choice wins; when unset
 * (``null``), mirrors the server's advertised capability. Matches the
 * chip's effective-on rule exactly (SearchEnrichedChip.vue).
 */
const enrichmentOn = computed(() => {
  const raw = search.filters.enrichResults
  if (raw === null) return Boolean(shell.enrichedSearchAvailable)
  return raw === true
})

function readRelatedTopics(hit: SearchHit): RelatedTopicPayload[] {
  const qe = (hit.metadata as Record<string, unknown> | undefined)
    ?.query_enrichments
  if (!qe || typeof qe !== 'object') return []
  const rt = (qe as Record<string, unknown>).related_topics
  if (!Array.isArray(rt)) return []
  const out: RelatedTopicPayload[] = []
  for (const item of rt) {
    if (item == null || typeof item !== 'object') continue
    const row = item as Record<string, unknown>
    const id = row.topic_id
    if (typeof id !== 'string' || !id.trim()) continue
    const lbl = row.topic_label
    const sim = row.similarity
    out.push({
      topic_id: id.trim(),
      topic_label: typeof lbl === 'string' && lbl.trim() ? lbl.trim() : undefined,
      similarity: typeof sim === 'number' && Number.isFinite(sim) ? sim : undefined,
    })
  }
  return out
}

/**
 * Aggregate related_topics across every hit: sum similarity, count hits,
 * pick the best label. Order by summed similarity desc, then hit-count
 * desc, then topic_id asc for a stable tiebreak.
 */
const aggregated = computed<AggregatedTopic[]>(() => {
  const map = new Map<string, AggregatedTopic>()
  for (const hit of search.results) {
    for (const t of readRelatedTopics(hit)) {
      const entry = map.get(t.topic_id) ?? {
        topic_id: t.topic_id,
        label: t.topic_label ?? t.topic_id,
        summedSimilarity: 0,
        hitCount: 0,
      }
      entry.summedSimilarity += t.similarity ?? 0
      entry.hitCount += 1
      // Prefer a real label over a raw id if one shows up on any hit.
      if (t.topic_label && entry.label === entry.topic_id) {
        entry.label = t.topic_label
      }
      map.set(t.topic_id, entry)
    }
  }
  return Array.from(map.values()).sort((a, b) => {
    if (b.summedSimilarity !== a.summedSimilarity) {
      return b.summedSimilarity - a.summedSimilarity
    }
    if (b.hitCount !== a.hitCount) return b.hitCount - a.hitCount
    return a.topic_id.localeCompare(b.topic_id)
  })
})

const TOP_N = 6
const topRelatedTopics = computed(() => aggregated.value.slice(0, TOP_N))
const overflowCount = computed(() =>
  Math.max(0, aggregated.value.length - TOP_N),
)

const hasContent = computed(() => topRelatedTopics.value.length > 0)
const hasError = computed(() => search.enrichmentCallFailed)
const isSkeleton = computed(
  () => enrichmentOn.value && search.loading && !hasContent.value && !hasError.value,
)
const isHidden = computed(
  () => !enrichmentOn.value || (!hasContent.value && !hasError.value && !isSkeleton.value),
)

function focusTopic(topicId: string): void {
  const id = topicId.trim()
  if (!id) return
  subject.focusTopic(id)
}
</script>

<template>
  <section
    v-if="!isHidden"
    class="rounded border border-gi/40 bg-gi/5 p-3 text-sm text-elevated-foreground"
    data-testid="enriched-answer-hero"
    aria-label="Enriched answer summary"
  >
    <div class="mb-2 flex flex-wrap items-center gap-2">
      <span
        class="rounded bg-gi/20 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-gi"
        title="Server-side QueryEnricher chain (RFC-088 chunk 5) — deterministic, no LLM."
      >
        AI-generated / grounded
      </span>
      <h3 class="text-xs font-semibold text-surface-foreground">
        Related topics from your query
      </h3>
    </div>

    <div
      v-if="hasError"
      class="text-[11px] text-warning"
      data-testid="enriched-answer-error"
      role="alert"
    >
      Enrichment failed for this query — vector hits above are still valid.
    </div>

    <div
      v-else-if="isSkeleton"
      class="flex animate-pulse gap-2"
      data-testid="enriched-answer-skeleton"
      aria-label="Loading enrichment output"
    >
      <span
        v-for="i in 4"
        :key="i"
        class="h-5 w-20 rounded bg-overlay"
      />
    </div>

    <template v-else>
      <ul
        class="flex flex-wrap gap-1.5"
        data-testid="enriched-answer-topics"
      >
        <li v-for="t in topRelatedTopics" :key="t.topic_id">
          <button
            type="button"
            class="inline-flex items-center gap-1 rounded border border-border/60 bg-surface px-1.5 py-0.5 text-[11px] text-surface-foreground hover:bg-overlay"
            :data-testid="`enriched-answer-topic-${t.topic_id}`"
            :title="`Open Topic panel — surfaced by ${t.hitCount} of ${search.results.length} hits (score ${t.summedSimilarity.toFixed(2)})`"
            @click="focusTopic(t.topic_id)"
          >
            <span class="truncate">{{ t.label }}</span>
            <span class="text-muted">·</span>
            <span class="text-muted">{{ t.hitCount }}</span>
          </button>
        </li>
        <li
          v-if="overflowCount > 0"
          class="inline-flex items-center px-1 text-[10px] text-muted"
          data-testid="enriched-answer-overflow"
        >
          +{{ overflowCount }} more
        </li>
      </ul>
      <p class="mt-1.5 text-[10px] text-muted">
        Synthesised from
        <span class="font-medium">query_topic_relatedness</span>
        (deterministic, no LLM). Click a topic to open its detail panel.
      </p>
    </template>
  </section>
</template>
