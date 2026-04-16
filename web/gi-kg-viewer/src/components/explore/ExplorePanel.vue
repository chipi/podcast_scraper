<script setup lang="ts">
import { ref } from 'vue'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useExploreStore } from '../../stores/explore'
import { useShellStore } from '../../stores/shell'
import { truncate } from '../../utils/formatting'
import { quoteAttributionDisplayFromId } from '../../utils/parsing'
import {
  GI_QUOTE_SPEAKER_UNAVAILABLE_HINT,
  SUPPORTING_QUOTE_SPEAKER_UNAVAILABLE_TESTID,
} from '../../utils/transcriptSourceDisplay'
import HelpTip from '../shared/HelpTip.vue'

const emit = defineEmits<{ 'go-graph': [] }>()

const shell = useShellStore()
const ex = useExploreStore()
const nav = useGraphNavigationStore()

const expandedQuotes = ref<Set<string>>(new Set())

function toggleQuotes(insightId: string): void {
  if (expandedQuotes.value.has(insightId)) {
    expandedQuotes.value.delete(insightId)
  } else {
    expandedQuotes.value.add(insightId)
  }
}

function focusNode(id: string): void {
  const trimmed = id.trim()
  if (!trimmed) return
  nav.requestFocusNode(trimmed)
  emit('go-graph')
}
</script>

<template>
  <section class="rounded-lg border border-border bg-surface p-4">
    <div class="mb-2 flex items-center gap-1.5">
      <h2 class="text-sm font-medium text-surface-foreground">
        Explore &amp; query
      </h2>
      <HelpTip>
        <p class="font-medium text-surface-foreground">
          What this panel is for
        </p>
        <p class="mt-1 text-muted">
          Browse and filter <strong class="text-surface-foreground">Grounded Insights</strong> across
          episodes in your corpus (insight text, linked quotes, topic/speaker rollups). Data comes from
          <code class="rounded bg-canvas px-0.5 text-[10px]">.gi.json</code> only — not the knowledge
          graph and not full transcript/summary search.
        </p>
        <p class="mt-2 text-muted">
          Use <strong class="text-surface-foreground">Topic / speaker</strong> for manual filters, or
          <strong class="text-surface-foreground">Quick questions</strong> for preset phrases — each
          has its own <span class="text-surface-foreground">?</span> with details.
        </p>
        <p class="mt-2 text-muted">
          For free-form, whole-corpus semantic search (summaries, chunks, KG, …), use
          <strong class="text-surface-foreground">Semantic search</strong>.
        </p>
      </HelpTip>
    </div>
    <p class="mb-2 text-xs text-muted">
      <span class="text-surface-foreground">GI only</span>
      — see <span class="text-surface-foreground">?</span> above for scope; subsection
      <span class="text-surface-foreground">?</span> tips explain filters and phrases.
    </p>
    <p
      v-if="!shell.healthStatus"
      class="mb-2 text-xs text-muted"
    >
      Requires the API.
    </p>

    <div class="space-y-4">
      <div>
        <div class="mb-1 flex items-center gap-1.5">
          <h3 class="text-xs font-semibold text-muted">
            Topic / speaker
          </h3>
          <HelpTip>
            <p class="font-medium text-surface-foreground">
              Topic / speaker filters
            </p>
            <ul class="mt-1 list-disc space-y-1 pl-4 text-muted">
              <li>
                <span class="font-medium text-surface-foreground">Topic contains</span> — substring
                on insight topic labels. If
                <code class="rounded bg-canvas px-0.5 text-[10px]">&lt;corpus&gt;/search/vectors.faiss</code>
                exists, this field can use the vector index to pick relevant episodes first
                (routing), then GI files load from disk. That is <em>not</em> the same as
                <strong class="text-surface-foreground">Semantic search</strong>, which ranks all
                indexed types (summaries, transcript chunks, KG, …).
              </li>
              <li>
                <span class="font-medium text-surface-foreground">Speaker contains</span> — substring
                on speaker attribution for quotes linked to insights.
              </li>
              <li>
                <span class="font-medium text-surface-foreground">Grounded only</span> — keep insights
                that pass grounding checks.
              </li>
              <li>
                <span class="font-medium text-surface-foreground">Strict schema</span> — stricter
                validation when reading artifacts.
              </li>
              <li>
                <span class="font-medium text-surface-foreground">Limit / Sort / Min confidence</span> —
                cap rows, order by model confidence or time, optional confidence floor.
              </li>
            </ul>
          </HelpTip>
        </div>
        <div class="space-y-2">
          <label class="block text-xs text-muted">
            Topic contains
            <input
              v-model="ex.filters.topic"
              type="text"
              class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
              :disabled="!shell.healthStatus"
            >
          </label>
          <label class="block text-xs text-muted">
            Speaker contains
            <input
              v-model="ex.filters.speaker"
              type="text"
              class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
              :disabled="!shell.healthStatus"
            >
          </label>
          <div class="flex flex-wrap gap-3 text-xs text-muted">
            <label class="flex items-center gap-1">
              <input
                v-model="ex.filters.groundedOnly"
                type="checkbox"
                class="rounded border-border"
                :disabled="!shell.healthStatus"
              >
              Grounded only
            </label>
            <label class="flex items-center gap-1">
              <input
                v-model="ex.filters.strict"
                type="checkbox"
                class="rounded border-border"
                :disabled="!shell.healthStatus"
              >
              Strict schema
            </label>
          </div>
          <div class="grid grid-cols-2 gap-2">
            <label class="text-xs text-muted">
              Limit
              <input
                v-model.number="ex.filters.limit"
                type="number"
                min="1"
                max="500"
                class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
                :disabled="!shell.healthStatus"
              >
            </label>
            <label class="text-xs text-muted">
              Sort
              <select
                v-model="ex.filters.sortBy"
                class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
                :disabled="!shell.healthStatus"
              >
                <option value="confidence">
                  Confidence
                </option>
                <option value="time">
                  Time
                </option>
              </select>
            </label>
          </div>
          <label class="block text-xs text-muted">
            Min confidence (optional)
            <input
              v-model="ex.filters.minConfidence"
              type="text"
              class="mt-0.5 w-full rounded border border-border bg-elevated px-2 py-1 text-sm"
              placeholder="e.g. 0.5"
              :disabled="!shell.healthStatus"
            >
          </label>
          <button
            type="button"
            class="rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
            :disabled="!shell.healthStatus || ex.loading"
            @click="ex.runFilteredExplore(shell.corpusPath)"
          >
            Run explore
          </button>
        </div>
      </div>

      <div class="border-t border-border pt-3">
        <div class="mb-1 flex items-center gap-1.5">
          <h3 class="text-xs font-semibold text-muted">
            Quick questions
          </h3>
          <HelpTip>
            <p class="font-medium text-surface-foreground">
              Supported phrases (case-insensitive)
            </p>
            <ul class="mt-1 list-disc space-y-1 pl-4 text-muted">
              <li>
                <span class="font-medium text-surface-foreground">Topic:</span>
                “What insights about …”, “What insights are there about …”, “Insights about …”,
                “Show me insights about …”, “Tell me about insights on …”, “What are insights about …”
              </li>
              <li>
                <span class="font-medium text-surface-foreground">Speaker:</span>
                “What did … say?”
              </li>
              <li>
                <span class="font-medium text-surface-foreground">Speaker + topic:</span>
                “What did … say about …?”
              </li>
              <li>
                <span class="font-medium text-surface-foreground">Rank topics:</span>
                “Top topics”, “Which topics have the most insights?”, “What topics have the most insights?”,
                “Rank topics by insights”, “What are the top topics”, “Show topic rankings”
              </li>
            </ul>
          </HelpTip>
        </div>
        <textarea
          v-model="ex.nlQuestion"
          rows="2"
          class="w-full rounded border border-border bg-elevated px-2 py-1.5 text-sm"
          placeholder="What insights about machine learning?"
          :disabled="!shell.healthStatus"
        />
        <button
          type="button"
          class="mt-2 rounded border border-border px-3 py-1.5 text-sm font-medium hover:bg-overlay disabled:opacity-40"
          :disabled="!shell.healthStatus || ex.loading"
          @click="ex.runNaturalLanguage(shell.corpusPath)"
        >
          Run quick question
        </button>
      </div>

      <button
        type="button"
        class="text-xs text-muted underline hover:text-surface-foreground"
        :disabled="!shell.healthStatus"
        @click="ex.clearOutput()"
      >
        Clear results
      </button>
    </div>

    <p
      v-if="ex.error"
      class="mt-2 text-xs text-danger"
    >
      {{ ex.error }}
    </p>

    <div
      v-if="ex.last && !ex.error"
      class="mt-3 max-h-[32rem] space-y-2 overflow-y-auto rounded border border-border bg-elevated p-2 text-xs"
    >
      <div v-if="ex.last.error">
        <p class="font-medium text-warning">
          {{ ex.last.error }}
        </p>
        <p
          v-if="ex.last.detail"
          class="text-muted"
        >
          {{ ex.last.detail }}
        </p>
      </div>

      <template v-else>
        <!-- NL explanation -->
        <p
          v-if="ex.last.kind === 'natural_language' && ex.last.explanation"
          class="text-muted"
        >
          {{ ex.last.explanation }}
        </p>

        <!-- Summary metrics -->
        <div
          v-if="ex.summaryBlock"
          class="flex flex-wrap gap-3 rounded bg-surface px-2 py-1.5 text-[11px] text-muted"
        >
          <span><strong class="text-surface-foreground">{{ ex.summaryBlock.insight_count }}</strong> insights</span>
          <span>
            <strong class="text-surface-foreground">{{ ex.summaryBlock.grounded_insight_count }}</strong> grounded
            <template v-if="ex.summaryBlock.insight_count > 0">
              ({{ ((ex.summaryBlock.grounded_insight_count / ex.summaryBlock.insight_count) * 100).toFixed(1) }}%)
            </template>
          </span>
          <span><strong class="text-surface-foreground">{{ ex.summaryBlock.quote_count }}</strong> quotes</span>
          <span
            v-if="ex.summaryBlock.episode_count > 0"
          >
            <strong class="text-surface-foreground">{{ ex.summaryBlock.episode_count }}</strong> episodes
          </span>
          <span
            v-if="ex.summaryBlock.speaker_count > 0"
          >
            <strong class="text-surface-foreground">{{ ex.summaryBlock.speaker_count }}</strong> speakers
          </span>
          <span
            v-if="ex.summaryBlock.episodes_searched > 0"
          >
            ({{ ex.summaryBlock.episodes_searched }} searched)
          </span>
        </div>

        <!-- Topics leaderboard -->
        <div
          v-if="ex.leaderboardRows.length"
          class="overflow-x-auto"
        >
          <table class="w-full border-collapse text-left text-[11px]">
            <thead>
              <tr class="border-b border-border text-muted">
                <th class="py-1 pr-2">
                  Topic
                </th>
                <th class="py-1">
                  Insights
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, i) in ex.leaderboardRows"
                :key="`${row.topic_id}-${i}`"
                class="cursor-pointer border-b border-border/60 transition-colors hover:bg-overlay"
                @click="focusNode(row.topic_id)"
              >
                <td class="py-1 pr-2">
                  {{ row.label }}
                </td>
                <td class="py-1">
                  {{ row.insight_count }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Top speakers -->
        <div
          v-if="ex.topSpeakers.length"
          class="rounded bg-surface px-2 py-1.5"
        >
          <p class="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">
            Top speakers
          </p>
          <div class="space-y-0.5 text-[11px]">
            <div
              v-for="(sp, i) in ex.topSpeakers"
              :key="`sp-${i}`"
              class="flex items-center justify-between gap-2"
            >
              <span class="text-surface-foreground">{{ sp.name || sp.speaker_id }}</span>
              <span class="text-muted">{{ sp.quote_count }} quotes · {{ sp.insight_count }} insights</span>
            </div>
          </div>
        </div>

        <!-- Insight cards -->
        <article
          v-for="ins in ex.insightRows"
          :key="ins.insight_id"
          class="cursor-pointer rounded border border-border bg-surface p-2 transition-colors hover:border-primary/50 hover:bg-overlay"
          @click="focusNode(ins.insight_id)"
        >
          <div class="mb-1 flex flex-wrap items-center gap-2">
            <span class="font-mono text-[10px] text-primary">{{ ins.insight_id }}</span>
            <span
              v-if="ins.confidence != null"
              class="rounded bg-overlay px-1 py-0.5 text-[10px] text-muted"
            >{{ ins.confidence.toFixed(2) }}</span>
            <span
              v-if="ins.grounded === true"
              class="rounded bg-green-600/20 px-1 py-0.5 text-[10px] font-medium text-green-400"
            >grounded</span>
            <span
              v-else-if="ins.grounded === false"
              class="rounded bg-yellow-600/20 px-1 py-0.5 text-[10px] font-medium text-yellow-400"
            >ungrounded</span>
            <svg
              class="ml-auto h-3 w-3 shrink-0 text-primary/60"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="3" />
              <circle cx="12" cy="12" r="9" stroke-dasharray="4 3" />
            </svg>
          </div>
          <p class="leading-snug text-surface-foreground">
            {{ truncate(ins.text, 280) }}
          </p>
          <div
            v-if="ins.episode"
            class="mt-1 text-[10px] text-muted"
          >
            <span v-if="ins.episode.title">{{ truncate(ins.episode.title, 60) }}</span>
            <span v-if="ins.episode.episode_id"> · {{ ins.episode.episode_id }}</span>
            <span v-if="ins.episode.publish_date"> · {{ ins.episode.publish_date }}</span>
          </div>

          <!-- Supporting quotes (collapsible) -->
          <div
            v-if="ins.supporting_quotes?.length"
            class="mt-1.5"
            @click.stop
          >
            <button
              type="button"
              class="text-[10px] text-muted underline hover:text-surface-foreground"
              @click="toggleQuotes(ins.insight_id)"
            >
              {{ expandedQuotes.has(ins.insight_id) ? 'Hide' : 'Show' }}
              {{ ins.supporting_quotes.length }}
              {{ ins.supporting_quotes.length === 1 ? 'quote' : 'quotes' }}
            </button>
            <div
              v-if="expandedQuotes.has(ins.insight_id)"
              class="mt-1 space-y-1"
            >
              <blockquote
                v-for="(q, qi) in ins.supporting_quotes"
                :key="qi"
                class="border-l-2 border-primary/40 pl-2 text-[11px] leading-snug text-muted"
              >
                <p>{{ truncate(q.text, 300) }}</p>
                <p
                  v-if="q.speaker_name || q.speaker_id"
                  class="mt-0.5 text-[10px] font-medium text-primary"
                >
                  —
                  {{
                    q.speaker_name ||
                      quoteAttributionDisplayFromId(
                        typeof q.speaker_id === 'string' ? q.speaker_id : '',
                      )
                  }}
                  <span
                    v-if="q.start_ms != null"
                    class="font-normal text-muted"
                  >
                    ({{ (q.start_ms / 1000).toFixed(1) }}s{{ q.end_ms != null ? ` – ${(q.end_ms / 1000).toFixed(1)}s` : '' }})
                  </span>
                </p>
                <p
                  v-else-if="String(q.text ?? '').trim()"
                  class="mt-0.5 text-[10px] leading-snug text-muted/80"
                  :data-testid="SUPPORTING_QUOTE_SPEAKER_UNAVAILABLE_TESTID"
                >
                  {{ GI_QUOTE_SPEAKER_UNAVAILABLE_HINT }}
                </p>
              </blockquote>
            </div>
          </div>
        </article>

        <p
          v-if="!ex.insightRows.length && !ex.leaderboardRows.length"
          class="text-muted"
        >
          No tabular rows in this response (empty corpus, filters, or pattern result).
        </p>
      </template>
    </div>
  </section>
</template>
