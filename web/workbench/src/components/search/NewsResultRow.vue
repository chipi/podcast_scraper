<script setup lang="ts">
/**
 * A news hit in search results (UXS-019 "The news result row"): a sibling of the podcast item
 * row, never a variant of the text row. It shows the item's metadata and, when its rights allow,
 * the platform's summary — never the chunk that matched (ADR-179). Its title and the rail button
 * open the article card; "Open article" links out.
 */
import { computed } from 'vue'
import type { NewsItem } from '../../api/searchApi'
import { formatRelativeShort } from '../../utils/relativeTime'
import { primaryLanguage } from '../../utils/language'
import LanguageBadge from '../shared/LanguageBadge.vue'

const props = defineProps<{ item: NewsItem }>()
const emit = defineEmits<{ open: [item: NewsItem] }>()

const EDITORIAL_BADGES: Record<string, string> = {
  opinion: 'Opinion',
  analysis: 'Analysis',
  paid: 'Paid',
}

const when = computed(() => formatRelativeShort(props.item.published_at, Date.now()))
const byline = computed(() => (props.item.byline?.length ? `By ${props.item.byline.join(', ')}` : ''))
const meta = computed(() =>
  [props.item.publisher?.name ?? props.item.site, byline.value, when.value].filter((s) => !!s).join(' · '),
)
const editorial = computed(() => EDITORIAL_BADGES[props.item.editorial_type ?? ''] ?? '')
const showLanguage = computed(() => {
  const lang = primaryLanguage(props.item.language)
  return !!lang && lang !== 'en'
})
const fromFeed = computed(() => ['teaser', 'video'].includes(props.item.text_status ?? ''))
const summary = computed(() => (props.item.serveable === 'summary' ? props.item.summary : null))
</script>

<template>
  <article
    class="min-w-0 rounded-sm border border-border border-l-[3px] border-l-news bg-surface px-3 py-2"
    data-testid="search-result-news"
  >
    <div class="flex min-w-0 items-center gap-2 text-[10px] text-muted">
      <span class="font-semibold uppercase tracking-wider text-news">News</span>
      <span class="min-w-0 truncate" data-testid="search-result-news-meta">{{ meta }}</span>
      <span class="ml-auto flex shrink-0 items-center gap-1">
        <span
          v-if="editorial"
          class="rounded border border-border px-1 text-[10px] text-surface-foreground"
          data-testid="search-result-news-editorial"
        >{{ editorial }}</span>
        <LanguageBadge v-if="showLanguage" :lang="item.language" />
      </span>
    </div>
    <button
      type="button"
      class="mt-0.5 block w-full min-w-0 break-words text-left text-[13px] font-semibold leading-snug text-surface-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      data-testid="search-result-news-title"
      @click="emit('open', item)"
    >
      {{ item.title }}
    </button>
    <p
      v-if="summary"
      class="mt-1 line-clamp-3 break-words text-[12px] leading-snug text-muted"
      data-testid="search-result-news-summary"
    >
      {{ summary }}
    </p>
    <p v-if="summary && fromFeed" class="mt-0.5 text-[10px] text-muted">Summary from the feed</p>
    <div class="mt-1 flex items-center justify-end gap-3 text-[11px]">
      <span v-if="item.paywalled" class="text-[10px] text-muted">Paywalled</span>
      <a
        v-if="item.url"
        :href="item.url"
        target="_blank"
        rel="noopener noreferrer"
        class="text-link hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        data-testid="search-result-news-open"
      >Open article ↗</a>
      <button
        type="button"
        class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        data-testid="search-result-news-rail"
        :aria-label="`Open ${item.title} in the rail`"
        @click="emit('open', item)"
      >
        →
      </button>
    </div>
  </article>
</template>
