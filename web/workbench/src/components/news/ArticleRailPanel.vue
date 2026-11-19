<script setup lang="ts">
/**
 * The article card in the subject rail (UXS-019 "The article card"). It shows a news item's
 * metadata, its summary when its rights allow, a link out and the topics and people its KG names.
 * Never the body, a transcript, a quote or a "Read more" (ADR-179).
 */
import { computed, ref, watch } from 'vue'
import { fetchNewsItem, type NewsItemDetail, type NewsKgNode } from '../../api/newsApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { formatAbsoluteUtc } from '../../utils/relativeTime'
import { primaryLanguage } from '../../utils/language'
import { StaleGeneration } from '../../utils/staleGeneration'
import LanguageBadge from '../shared/LanguageBadge.vue'

const shell = useShellStore()
const subject = useSubjectStore()

const detail = ref<NewsItemDetail | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const gate = new StaleGeneration()

async function load(itemId: string | null): Promise<void> {
  const root = shell.corpusPath?.trim()
  detail.value = null
  error.value = null
  if (!itemId || !root) return
  const seq = gate.bump()
  loading.value = true
  try {
    const got = await fetchNewsItem(root, itemId)
    if (gate.isStale(seq)) return
    detail.value = got
  } catch (e) {
    if (gate.isStale(seq)) return
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gate.isCurrent(seq)) loading.value = false
  }
}
watch(() => subject.articleItemId, (id) => void load(id), { immediate: true })

const item = computed(() => detail.value?.item ?? null)
const EDITORIAL: Record<string, string> = { opinion: 'Opinion', analysis: 'Analysis', paid: 'Paid' }
const editorial = computed(() => EDITORIAL[item.value?.editorial_type ?? ''] ?? '')
const showLanguage = computed(() => {
  const lang = primaryLanguage(item.value?.language)
  return !!lang && lang !== 'en'
})
const partialText = computed(() => ['teaser', 'video'].includes(item.value?.text_status ?? ''))
/** The link out's accessible name says where it goes and that it leaves the app (UXS-019). */
const openLabel = computed(() => {
  const domain = item.value?.publisher?.domain || item.value?.site || ''
  return domain
    ? `Open article on ${domain}, opens in a new tab`
    : 'Open article, opens in a new tab'
})
const authors = computed(() => {
  const byLabel = new Map<string, NewsKgNode>()
  for (const n of detail.value?.kg_nodes ?? []) {
    if (n.role === 'author') byLabel.set(n.label.toLowerCase(), n)
  }
  return (item.value?.byline ?? []).map((name) => ({ name, node: byLabel.get(name.toLowerCase()) }))
})
const chips = computed(() => (detail.value?.kg_nodes ?? []).filter((n) => n.role !== 'author'))

function openNode(n: NewsKgNode): void {
  if (n.type === 'Topic') subject.focusTopic(n.id, { uiTitle: n.label })
  else if (n.type === 'Person') subject.focusPerson(n.id, { uiTitle: n.label })
  else subject.focusEntity(n.id, { uiTitle: n.label })
}
</script>

<template>
  <div class="min-w-0 overflow-y-auto px-1 py-2" data-testid="article-rail-panel">
    <p v-if="loading" class="text-[11px] text-muted">Loading…</p>
    <p v-else-if="error" class="text-[11px] text-warning">{{ error }}</p>
    <template v-else-if="item">
      <h3
        class="break-words text-sm font-semibold leading-snug text-surface-foreground"
        data-testid="article-title"
      >
        {{ item.title }}
      </h3>
      <p v-if="item.publisher" class="mt-1 text-[11px] text-surface-foreground" data-testid="article-publisher">
        <button
          type="button"
          class="hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          data-testid="article-publisher-open"
          :title="`All of ${item.publisher.name}'s articles`"
          @click="shell.openLibraryNews(item.publisher.id)"
        >{{ item.publisher.name }}</button>
        <span v-if="item.publisher.domain" class="text-muted">· {{ item.publisher.domain }}</span>
      </p>
      <p v-else-if="item.site" class="mt-1 text-[11px] text-surface-foreground" data-testid="article-site">
        {{ item.site }}
      </p>
      <p v-if="authors.length" class="mt-0.5 text-[11px] text-muted" data-testid="article-byline">
        By
        <template v-for="(a, i) in authors" :key="a.name">
          <button
            v-if="a.node"
            type="button"
            class="text-person hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            @click="openNode(a.node)"
          >{{ a.name }}</button>
          <span v-else>{{ a.name }}</span>{{ i < authors.length - 1 ? ', ' : '' }}
        </template>
      </p>
      <p v-if="item.published_at" class="mt-0.5 text-[10px] text-muted" data-testid="article-published">
        {{ formatAbsoluteUtc(item.published_at) }}
      </p>
      <div class="mt-1.5 flex flex-wrap items-center gap-1">
        <span v-if="editorial" class="rounded border border-border px-1 text-[10px]">{{ editorial }}</span>
        <LanguageBadge v-if="showLanguage" :lang="item.language" />
        <span v-if="item.paywalled" class="rounded border border-border px-1 text-[10px] text-muted">Paywalled</span>
      </div>
      <template v-if="item.serveable === 'summary'">
        <p
          v-if="item.summary"
          class="mt-2 break-words text-[12px] leading-relaxed text-surface-foreground"
          data-testid="article-summary"
        >
          {{ item.summary }}
        </p>
        <p v-else-if="!partialText" class="mt-2 text-[11px] text-muted" data-testid="article-no-summary">
          No summary yet
        </p>
      </template>
      <p v-else class="mt-2 text-[11px] text-muted" data-testid="article-metadata-only">
        This publisher's items show their title and link only.
      </p>
      <a
        v-if="item.url"
        :href="item.url"
        target="_blank"
        rel="noopener noreferrer"
        :aria-label="openLabel"
        class="mt-3 block w-full rounded border border-border px-2 py-1.5 text-center text-[12px] font-medium text-link hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        data-testid="article-open"
      >Open article ↗</a>
      <section v-if="chips.length" class="mt-3" data-testid="article-kg">
        <h4 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Topics and entities
        </h4>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="n in chips"
            :key="n.id"
            type="button"
            class="rounded border px-1.5 py-0.5 text-[11px] hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            :class="n.type === 'Topic' ? 'border-topic/60 text-topic' : 'border-kg/60 text-kg'"
            data-testid="article-kg-chip"
            @click="openNode(n)"
          >
            {{ n.label }}
          </button>
        </div>
      </section>
    </template>
  </div>
</template>
