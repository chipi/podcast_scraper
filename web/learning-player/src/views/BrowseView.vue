<script setup lang="ts">
/**
 * Browse hub (#14, revised) — one tabbed page (Episodes · Topics · People), Episodes active by
 * default, like Library. The three corpus indexes render INLINE as tab panels rather than three
 * buttons that navigate away (which was a navigation hustle). Each panel reuses the standalone index
 * view in `embedded` mode, which drops its page heading and — for Topics/People — its back-to-Home
 * button (that button is only meaningful on the standalone routes reached from Home, not here).
 *
 * v-show (not v-if) keeps each panel mounted so switching tabs never refetches; supports ?tab= for
 * deep links.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
defineOptions({ name: 'BrowseView' }) // stable name for <keep-alive :include> (App.vue)
import CatalogView from './CatalogView.vue'
import ShowBrowseView from './ShowBrowseView.vue'
import TopicBrowseView from './TopicBrowseView.vue'
import PersonBrowseView from './PersonBrowseView.vue'

const { t } = useI18n()
const route = useRoute()

type Tab = 'episodes' | 'shows' | 'topics' | 'people'
const tabs: { key: Tab; label: string }[] = [
  { key: 'episodes', label: 'browse.episodes' },
  { key: 'shows', label: 'browse.shows' },
  { key: 'topics', label: 'browse.topics' },
  { key: 'people', label: 'browse.people' },
]
const initial = String(route.query.tab || '')
const tab = ref<Tab>(tabs.some((tb) => tb.key === initial) ? (initial as Tab) : 'episodes')
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="browse-view">
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">{{ t('browse.hubTitle') }}</h1>

    <div role="tablist" :aria-label="t('browse.hubTitle')" class="mb-6 flex flex-wrap gap-1 border-b border-border">
      <button
        v-for="tb in tabs"
        :key="tb.key"
        type="button"
        role="tab"
        :aria-selected="tab === tb.key"
        :data-testid="`browse-tab-${tb.key}`"
        class="-mb-px shrink-0 whitespace-nowrap border-b-2 px-3 py-2 text-sm font-bold transition"
        :class="
          tab === tb.key
            ? 'border-accent text-canvas-foreground'
            : 'border-transparent text-muted hover:text-canvas-foreground'
        "
        @click="tab = tb.key"
      >
        {{ t(tb.label) }}
      </button>
    </div>

    <div v-show="tab === 'episodes'" role="tabpanel"><CatalogView embedded /></div>
    <div v-show="tab === 'shows'" role="tabpanel"><ShowBrowseView embedded /></div>
    <div v-show="tab === 'topics'" role="tabpanel"><TopicBrowseView embedded /></div>
    <div v-show="tab === 'people'" role="tabpanel"><PersonBrowseView embedded /></div>
  </section>
</template>
