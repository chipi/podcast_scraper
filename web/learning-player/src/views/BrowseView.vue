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
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
defineOptions({ name: 'BrowseView' }) // stable name for <keep-alive :include> (App.vue)
import Tabs from '../components/Tabs.vue'
import { panelAttrs, type TabSpec } from '../components/tabs'
import CatalogView from './CatalogView.vue'
import ShowBrowseView from './ShowBrowseView.vue'
import TopicBrowseView from './TopicBrowseView.vue'
import PersonBrowseView from './PersonBrowseView.vue'

const { t } = useI18n()
const route = useRoute()

type Tab = 'episodes' | 'shows' | 'topics' | 'people'
// Shared tab strip (#1594 item 7). This one already had roles and `role="tabpanel"`; what it
// lacked was the `aria-controls`/`aria-labelledby` PAIR between them and any arrow-key movement.
const TAB_KEYS: { key: Tab; labelKey: string }[] = [
  { key: 'episodes', labelKey: 'browse.episodes' },
  { key: 'shows', labelKey: 'browse.shows' },
  { key: 'topics', labelKey: 'browse.topics' },
  { key: 'people', labelKey: 'browse.people' },
]
const tabs = computed<TabSpec<Tab>[]>(() =>
  TAB_KEYS.map((tb) => ({ key: tb.key, label: t(tb.labelKey), testid: `browse-tab-${tb.key}` })),
)
const initial = String(route.query.tab || '')
const tab = ref<Tab>(TAB_KEYS.some((tb) => tb.key === initial) ? (initial as Tab) : 'episodes')

// This view is kept-alive (App.vue), so setup runs once — without this watch a later in-app
// navigation to ?tab=<other> (e.g. Home's "Browse people" chip after the hub was already opened on
// Topics) would leave the stale tab selected. Re-sync whenever the query tab changes.
watch(
  () => route.query.tab,
  (v) => {
    const q = String(v || '')
    if (TAB_KEYS.some((tb) => tb.key === q)) tab.value = q as Tab
  }
)
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="browse-view">
    <!--
      A kicker + one line under the title (#2004 follow-up).

      Home spends this band on the ask-hero; Browse spent it on nothing, so the same vertical gap
      read as empty here and purposeful there. This is the smallest thing that earns the space: it
      says what Browse is for, which the bare word does not.
    -->
    <span class="lp-kicker text-topic">{{ t('browse.hubKicker') }}</span>
    <h1 class="mt-1 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.hubTitle') }}
    </h1>
    <p class="mb-4 mt-1 text-sm text-muted">{{ t('browse.hubLede') }}</p>

    <Tabs
      v-model="tab"
      :tabs="tabs"
      :label="t('browse.hubTitle')"
      id-prefix="browse"
      class="mb-6"
    />

    <div v-show="tab === 'episodes'" v-bind="panelAttrs('browse', 'episodes')" data-testid="browse-panel-episodes"><CatalogView embedded /></div>
    <div v-show="tab === 'shows'" v-bind="panelAttrs('browse', 'shows')" data-testid="browse-panel-shows"><ShowBrowseView embedded /></div>
    <div v-show="tab === 'topics'" v-bind="panelAttrs('browse', 'topics')" data-testid="browse-panel-topics"><TopicBrowseView embedded /></div>
    <div v-show="tab === 'people'" v-bind="panelAttrs('browse', 'people')" data-testid="browse-panel-people"><PersonBrowseView embedded /></div>
  </section>
</template>
