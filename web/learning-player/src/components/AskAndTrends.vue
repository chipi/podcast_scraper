<script setup lang="ts">
/**
 * "Find any moment you've heard" + Trends — ONE block that Home and Discover both render
 * (operator 2026-10-05: "Home and Discover are the same screens … things have to be identical").
 *
 * The two pages already shared `DiscoveryExplorer`, and still diverged, because each page wrapped it
 * itself: Discover added its own `px-4` gutter (Trends was 32px narrower), spaced it `mt-4` against
 * Home's `mt-7`, put search above Trends where Home put it below, showed 10 rows against Home's 3,
 * gave only Home topic chips under the search box, and opened more rows in a different way. Sharing
 * the component was not enough; the WRAPPING had to be shared too, which is this.
 *
 * Fixed here, for both: search (with its two trending-topic chips) first, then Trends; one spacing;
 * 3 rows on a phone and 5 on desktop; more rows expand in place from the header's "all ›". The page
 * decides only what a tap opens (Home: an overlay; Discover: the page) and keeps its own test ids
 * through `prefix`, so the specs written against either page still find their elements.
 */
import { onActivated, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
import { useRouter } from "vue-router"
import DiscoveryExplorer from "./DiscoveryExplorer.vue"
import SectionHeading from "./SectionHeading.vue"
import { getTrendingTopics } from "../services/api"
import { useIsDesktop } from "../composables/useMediaQuery"

type Kind = "topic" | "theme" | "storyline" | "person"

const props = defineProps<{
  /** `home` or `browse` — namespaces the test ids each page's specs already use. */
  prefix: "home" | "browse"
  /** Which Trends kind to open on (Discover's `?trends=`). */
  kind?: Kind
}>()
const emit = defineEmits<{ (e: "open", payload: { kind: Kind; id: string; rank: number }): void }>()

const { t } = useI18n()
const router = useRouter()
const isDesktop = useIsDesktop()

const query = ref("")
/* Both pages are kept-alive, so the box kept whatever you last typed and greeted you with a stale
   query on the way back — which reads as the app remembering something you did not ask it to.
   Cleared on re-entry; the search you ran is still on the results page, where it belongs. Blank
   submits are ignored rather than routing to an empty results page. */
onActivated(() => {
  query.value = ""
})
function goSearch(q: string): void {
  const term = q.trim()
  if (term) void router.push({ name: "search", query: { q: term } })
}

/**
 * Two trending topics as tappable examples under the box (#1964 follow-up, UXS-012 §103): the box
 * says "ask across every episode" and, without examples, offers an empty field you have to already
 * know what to type into. `getTrendingTopics()` is memoised, so this costs no extra request, and it
 * is silent on failure — a box without chips is fine, one with an error where they should be is not.
 */
const heroTopics = ref<Array<{ id: string; label: string }>>([])
onMounted(async () => {
  try {
    const res = await getTrendingTopics()
    heroTopics.value = (res.topics ?? [])
      .slice(0, 4)
      .map((tp) => ({ id: tp.topic_id, label: tp.topic_label || tp.topic_id.split(":").pop() || "" }))
      .filter((tp) => tp.label)
  } catch {
    heroTopics.value = []
  }
})

/**
 * Each page keeps its own test ids, spelled out as literals in the bindings below rather than built
 * from `prefix`: the surface-map and touch checks read the source for the ids the specs depend on.
 */
const home = props.prefix === "home"

/** The Trends block, so Discover can scroll it into view for a `?trends=` deep link. */
const trendsEl = ref<HTMLElement | null>(null)
defineExpose({ trendsEl })
</script>

<template>
  <!-- Same lp-search markup on both pages: two entry points to one capability are one control. -->
  <section class="mt-7" :data-testid="home ? 'home-search-section' : 'browse-search-section'">
    <SectionHeading :title="t('ask.title')" />
    <form class="lp-search mt-3 flex gap-2" @submit.prevent="goSearch(query)">
      <label class="sr-only" :for="`${prefix}-search`">{{ t("ask.kicker") }}</label>
      <input
        :id="`${prefix}-search`"
        v-model="query"
        type="search"
        :placeholder="t('ask.placeholder')"
        :data-testid="home ? 'home-search-input' : 'browse-search-input'"
        class="h-11 min-w-0 flex-1 rounded-full border border-border bg-surface px-4 text-sm"
      />
      <button
        type="submit"
        :data-testid="home ? 'home-search-submit' : 'browse-search-submit'"
        class="h-11 shrink-0 rounded-full bg-accent px-5 font-bold text-accent-foreground"
      >
        {{ t("search.title") }}
      </button>
    </form>
    <!-- Just TWO examples: a single row that always fits, rather than a longer list clipped at the
         screen edge. The test ids keep Home's names on both pages — the chips were Home's. -->
    <div v-if="heroTopics.length" data-testid="home-topic-chips" class="mt-3 flex flex-wrap gap-2">
      <button
        v-for="tp in heroTopics.slice(0, 2)"
        :key="tp.id"
        type="button"
        data-testid="home-topic-chip"
        class="rounded-full border border-topic/40 px-3 py-1.5 text-sm font-semibold text-topic transition hover:bg-overlay"
        @click="goSearch(tp.label)"
      >
        {{ tp.label }}
      </button>
    </div>
  </section>

  <section
    id="trends"
    ref="trendsEl"
    class="mt-7 scroll-mt-4"
    :data-testid="home ? 'home-discovery' : 'browse-discovery'"
  >
    <!-- 5 rows on desktop, 3 on a phone, expanding in place from the header on both pages. A prop
         cannot follow a media query in CSS, hence `useIsDesktop`. -->
    <DiscoveryExplorer
      :collapsed="isDesktop ? 5 : 3"
      see-all
      :kind="props.kind"
      :title="t('browse.trendsTitle')"
      @open="emit('open', $event)"
    />
  </section>
</template>
