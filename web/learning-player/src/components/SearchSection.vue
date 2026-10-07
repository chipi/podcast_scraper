<script setup lang="ts">
/**
 * "Find what's worth hearing" (operator 2026-10-07; it was "Find any moment you've heard", which
 * read as a search of your own history to someone who had heard nothing yet) — the search box, with two trending-topic chips under it. Home and
 * Discover both render THIS, so the two entry points to one capability are one control and look
 * identical (operator 2026-10-05). Spacing is owned here, not by the page, for the same reason:
 * each page wrapping it its own way is exactly how the two screens drifted apart.
 */
import { onActivated, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
import { useRouter } from "vue-router"
import SectionHeading from "./SectionHeading.vue"
import { getTrendingTopics } from "../services/api"

const props = defineProps<{
  /** `home` or `browse` — selects the test ids each page's specs already use. */
  prefix: "home" | "browse"
}>()

const { t } = useI18n()
const router = useRouter()
// Each page keeps its own test ids, spelled out as literals in the bindings below rather than built
// from `prefix`: the surface-map and touch checks read the source for the ids the specs depend on.
const home = props.prefix === "home"

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
 * Two trending topics as tappable examples (#1964 follow-up, UXS-012 §103): the box says "ask
 * across every episode" and, without examples, offers an empty field you have to already know what
 * to type into. `getTrendingTopics()` is memoised, so this costs no extra request, and it is silent
 * on failure — a box without chips is fine, one with an error where they should be is not.
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
</script>

<template>
  <section class="mt-7" :data-testid="home ? 'home-search-section' : 'browse-search-section'">
    <SectionHeading :title="t('ask.title')" />
    <form class="lp-search mt-3 flex gap-2" @submit.prevent="goSearch(query)">
      <label class="sr-only" :for="home ? 'home-search' : 'browse-search'">{{ t("ask.kicker") }}</label>
      <input
        :id="home ? 'home-search' : 'browse-search'"
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
    <!-- "Try:" says what the chips ARE — suggested searches (operator 2026-10-07: a beta tester could
         not tell). -->
    <div v-if="heroTopics.length" data-testid="home-topic-chips" class="mt-3 flex flex-wrap items-center gap-2">
      <span class="text-sm font-semibold text-muted" data-testid="home-topic-chips-label">{{ t("ask.tryLabel") }}</span>
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
</template>
