<script setup lang="ts">
/**
 * The four kinds of interest, one section each — Topics, People, Themes, Storylines (beta feedback
 * 2026-10-04). Shared by the Profile Interests tab and the onboarding picker.
 *
 * It replaced one mixed strip plus a picker that offered only the top 12 themes and storylines:
 * people and plain topics could be followed from an entity card and nowhere else, so the screen
 * that edits interests could not ADD most of what it listed. Each section now shows what is
 * followed, suggests what is trending, and searches everything of its kind.
 *
 * It never writes. It reports `toggle` and the parent decides what that means — Profile persists
 * each tap through the interests store, the picker keeps a local selection until Save. One
 * component, two persistence contracts, and neither leaks into the other.
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue"
import { useI18n } from "vue-i18n"
import CheckIcon from "./CheckIcon.vue"
import CloseIcon from "./CloseIcon.vue"
import { getStorylines, getTopClusters, getTrending, searchInterests } from "../services/api"
import type { InterestHit, TrendingEntity } from "../services/types"
import { dedupeByLabel, interestKind, interestLabel, type InterestKind } from "../utils/interests"

const props = withDefaults(
  defineProps<{
    /** Followed tokens, all kinds mixed — each section takes its own. */
    selected: string[]
    /** Whether a followed chip's label opens its card (Profile) or is plain text (the picker). */
    openable?: boolean
  }>(),
  { openable: false }
)
const emit = defineEmits<{
  (e: "toggle", id: string): void
  (e: "open", target: { kind: InterestKind; id: string }): void
}>()
const { t } = useI18n()

const KINDS: InterestKind[] = ["topic", "person", "theme", "storyline"]
/** How many suggestions a section shows once followed ones are taken out. */
const SHOWN = 8
/** Shorter queries match most of the corpus, which is a list, not an answer. */
const MIN_QUERY = 2

// Real labels win over de-slugged ids, and a storyline OPENS on its anchor topic, never on its
// `thc:` id. Both maps fill from every source the sections read, so a chip followed via search
// keeps its proper label after the search box is cleared.
const labels = reactive(new Map<string, string>())
const anchors = reactive(new Map<string, string>())
function learn(id: string, label: string, anchor?: string | null): void {
  if (label) labels.set(id, label)
  if (anchor) anchors.set(id, anchor)
}

const suggestions = reactive<Record<InterestKind, string[]>>({
  topic: [],
  person: [],
  theme: [],
  storyline: [],
})
const loading = ref(true)

interface SearchState {
  q: string
  hits: string[]
  busy: boolean
  failed: boolean
  seq: number
}
const blank = (): SearchState => ({ q: "", hits: [], busy: false, failed: false, seq: 0 })
const search = reactive<Record<InterestKind, SearchState>>({
  topic: blank(),
  person: blank(),
  theme: blank(),
  storyline: blank(),
})
const timers = new Map<InterestKind, ReturnType<typeof setTimeout>>()

const selectedSet = computed(() => new Set(props.selected))

const following = computed(() => {
  const out: Record<InterestKind, string[]> = { topic: [], person: [], theme: [], storyline: [] }
  for (const id of props.selected) out[interestKind(id)].push(id)
  // Within a section only: `topic:x` and `tc:x` read the same but are different kinds, and each
  // belongs in its own section rather than one hiding the other.
  for (const k of KINDS) out[k] = dedupeByLabel(out[k], labels)
  return out
})

function label(id: string): string {
  return interestLabel(id, labels)
}

/** Where a followed chip leads, or null when it leads nowhere — see ProfileView's original note:
 *  a theme has no card destination yet, and a storyline without a resolved anchor cannot open. */
function openTarget(id: string): string | null {
  const kind = interestKind(id)
  if (kind === "theme") return null
  if (kind === "storyline") return anchors.get(id) ?? null
  return id
}

function onOpen(id: string): void {
  const target = openTarget(id)
  if (target) emit("open", { kind: interestKind(id), id: target })
}

function visibleSuggestions(kind: InterestKind): string[] {
  return suggestions[kind].filter((id) => !selectedSet.value.has(id)).slice(0, SHOWN)
}

function onQuery(kind: InterestKind, value: string): void {
  const s = search[kind]
  s.q = value
  const pending = timers.get(kind)
  if (pending) clearTimeout(pending)
  if (value.trim().length < MIN_QUERY) {
    s.seq++ // drop any answer still in flight for a longer query
    s.hits = []
    s.busy = false
    s.failed = false
    return
  }
  s.busy = true
  timers.set(
    kind,
    setTimeout(() => void runSearch(kind, value.trim()), 250)
  )
}

async function runSearch(kind: InterestKind, q: string): Promise<void> {
  const s = search[kind]
  const mine = ++s.seq
  try {
    const hits: InterestHit[] = await searchInterests(kind, q)
    if (mine !== s.seq) return // a newer keystroke owns the box now
    for (const h of hits) learn(h.id, h.label, h.anchor_topic_id)
    s.hits = hits.map((h) => h.id)
    s.failed = false
  } catch {
    if (mine !== s.seq) return
    s.hits = []
    s.failed = true
  } finally {
    if (mine === s.seq) s.busy = false
  }
}

function fromTrending(rows: TrendingEntity[]): string[] {
  for (const r of rows) learn(r.entity_id, r.label, r.anchor_topic_id)
  return rows.map((r) => r.entity_id)
}

onMounted(async () => {
  // Trending is the suggestion; the plain top lists are the fallback for a corpus with no momentum
  // yet, AND the label source for followed themes and storylines that are not trending right now.
  const [topics, people, themes, stories, topThemes, topStories] = await Promise.all([
    getTrending("topic", "corpus", 20).catch(() => [] as TrendingEntity[]),
    getTrending("person", "corpus", 20).catch(() => [] as TrendingEntity[]),
    getTrending("theme", "corpus", 20).catch(() => [] as TrendingEntity[]),
    getTrending("storyline", "corpus", 20).catch(() => [] as TrendingEntity[]),
    getTopClusters(50).catch(() => []),
    getStorylines(50).catch(() => []),
  ])
  for (const c of topThemes) learn(c.id, c.label)
  for (const st of topStories) learn(st.id, st.label, st.anchor_topic_id)
  suggestions.topic = fromTrending(topics)
  suggestions.person = fromTrending(people)
  suggestions.theme = themes.length ? fromTrending(themes) : topThemes.map((c) => c.id)
  suggestions.storyline = stories.length ? fromTrending(stories) : topStories.map((st) => st.id)
  loading.value = false
})

onBeforeUnmount(() => {
  for (const timer of timers.values()) clearTimeout(timer)
})

/**
 * Each kind keeps the pill it had on the Profile strip this replaced (operator 2026-10-04: "look
 * like this"): a storyline in the accent on an accent tint, a theme and a person outlined in their
 * own hue, a topic filled. Followed, suggested and found items all wear their kind's pill, so the
 * colour teaches the kind; what differs between them is the mark — ×, + or ✓.
 */
function kindPill(kind: InterestKind): string {
  return {
    storyline: "bg-accent/15 font-semibold text-accent",
    person: "bg-overlay text-person ring-1 ring-inset ring-person/30",
    theme: "bg-overlay text-theme ring-1 ring-inset ring-theme/30",
    topic: "bg-overlay text-topic",
  }[kind]
}
</script>

<template>
  <div class="space-y-4">
    <section
      v-for="kind in KINDS"
      :key="kind"
      class="rounded-2xl border border-border p-5"
      :data-testid="`interests-section-${kind}`"
    >
      <h2 class="lp-section">{{ t(`interestSections.heading_${kind}`) }}</h2>
      <p class="mb-3 text-sm text-muted">{{ t(`interestSections.hint_${kind}`) }}</p>

      <!-- Following: tap the label to open it (Profile), × to stop following. -->
      <div v-if="following[kind].length" class="mb-4 flex flex-wrap gap-1.5">
        <span
          v-for="id in following[kind]"
          :key="id"
          class="inline-flex items-center rounded-full text-xs"
          :class="kindPill(kind)"
          :data-testid="`interest-following-${kind}`"
        >
          <button
            v-if="openable && openTarget(id)"
            type="button"
            class="lp-tap py-1 pl-2.5 pr-1 hover:brightness-125"
            :aria-label="t('profile.openInterest', { label: label(id) })"
            data-testid="interest-open"
            @click="onOpen(id)"
          >
            {{ label(id) }}
          </button>
          <span v-else class="py-1 pl-2.5 pr-1">{{ label(id) }}</span>
          <button
            type="button"
            class="lp-tap rounded-full p-1.5 opacity-70 hover:opacity-100"
            :aria-label="t('interestSections.remove', { name: label(id) })"
            data-testid="interest-remove"
            @click="emit('toggle', id)"
          >
            <CloseIcon :size="12" />
          </button>
        </span>
      </div>
      <p v-else class="mb-4 text-sm text-muted" data-testid="interest-none">
        {{ t(`interestSections.none_${kind}`) }}
      </p>

      <input
        :value="search[kind].q"
        type="search"
        :placeholder="t(`interestSections.search_${kind}`)"
        :aria-label="t(`interestSections.search_${kind}`)"
        :data-testid="`interest-search-${kind}`"
        class="lp-search mb-3 w-full rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
        @input="onQuery(kind, ($event.target as HTMLInputElement).value)"
      />

      <!-- Search results replace the suggestions while there is a query: one list at a time. -->
      <template v-if="search[kind].q.trim().length >= MIN_QUERY">
        <p v-if="search[kind].busy" class="text-sm text-muted">{{ t("interestSections.searching") }}</p>
        <p v-else-if="search[kind].failed" class="text-sm text-muted" data-testid="interest-search-failed">
          {{ t("interestSections.searchFailed") }}
        </p>
        <p v-else-if="!search[kind].hits.length" class="text-sm text-muted" data-testid="interest-no-match">
          {{ t("interestSections.noMatch", { q: search[kind].q.trim() }) }}
        </p>
        <div v-else class="flex flex-wrap gap-1.5">
          <button
            v-for="id in search[kind].hits"
            :key="id"
            type="button"
            :aria-pressed="selectedSet.has(id)"
            :aria-label="
              t(selectedSet.has(id) ? 'interestSections.remove' : 'interestSections.follow', {
                name: label(id),
              })
            "
            class="lp-tap inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs transition"
            :class="kindPill(kind)"
            data-testid="interest-result"
            @click="emit('toggle', id)"
          >
            <CheckIcon v-if="selectedSet.has(id)" :size="12" />
            <span v-else aria-hidden="true">+</span>
            {{ label(id) }}
          </button>
        </div>
      </template>
      <template v-else>
        <p class="mb-1.5 font-mono text-[10px] uppercase tracking-wide text-muted">
          {{ t("interestSections.suggested") }}
        </p>
        <p v-if="loading" class="text-sm text-muted">{{ t("interests.loading") }}</p>
        <div v-else-if="visibleSuggestions(kind).length" class="flex flex-wrap gap-1.5">
          <button
            v-for="id in visibleSuggestions(kind)"
            :key="id"
            type="button"
            :aria-label="t('interestSections.follow', { name: label(id) })"
            class="lp-tap inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs transition"
            :class="kindPill(kind)"
            data-testid="interest-suggestion"
            @click="emit('toggle', id)"
          >
            <span aria-hidden="true">+</span>
            {{ label(id) }}
          </button>
        </div>
        <p v-else class="text-sm text-muted" data-testid="interest-no-suggestions">
          {{ t("interestSections.noSuggestions") }}
        </p>
      </template>
    </section>
  </div>
</template>
