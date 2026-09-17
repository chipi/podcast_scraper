<script setup lang="ts">
/**
 * Resurfacing inbox (P3 Consolidation, #1125 / RFC-101 §5) — your past highlights, resurfaced on a
 * spaced schedule, each with a reflection prompt + one-tap jump back to the moment. Pacing controls
 * (pause/resume) live here. Read-time: the server decides what's due; this just renders + dismisses.
 * Embedded in the Library "Revisit" tab. Auth-gated (empty signed out).
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import ShowAllToggle from '../components/ShowAllToggle.vue'
import { useCappedSections } from '../composables/useCappedSections'
import {
  getEpisode,
  getResurfacing,
  markSurfaced,
  putResurfacingSettings,
} from '../services/api'
import { useResurfacingStore } from '../stores/resurfacing'
import type { ResurfacingItem } from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { formatPublishDate } from '../utils/format'

const { t, locale } = useI18n()
/**
 * The inbox writes THROUGH the store (#2004 item 14 follow-up).
 *
 * The badge exists to pull you back here — so the one flow it must get right is "you came, you
 * reviewed, the badge clears". LibraryView loads the store `onMounted`, but Library is in
 * KEEP_ALIVE_TABS, so that fires once per session: review all your due items and the badge still
 * reads 3 on both navs until the next capture or sign-in. Pausing had the same problem, leaving the
 * badge lit against the store's own paused rule.
 */
const resurfacing = useResurfacingStore()

const items = ref<ResurfacingItem[]>([])
const paused = ref(false)
const loaded = ref(false)

/** Episode titles for the group headings (slug → title); the slug is the fallback. */
const titles = ref<Record<string, string>>({})

async function load(): Promise<void> {
  const resp = await getResurfacing()
  items.value = resp.items
  paused.value = resp.paused
  loaded.value = true
  void hydrateTitles()
}

/**
 * Resolve the episode each due item came from.
 *
 * Not awaited by `load`: the list is useful before the titles arrive (each group falls back to its
 * slug), and one unresolvable episode must not hold up the rest. Failures are silent per slug for
 * the same reason.
 */
async function hydrateTitles(): Promise<void> {
  const slugs = [...new Set(items.value.map((i) => i.highlight.episode_slug))].filter(
    (s) => s && !titles.value[s],
  )
  await Promise.all(
    slugs.map(async (slug) => {
      const d = await getEpisode(slug).catch(() => null)
      if (d) titles.value[slug] = d.title
    }),
  )
}

interface RevisitGroup {
  slug: string
  title: string
  items: ResurfacingItem[]
}

/**
 * Due items grouped by the episode they came from (operator 2026-09-17).
 *
 * A flat list of prompts said nothing about WHERE each moment was from, so three moments from one
 * episode read as three unrelated cards. Grouping is the same structure Saved (`HighlightsView`) and
 * Search use — episode heading, then its captures — so the three surfaces stay one pattern.
 *
 * Group order follows the server's due order: the first time an episode appears in `items` fixes its
 * position, so the most-due episode leads and re-rendering cannot reshuffle the page.
 */
const groups = computed<RevisitGroup[]>(() => {
  const bySlug = new Map<string, ResurfacingItem[]>()
  for (const it of items.value) {
    const slug = it.highlight.episode_slug
    const list = bySlug.get(slug) ?? []
    list.push(it)
    bySlug.set(slug, list)
  }
  return [...bySlug.entries()].map(([slug, list]) => ({
    slug,
    title: titles.value[slug] ?? slug,
    items: list,
  }))
})

// Same capped sections + "Show all (N)" as the notes list on Boards (operator 2026-09-17): the due
// list is unbounded, so it gets the same paging rule rather than growing without limit.
const caps = useCappedSections()
const visibleGroups = computed(() => caps.visible('revisit-groups', groups.value))

/** The capture's own date, for the `KIND · DATE` kicker the notes rows use. */
function itemDate(unixSeconds: number): string {
  return formatPublishDate(new Date(unixSeconds * 1000).toISOString(), locale.value) ?? ''
}

/**
 * WHAT kind of capture this is, in words — the kicker half of the notes-style label.
 *
 * Reuses the `highlights.*` wording rather than minting `revisit.*` synonyms: the same moment
 * appears on Saved and here, and it must not be called two different things.
 */
function kindLabel(item: ResurfacingItem): string {
  const k = item.highlight.kind
  if (k === 'insight') return t('highlights.insight')
  if (k === 'span') return t('highlights.span')
  return t('highlights.moment')
}

/** The captured words, or '' for a moment saved before the text was stored. */
function quoteOf(item: ResurfacingItem): string {
  return item.highlight.quote_text?.trim() ?? ''
}

/** Mark a highlight seen → drop it from the list (the server advances its ladder). */
async function dismiss(item: ResurfacingItem): Promise<void> {
  items.value = items.value.filter((i) => i.highlight.id !== item.highlight.id)
  await markSurfaced(item.highlight.id)
  // Keep the nav badge honest: reviewing an item is exactly when the count should drop.
  void resurfacing.load()
}

async function togglePause(): Promise<void> {
  const next = !paused.value
  paused.value = next
  await putResurfacingSettings(next)
  await load() // pausing empties the due list; resuming re-fills it
  // Paused suppresses the badge — but only if the store hears about it.
  void resurfacing.load()
}

/**
 * The jump link's query — timestamp plus the `revisit` marker that advances the ladder (#35).
 *
 * Following the link IS reviewing; the player marks the highlight surfaced on arrival. Before this
 * the only advance path in the whole product was the dismiss button below, so a user who actually
 * revisited — the behaviour the feature exists to produce — never progressed, and kept being shown
 * the same items. Deliberately NOT marked here on click: navigation can be cancelled, and marking
 * on arrival covers Your Week and the digest email with the same mechanism.
 */
function jumpQuery(item: ResurfacingItem): Record<string, string> {
  const ms = item.highlight.start_ms
  return {
    ...(ms != null ? { t: String(Math.floor(ms / 1000)) } : {}),
    revisit: item.highlight.id,
  }
}

onMounted(load)
</script>

<template>
  <div>
    <div class="mb-4 flex items-center justify-between gap-3">
      <p class="text-sm text-muted">{{ t('revisit.intro') }}</p>
      <!-- A CD-player transport button (operator 2026-09-14): a big square play/pause. Running → a
           pause glyph (press to pause); paused → a play glyph (press to resume). -->
      <button
        type="button"
        class="lp-tap flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border transition hover:bg-overlay"
        data-testid="revisit-pause"
        :aria-pressed="paused"
        :aria-label="paused ? t('revisit.resume') : t('revisit.pause')"
        :title="paused ? t('revisit.resume') : t('revisit.pause')"
        @click="togglePause"
      >
        <svg v-if="paused" viewBox="0 0 24 24" fill="currentColor" class="h-5 w-5" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
        <svg v-else viewBox="0 0 24 24" fill="currentColor" class="h-5 w-5" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z" /></svg>
      </button>
    </div>

    <p v-if="paused" class="text-muted">{{ t('revisit.paused') }}</p>
    <p v-else-if="loaded && !items.length" class="text-muted">{{ t('revisit.empty') }}</p>

    <!-- Grouped by the episode each moment came from (operator 2026-09-17) — the Saved / Search
         structure: an episode heading that opens the episode, then its due captures. -->
    <template v-else>
      <section
        v-for="g in visibleGroups"
        :key="g.slug"
        class="mb-6"
        data-testid="revisit-group"
      >
        <RouterLink
          :to="{ name: 'player', params: { slug: g.slug } }"
          class="lp-section mb-2 block no-underline hover:text-accent"
          data-testid="revisit-group-title"
        >{{ g.title }}</RouterLink>
        <ul class="flex flex-col gap-3">
          <li
            v-for="item in g.items"
            :key="item.highlight.id"
            class="rounded-xl border border-border p-3"
            data-testid="revisit-item"
          >
            <!-- KIND · DATE, the same label the notes rows on Boards carry (operator). "Marked
                 moment" used to stand in as the BODY text, which is why a moment card said nothing
                 about itself — it is the label, and the quote below is the content. -->
            <span class="lp-kicker">{{ kindLabel(item) }} · {{ itemDate(item.highlight.created_at) }}</span>
            <!-- WHAT the moment is about: the captured words. A moment saved before the text was
                 stored has none and shows nothing here rather than a placeholder. -->
            <blockquote
              v-if="quoteOf(item)"
              class="mt-1 border-l-2 border-border pl-2 text-sm italic leading-snug text-canvas-foreground"
              data-testid="revisit-quote"
            >
              {{ quoteOf(item) }}
            </blockquote>
            <p v-if="item.highlight.speaker" class="lp-speaker mt-1 text-xs">
              {{ item.highlight.speaker }}
            </p>
            <!-- The reflection prompt is the QUESTION being asked of the moment, so it sits under
                 the moment rather than above it as the card's title. -->
            <p class="mt-2 text-sm text-muted" data-testid="revisit-prompt">
              {{ item.reflection_prompt }}
            </p>
            <div class="mt-2 flex items-center gap-3">
              <RouterLink
                :to="{ name: 'player', params: { slug: item.highlight.episode_slug }, query: jumpQuery(item) }"
                class="font-mono text-xs font-bold text-accent no-underline"
                data-testid="revisit-jump"
              >▶ {{ item.highlight.start_ms != null ? formatTime(item.highlight.start_ms / 1000) : t('revisit.open') }}</RouterLink>
              <button
                type="button"
                class="text-xs text-muted transition hover:text-canvas-foreground"
                @click="dismiss(item)"
              >{{ t('revisit.dismiss') }}</button>
            </div>
          </li>
        </ul>
      </section>
      <ShowAllToggle
        v-if="caps.overflows(groups.length)"
        :expanded="caps.expanded.has('revisit-groups')"
        :count="groups.length"
        @toggle="caps.toggle('revisit-groups')"
      />
    </template>
  </div>
</template>
