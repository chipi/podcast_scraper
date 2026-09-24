<script setup lang="ts">
/**
 * Resurfacing inbox (P3 Consolidation, #1125 / RFC-101 §5) — your past highlights, resurfaced on a
 * spaced schedule, each with a reflection prompt + one-tap jump back to the moment. Pacing controls
 * (pause/resume) live here. Read-time: the server decides what's due; this just renders + dismisses.
 * Embedded in the Library "Revisit" tab. Auth-gated (empty signed out).
 */
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRoute } from 'vue-router'
import EpisodeRow from '../components/EpisodeRow.vue'
import CheckIcon from '../components/CheckIcon.vue'
import BellOffIcon from '../components/BellOffIcon.vue'
import BookmarkIcon from '../components/BookmarkIcon.vue'
import ConfirmDialog from '../components/ConfirmDialog.vue'
import { useCaptureStore } from '../stores/capture'
import { useSignInGate } from '../composables/useSignInGate'
import ShowAllToggle from '../components/ShowAllToggle.vue'
import { useCappedSections } from '../composables/useCappedSections'
import { summaryFromDetail } from '../utils/episode'
import {
  getEpisode,
  getPlaybackList,
  getResurfacing,
  markSurfaced,
  putResurfacingSettings,
  retireHighlight,
} from '../services/api'
import { useResurfacingStore } from '../stores/resurfacing'
import type { EpisodeDetail, EpisodeSummary, ResurfacingItem } from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { formatPublishDate } from '../utils/format'
import { borderClass } from '../utils/highlightColors'
import { scrollBehavior } from '../utils/motion'

const { t, locale } = useI18n()
const route = useRoute()
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

/**
 * The episode each group is for (slug → detail), so the group header can be the real
 * `EpisodeCard` — artwork included — rather than a line of text (operator 2026-09-17).
 *
 * The whole detail is kept, not just the title: it already carries the artwork, show name, feed id
 * and date the card needs, so the card costs no extra request beyond the one this view was already
 * making for the heading.
 */
const details = ref<Record<string, EpisodeDetail>>({})

/**
 * When the listener last played each episode (slug → unix seconds), from their playback positions.
 *
 * Asked for directly (operator 2026-09-17): a moment resurfaces weeks after the fact, and "when did
 * I hear this" is the context that tells you whether you are revisiting something recent or
 * something you half-remember. `updated_at` is the last time the position moved, which is the last
 * time it was listened to. No new endpoint — this is the same list the queue's "recently played"
 * reads.
 */
const listenedAt = ref<Record<string, number>>({})

const loadError = ref(false)

async function load(): Promise<void> {
  // A throw here used to leave `loaded` false FOREVER: neither the items branch nor the empty-state
  // branch renders, so the tab showed its intro line and a pause button over nothing. Indis-
  // tinguishable from "you have no captures" — and the route 503s whenever the corpus is briefly
  // unavailable, so a restart blanked Revisit for everyone with no retry (review 2026-09-18).
  loadError.value = false
  try {
    const resp = await getResurfacing()
    items.value = resp.items
    paused.value = resp.paused
  } catch {
    loadError.value = true
    return
  } finally {
    loaded.value = true
  }
  void hydrateEpisodes()
  void hydrateListenedAt()
}

/** Tolerated like the episode hydration: no playback history just means no "listened" line. */
async function hydrateListenedAt(): Promise<void> {
  const positions = await getPlaybackList().catch(() => [])
  const next: Record<string, number> = {}
  for (const p of positions) {
    if (p.updated_at) next[p.slug] = p.updated_at
  }
  listenedAt.value = next
}

/**
 * Resolve the episode each due item came from.
 *
 * Not awaited by `load`: the list is useful before the episodes arrive (each group falls back to a
 * slug-titled card), and one unresolvable episode must not hold up the rest. Failures are silent per
 * slug for the same reason.
 */
async function hydrateEpisodes(): Promise<void> {
  const slugs = [...new Set(items.value.map((i) => i.highlight.episode_slug))].filter(
    (s) => s && !details.value[s],
  )
  await Promise.all(
    slugs.map(async (slug) => {
      const d = await getEpisode(slug).catch(() => null)
      if (d) details.value[slug] = d
    }),
  )
}

interface RevisitGroup {
  slug: string
  episode: EpisodeSummary
  items: ResurfacingItem[]
}

/**
 * The group header's episode, in the shape the shared card takes.
 *
 * Built through `summaryFromDetail` (the one adapter Queue / Recent / Saved use) with the SUMMARY
 * fields cleared: this is a review surface, and the episode's prose summary would compete with the
 * captured moment that is the actual content — the same reason Search's groups carry no summary.
 * Before the detail arrives, or if it never does, the card still renders with the slug as its title
 * rather than collapsing the group.
 */
function groupEpisode(slug: string): EpisodeSummary {
  const d = details.value[slug]
  const base = d
    ? summaryFromDetail(d)
    : ({
        slug,
        title: slug,
        feed_id: null,
        podcast_title: null,
        publish_date: null,
        duration_seconds: null,
        artwork_url: null,
        episode_image_url: null,
        feed_image_url: null,
        status: 'ready',
      } as unknown as EpisodeSummary)
  return { ...base, summary_preview: null, summary_text: null, summary_bullets: [] }
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
    episode: groupEpisode(slug),
    items: list,
  }))
})

// Same capped sections + "Show all (N)" as the notes list on Boards (operator 2026-09-17): the due
// list is unbounded, so it gets the same paging rule rather than growing without limit.
const caps = useCappedSections()

/**
 * Episode groups the user has folded away — the same idiom as Library → Saved (operator
 * 2026-09-18). Per-view and collapsed-by-exception: the moments are the content, so hiding them is
 * a choice rather than a default.
 */
const collapsed = ref<Set<string>>(new Set())
function toggleGroup(slug: string): void {
  const next = new Set(collapsed.value)
  if (!next.delete(slug)) next.add(slug)
  collapsed.value = next
}
const visibleGroups = computed(() => caps.visible('revisit-groups', groups.value))

/** "Listened 14 Sep 2026" for an episode with playback history; null when never played. */
function listenedLabel(slug: string): string | null {
  const at = listenedAt.value[slug]
  if (!at) return null
  const date = formatPublishDate(new Date(at * 1000).toISOString(), locale.value)
  return date ? t('revisit.listenedOn', { date }) : null
}

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
  // Optimistic, but REVERSIBLE. The card left the screen before the write was awaited and there was
  // no way back: a failed POST left the user believing they had reviewed something they had not,
  // and it reappeared on the next load with no explanation. The resurfacing store has had this
  // rollback since it was written; this view simply did not follow it (review 2026-09-18).
  const before = items.value
  items.value = items.value.filter((i) => i.highlight.id !== item.highlight.id)
  try {
    await markSurfaced(item.highlight.id)
  } catch {
    items.value = before
    return
  }
  // Keep the nav badge honest: reviewing an item is exactly when the count should drop.
  void resurfacing.load()
}

/**
 * Keep it, stop asking (operator 2026-09-18).
 *
 * Dropped from the list exactly like a review, because from here the two look the same — the item
 * leaves this surface. What differs is the server state: reviewing advances the ladder so it
 * returns later, retiring takes it off the ladder for good. Neither touches the capture.
 */
/**
 * Scroll to the capture Home sent us to (`?focus=<id>`), and ring it (operator 2026-09-18).
 *
 * Home's rail links HERE rather than to the player: from Home the user is choosing what to do
 * with a capture, and the three outcomes live on this card. Landing at the top of a long list
 * with no idea which item was tapped is the thing that would make the rail feel broken.
 *
 * The ring is presentation only and is dropped on the first interaction, so it marks "this is the
 * one" without becoming a second kind of selected state the user has to dismiss.
 */
const focusId = ref<string | null>(null)
const cardEls = new Map<string, HTMLElement>()

function registerCard(id: string, el: unknown): void {
  if (el instanceof HTMLElement) cardEls.set(id, el)
  else cardEls.delete(id)
}

watch(
  () => [route.query.focus, items.value.length] as const,
  async ([focus]) => {
    const id = typeof focus === 'string' ? focus : null
    if (!id || !items.value.some((i) => i.highlight.id === id)) return
    focusId.value = id
    await nextTick()
    // `center`, not `start`: a card scrolled to the very top sits under the sticky masthead.
    cardEls.get(id)?.scrollIntoView({ block: 'center', behavior: scrollBehavior() })
  },
  { immediate: true },
)

async function retire(item: ResurfacingItem): Promise<void> {
  const before = items.value
  items.value = items.value.filter((i) => i.highlight.id !== item.highlight.id)
  try {
    await retireHighlight(item.highlight.id)
  } catch {
    items.value = before
    return
  }
  void resurfacing.load()
}

/**
 * Deleting the capture itself — the fourth outcome, and the only destructive one.
 *
 * Confirm-gated per #1594: it destroys something the user WROTE, along with any notes on it, and
 * the create endpoint mints new ids so there is no undo to offer. The capture store owns the
 * cascade, so this goes through it rather than calling the endpoint directly.
 */
const pendingDelete = ref<string | null>(null)
const capture = useCaptureStore()
const { gated } = useSignInGate()
async function confirmDelete(): Promise<void> {
  const id = pendingDelete.value
  pendingDelete.value = null
  if (!id) return
  // Gated (#1590): signed out this write returns 401, the store swallows it, and the card would
  // disappear from the list and then reappear — which reads as the user's own action failing.
  await gated(async () => {
    // Restored on failure like the other two. The capture store rolls `highlights` back on a
    // permanent error, so without this the two surfaces disagree: the capture is gone from Revisit
    // and still present in Saved, in the same session (review 2026-09-18).
    const before = items.value
    items.value = items.value.filter((i) => i.highlight.id !== id)
    try {
      await capture.remove(id)
    } catch {
      items.value = before
      return
    }
    void resurfacing.load()
  })()
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

    <!-- An error is NOT an empty state. Saying "nothing due" when the request failed tells the user
         something false about their own captures. -->
    <div v-if="loadError" class="text-muted" data-testid="revisit-load-error">
      <p>{{ t('revisit.loadError') }}</p>
      <button
        type="button"
        class="lp-tap mt-2 rounded-full border border-border px-3 py-1 text-sm font-bold text-accent"
        data-testid="revisit-retry"
        @click="load"
      >
        {{ t('common.retry') }}
      </button>
    </div>
    <p v-else-if="paused" class="text-muted">{{ t('revisit.paused') }}</p>
    <p v-else-if="loaded && !items.length" class="text-muted">{{ t('revisit.empty') }}</p>

    <!-- Revisit IS Saved, in a different context (operator 2026-09-18): the same captures, surfaced
         because they are due rather than because you went looking. So it renders the same way —
         episode heading, fold control, flat list of captures — and only the framing differs: a
         reflection prompt on each, "Mark reviewed" instead of the Saved row's edit controls. -->
    <template v-else-if="!loadError">
      <section v-for="g in visibleGroups" :key="g.slug" class="mb-6" data-testid="revisit-group">
        <!-- Same shape as Library → Saved (operator 2026-09-18): the shared `EpisodeRow` as the
             heading, with the fold control in its `#trailing` slot, and the moments as a flat list
             beneath it.

             This replaced a card-in-a-card-in-a-card: an outer bordered container per group, a
             toggle row of its own, and then a bordered box per moment. Three nested frames to say
             "these four moments are from this episode", when a heading and a list say it with one. -->
        <div class="mb-2">
          <EpisodeRow :episode="g.episode">
            <template #trailing>
              <button
                type="button"
                class="lp-tap shrink-0 self-center rounded-full px-2 py-1 text-xs font-bold text-accent"
                :aria-expanded="!collapsed.has(g.slug)"
                :aria-label="
                  collapsed.has(g.slug)
                    ? t('highlights.expandGroup', { title: g.episode.title })
                    : t('highlights.collapseGroup', { title: g.episode.title })
                "
                data-testid="revisit-group-collapse"
                @click="toggleGroup(g.slug)"
              >{{ collapsed.has(g.slug) ? '▼' : '▲' }}</button>
            </template>
          </EpisodeRow>
          <!-- WHEN this episode was listened to, and how many moments are due — one muted line
               under the row rather than two slots inside a card. Absent when there is no playback
               history rather than guessed at from the capture date. -->
          <p class="lp-kicker mt-1">
            <span v-if="listenedLabel(g.slug)" data-testid="revisit-listened">{{
              listenedLabel(g.slug)
            }}</span>
            <template v-if="listenedLabel(g.slug)"> · </template>
            <span>{{ t('revisit.momentCount', g.items.length) }}</span>
          </p>
        </div>
        <ul v-show="!collapsed.has(g.slug)" class="flex flex-col gap-3">
          <!-- The same card frame Saved uses, colour stripe included: it is the same capture, so
               a moment you filed under amber stays amber when it comes back to you. Revisit was
               dropping the colour entirely, which made the two surfaces look like two features. -->
          <li
            v-for="item in g.items"
            :key="item.highlight.id"
            :ref="(el) => registerCard(item.highlight.id, el)"
            class="rounded-xl border border-l-4 border-border p-3 transition"
            :class="[
              borderClass(item.highlight.color),
              focusId === item.highlight.id ? 'ring-2 ring-accent' : '',
            ]"
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
              <!-- Three outcomes, all visible, in the app's 32px circle idiom (operator
                   2026-09-18) — the same `lp-tap h-8 w-8 rounded-full border border-border` shape
                   FavoriteButton and the Saved cards use, so these read as controls the user has
                   already met. Icons are DRAWN, never characters: CloseIcon records why.

                   All three are OUTLINES. The tick was a filled accent disc to mark it as the
                   primary, but a filled tick is the universal "this is done" marker — a state, and
                   the same mistake as labelling the button "✓ Reviewed" (operator 2026-09-18).
                   There is no reviewed state to show here anyway: pressing it removes the card from
                   the list, so a reviewed item is never on this screen. Emphasis comes from the
                   tick's accent COLOUR instead, which says "press this" without claiming done. -->
              <span class="ms-auto flex items-center gap-2">
                <button
                  type="button"
                  class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-accent text-accent transition hover:bg-accent/10"
                  data-testid="revisit-dismiss"
                  :aria-label="t('revisit.dismiss')"
                  :title="t('revisit.dismiss')"
                  @click="dismiss(item)"
                ><CheckIcon /></button>
                <button
                  type="button"
                  class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground"
                  data-testid="revisit-retire"
                  :aria-label="t('revisit.retire')"
                  :title="t('revisit.retire')"
                  @click="retire(item)"
                ><BellOffIcon /></button>
                <!-- The FILLED bookmark, not a ✕ (operator 2026-09-18). This action is an UNSAVE,
                     and an unsave should show the glyph that did the saving, filled, so that tapping
                     it reads as undoing the save rather than as a generic delete.

                     Bookmark and not a heart because of which save it was: the heart
                     (`FavoriteButton`) takes episodes, people, topics, shows and storylines, while
                     everything on this screen is a CAPTURE, saved with the transcript's bookmark —
                     `types.ts` puts it plainly, "an insight is a capture, saved via the highlights
                     path, never a favorite" (RFC-121 / #1593).

                     Accent by default, danger on hover: accent is the saved state it currently
                     shows; danger is what pressing it does. Still destructive, so it still asks
                     first — the #1594 rule, via the same ConfirmDialog the Saved list opens for this
                     very object. -->
                <button
                  type="button"
                  class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-accent transition hover:text-danger"
                  data-testid="revisit-delete"
                  :aria-label="t('revisit.remove')"
                  :title="t('revisit.remove')"
                  @click="pendingDelete = item.highlight.id"
                ><BookmarkIcon filled /></button>
              </span>
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

    <ConfirmDialog
      :open="pendingDelete !== null"
      :title="t('highlights.confirmDeleteTitle')"
      :body="t('highlights.confirmDeleteBody')"
      :confirm-label="t('highlights.confirmDelete')"
      data-testid="revisit-delete-confirm"
      @confirm="confirmDelete"
      @cancel="pendingDelete = null"
    />
  </div>
</template>
