<script setup lang="ts">
/**
 * Person card BODY — the person-specific sections of the entity card: the external bio (photo +
 * "often appears with" + prose), a bio-less person's signals, hosted shows, the episode list,
 * related people/topics, and notes. The shell ({@link EntityCardBody}) owns the back-stack, the
 * header (kicker / title / follow / save / dismiss) and the load; this just renders the loaded
 * `PersonCard`. Graph navigation (tapping a related chip / a signal) emits `open`; `close` dismisses
 * the whole card (the shell re-emits it upward).
 */
import CollapsibleSection from "./CollapsibleSection.vue"
import { computed, ref } from "vue"
import { useClampedProse } from "../composables/useClampedProse"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"
import type { Entity, EpisodeSummary, PersonCard, PersonShow, Topic } from "../services/types"
import { personName } from "../utils/personName"
import EntitySignals from "./EntitySignals.vue"
import ProfileAvatar from "./ProfileAvatar.vue"
import NoteComposer from "./NoteComposer.vue"
import EntityEpisodeList from "./EntityEpisodeList.vue"
import ThemeCard from "./ThemeCard.vue"
import StorylineCard from "./StorylineCard.vue"

const props = defineProps<{
  person: PersonCard
  /** This card's stack depth; a theme or storyline it opens sits one level deeper. */
  depth?: number
}>()
const emit = defineEmits<{
  (e: "open", payload: { kind: "person" | "topic"; id: string }): void
  (e: "close"): void
}>()
const { t } = useI18n()
const router = useRouter()

// The person page's own title. Cased here so the heading, the share card and the document title
// all read the same thing.
const label = computed(() => personName(props.person.label ?? ""))
// External bio (wave-G, person_web enricher). Extractive + attributed.
const personWeb = computed(() => props.person.web ?? null)
// Wikimedia's image "Artist" field can carry HTML — render the visible TEXT only.
// Bio clamp, driven by the photo column (lp-media-*). Defaults to clipped so a "Show more" is
// never hidden before layout has happened.
const bioEl = ref<HTMLElement | null>(null)
const bioExpanded = ref(false)

// Shared measurement — see `useClampedProse`. This file's copy observed via `onMounted`, which
// misses a window that only appears in a second render, so the bio could sit stuck on its safe
// "clipped" default forever.
const { clipped: bioClipped } = useClampedProse(bioEl, bioExpanded)

const photoArtist = computed(() =>
  (personWeb.value?.image_artist ?? "")
    .replace(/<[^>]*>/g, "")
    .replace(/\s+/g, " ")
    .trim()
)
// The complete credit — source, text licence, photo licence, photo artist — shown on the
// attribution's `title` so nothing is LOST by the compact rendering above, only folded away.
const attributionFull = computed(() => {
  const w = personWeb.value
  if (!w) return ""
  const parts = [t("ec.bioVia", { source: w.source })]
  if (w.license) parts.push(w.license)
  if (w.image_license) parts.push(t("ec.photoLicense", { license: w.image_license }))
  if (photoArtist.value) parts.push(t("ec.photoBy", { artist: photoArtist.value }))
  return parts.join(" · ")
})

const episodes = computed<EpisodeSummary[]>(() => props.person.episodes ?? [])
const episodeCount = computed(() => props.person.episode_count ?? 0)
// Per-show role (#3): a person hosts some shows and guests on others. Surface the shows they HOST,
// and drop those shows' back-catalogue from the episode list — show OTHER-show appearances there.
const hostShows = computed<PersonShow[]>(() =>
  (props.person.shows ?? []).filter((s) => (s.role ?? "").toLowerCase() === "host")
)
const hostFeedIds = computed(() => new Set(hostShows.value.map((s) => s.feed_id)))
const shownEpisodes = computed<EpisodeSummary[]>(() =>
  hostShows.value.length
    ? episodes.value.filter((e) => !hostFeedIds.value.has(e.feed_id))
    : episodes.value
)
const relatedPeople = computed<Entity[]>(() => props.person.related_people ?? [])

/**
 * Localise a speaker role for the co-appearance chips — the SAME map and keys `KnowledgePanel` and
 * `PodcastSignalsBand` use, so the three surfaces that list people cannot drift on what a role is
 * called. An unrecognised role falls through to its raw string rather than vanishing; an ABSENT one
 * returns "" and the caller renders no badge.
 */
const ROLE_LABEL_KEYS: Record<string, string> = {
  host: "ec.roleHost",
  guest: "ec.roleGuest",
  mentioned: "ec.roleMentioned",
}
function roleLabel(role: string | null | undefined): string {
  if (!role) return ""
  const key = ROLE_LABEL_KEYS[role.toLowerCase()]
  return key ? t(key) : role
}
const relatedTopics = computed<Topic[]>(() => props.person.related_topics ?? [])
// The themes and storylines those topics belong to, each once, in the order first met. A storyline
// opens from ANY member topic id (StorylineCard reconstructs it from one), so keep the first.
const relatedThemes = computed(() => {
  const seen = new Map<string, { id: string; label: string }>()
  for (const tp of relatedTopics.value)
    if (tp.cluster_id && tp.cluster_label && !seen.has(tp.cluster_id))
      seen.set(tp.cluster_id, { id: tp.cluster_id, label: tp.cluster_label })
  return [...seen.values()]
})
const relatedStorylines = computed(() => {
  const seen = new Map<string, { id: string; label: string; topicId: string }>()
  for (const tp of relatedTopics.value)
    if (tp.storyline_id && tp.storyline_label && !seen.has(tp.storyline_id))
      seen.set(tp.storyline_id, { id: tp.storyline_id, label: tp.storyline_label, topicId: tp.id })
  return [...seen.values()]
})
const themeOpenId = ref<string | null>(null)
const storylineOpenTopicId = ref<string | null>(null)

function searchLibrary(): void {
  const term = label.value.trim()
  emit("close")
  if (term) void router.push({ name: "search", query: { q: term } })
}
</script>

<template>
  <!-- Bio block (wave-G): photo with the "who is this" descriptor BESIDE it, then the prose.
       Only when the web enricher matched.

       The photo and descriptor stay side by side at every width (operator 2026-09-17). The photo
       used to be 176px and full-width-stacked on mobile, which is the only layout a phone ever
       got: a large square pushing the bio below the fold, with the descriptor stranded up under
       the name where it read as a competing subtitle. Smaller, and captioned by the descriptor, it
       reads as one identity unit. -->
  <section v-if="personWeb" class="mb-4" data-testid="ec-person-bio">
    <div class="lp-media-row gap-3">
      <!-- LEFT COLUMN: the photo, with the hosted shows directly beneath it — literally under the
           image, not under the whole row (operator 2026-09-17). The bio keeps flowing beside both. -->
      <div class="lp-media-aside w-28">
        <ProfileAvatar
          :name="label"
          :src="personWeb.image_url"
          :size="112"
          shape="square"
          data-testid="ec-person-photo"
        />
        <p v-if="hostShows.length" class="mt-2 text-xs text-muted" data-testid="ec-host-shows">
          {{ t("ec.hostOf") }}
          <template v-for="(s, i) in hostShows" :key="s.feed_id">
            <RouterLink
              :to="{ name: 'podcast', params: { feedId: s.feed_id } }"
              class="font-semibold text-canvas-foreground underline decoration-border underline-offset-2 hover:decoration-current"
              data-testid="ec-host-show-link"
              @click="emit('close')"
              >{{ s.title }}</RouterLink
            ><span v-if="i < hostShows.length - 2">, </span
            ><span v-else-if="i === hostShows.length - 2">{{ ` ${t("ec.andJoin")} ` }}</span>
          </template>
        </p>
        <!-- Attribution, under the hosted shows in the photo's column (operator 2026-09-17).
             Only the SOURCE shows; the licences ride in the title tooltip. Spelled out in full it
             wrapped to six lines in a 112px column — "VIA FIXTURE · CC0-1.0 · PHOTO CC0-1.0 ·
             PHOTO: GENERATED FIXTURE AVATAR" — which made the column taller than the bio beside it,
             and real Wikipedia credits carry longer artist names still. The credit stays VISIBLE
             next to the photo it belongs to; only the licence detail is one press away. -->
        <p v-if="personWeb" class="lp-kicker mt-2" data-testid="ec-person-attribution">
          <!-- The tooltip hangs on the SOURCE itself, not on the photo (operator 2026-09-17):
               provenance belongs to the word that names the provenance, and a tooltip on the image
               is undiscoverable — nothing about a photo suggests it holds licence text. -->
          <a
            v-if="personWeb.source_url"
            :href="personWeb.source_url"
            target="_blank"
            rel="noopener"
            class="underline"
            :title="attributionFull"
            >{{ t("ec.bioVia", { source: personWeb.source }) }}</a
          >
          <span v-else :title="attributionFull">{{
            t("ec.bioVia", { source: personWeb.source })
          }}</span>
        </p>
      </div>
      <!-- The BIO sits beside the photo, at every width. It used to run full-width UNDER a 176px
           square, which on a phone meant the photo owned the first screen on its own and the prose
           started below the fold (operator 2026-09-17). -->
      <!-- The bio is driven by the photo column, not a line count (operator 2026-09-17): it fills
           the height the photo + hosted-shows + attribution stack sets and clips there, with
           "Show more" footed against the bottom of that column. -->
      <div class="lp-media-body">
        <div ref="bioEl" class="lp-media-clip" :class="bioExpanded ? 'lp-media-clip--open' : ''">
          <p
            class="text-sm leading-relaxed text-canvas-foreground"
            data-testid="ec-person-bio-text"
          >
            {{ personWeb.bio }}
          </p>
        </div>
        <button
          v-if="bioClipped || bioExpanded"
          type="button"
          class="lp-media-foot w-fit pt-1 text-xs font-bold text-accent"
          data-testid="ec-person-bio-more"
          @click="bioExpanded = !bioExpanded"
        >
          {{ bioExpanded ? t("podcast.showLess") : t("podcast.showMore") }}
        </button>
      </div>
    </div>
  </section>

  <!-- No external bio → no photo column to sit under, so the hosted shows render here instead.
       Same prose treatment; a heading with one bordered row per show becomes a wall for a prolific
       host. Guest appearances stay in the episode list below. -->
  <p v-if="!personWeb && hostShows.length" class="mb-3 text-sm text-muted" data-testid="ec-host-shows">
    {{ t("ec.hostOf") }}
    <template v-for="(s, i) in hostShows" :key="s.feed_id">
      <RouterLink
        :to="{ name: 'podcast', params: { feedId: s.feed_id } }"
        class="font-semibold text-canvas-foreground underline decoration-border underline-offset-2 hover:decoration-current"
        data-testid="ec-host-show-link"
        @click="emit('close')"
        >{{ s.title }}</RouterLink
      ><span v-if="i < hostShows.length - 2">, </span
      ><span v-else-if="i === hostShows.length - 2">{{ ` ${t("ec.andJoin")} ` }}</span>
    </template>
  </p>


  <!-- ORDER (operator 2026-10-05): often appears with -> related people -> related topics (with
       their themes and storylines) -> where they agree -> episodes -> notes. The signals are our own
       derived data, so they follow the sourced block and the shows; they render in two halves so
       related people and topics sit between them (one request — the signals call is cached). -->
  <EntitySignals kind="person" :id="person.id" only="coappears" @open="(p) => emit('open', p)" />


  <section v-if="relatedPeople.length" class="mb-4">
    <CollapsibleSection :title="t('ec.relatedPeople')" section-key="person-related-people" :level="3">
      <div class="flex flex-wrap gap-1.5">
        <!--
          Role badge, matching the show page and the episode Insights panel (operator 2026-09-27) —
          same markup, same `ec.role*` keys. Three surfaces that list people, one appearance.

          The role is aggregated SERVER-side over the episodes this person shares with the card's
          subject. It could not be read straight off the entity: the builder collects co-appearing
          people last-write-wins, so an unaggregated `role` is whichever shared episode happened to be
          processed last — a co-host would read "mentioned" whenever their final shared episode merely
          mentioned them.

          Roleless renders unbadged rather than guessing "mentioned".
        -->
        <button
          v-for="p in relatedPeople"
          :key="p.id"
          type="button"
          data-testid="ec-related-person"
          :data-role="p.role?.toLowerCase()"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
          @click="emit('open', { kind: 'person', id: p.id })"
        >
          {{ personName(p.name)
          }}<span
            v-if="roleLabel(p.role)"
            data-testid="ec-related-person-role"
            class="ml-1 rounded-full bg-canvas/50 px-1.5 py-0.5 text-[0.6rem] font-bold uppercase tracking-wide"
            >{{ roleLabel(p.role) }}</span
          >
        </button>
      </div>
    </CollapsibleSection>
  </section>


  <!-- Related topics, MIXED with the themes and storylines those topics belong to, each pill in its
       kind's colour and naming its kind — the episode notes' convention for a mixed group (operator
       2026-10-05). Moved up from the foot of the page to sit directly under related people. The
       groupings come from the topics themselves (the server enriches each with its theme and
       storyline), so no extra request. A theme or storyline opens ON TOP, as on the topic card. -->
  <section v-if="relatedTopics.length" class="mb-4" data-testid="ec-person-related">
    <CollapsibleSection :title="t('ec.relatedTopics')" section-key="person-related-topics" :level="3">
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="th in relatedThemes"
          :key="th.id"
          type="button"
          data-testid="ec-person-related-theme"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs font-semibold text-theme ring-1 ring-inset ring-theme/40 transition hover:bg-elevated"
          @click="themeOpenId = th.id"
        >
          <span class="mr-1.5 font-mono text-[10px] uppercase tracking-wide opacity-80">{{ t("kp.themeKind") }}</span>{{ th.label }}
        </button>
        <button
          v-for="sl in relatedStorylines"
          :key="sl.id"
          type="button"
          data-testid="ec-person-related-storyline"
          class="lp-storyline-chip rounded-full px-2.5 py-1 text-xs font-semibold text-storyline transition"
          @click="storylineOpenTopicId = sl.topicId"
        >
          <span class="mr-1.5 font-mono text-[10px] uppercase tracking-wide opacity-80">{{ t("kp.storylineKind") }}</span>{{ sl.label }}
        </button>
        <button
          v-for="tp in relatedTopics"
          :key="tp.id"
          type="button"
          data-testid="ec-person-related-topic"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
          @click="emit('open', { kind: 'topic', id: tp.id })"
        >
          <span class="mr-1.5 font-mono text-[10px] uppercase tracking-wide opacity-80">{{ t("notes.kind_topic") }}</span>{{ tp.label }}
        </button>
      </div>
    </CollapsibleSection>
  </section>

  <!-- Search transcripts — after the related pills, before "Where they agree": a pill-shaped
       button between chip groups and the agreement rows. Content-width, never full-bleed. -->
  <button
    type="button"
    class="mb-4 block w-fit max-w-full rounded-full border border-border px-4 py-2 text-left text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
    data-testid="ec-search-library"
    @click="searchLibrary"
  >
    {{ t("ec.searchLibrary", { term: label }) }}
  </button>

  <!-- "Where they agree" — the second half of the signals, after who this person is connected to. -->
  <EntitySignals kind="person" :id="person.id" :name="label" only="consensus" @open="(p) => emit('open', p)" />

  <!-- Episodes (newest-first, STATED not offered as a control — #2004 item 11). Host-show
       back-catalogue is dropped above, so this is "also appears in" when they host anything.

       BELOW related people and related topics (operator 2026-09-19). It used to sit directly under
       the biography, which put a long list between the two chip groups and the reader: on a
       frequent guest you scrolled past dozens of episode rows to reach two rows of pills. Who this
       person is connected to is the shorter, denser answer, so it comes first; the episodes are the
       archive you descend into, and they sit above the notes where the page bottoms out. -->
  <section v-if="shownEpisodes.length" class="mb-4">
    <CollapsibleSection section-key="person-episodes" :level="3">
      <template #title>
        <span>{{
          hostShows.length
            ? t("ec.personOtherEpisodes", shownEpisodes.length, {
                named: { count: shownEpisodes.length },
              })
            : t("ec.personEpisodes", episodeCount, { named: { count: episodeCount } })
        }}</span>
        <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
      </template>
      <EntityEpisodeList :episodes="shownEpisodes" />
    </CollapsibleSection>
  </section>

  <!-- Notes on this person (PD.4). -->
  <NoteComposer target="person" :target-id="person.id" />

  <!-- A theme or storyline from the related pills, opened ON TOP (teleported sheet) rather than
       navigating away — the topic card's rule, for the same reason (a route change under a top-layer
       sheet reads as a dead tap). -->
  <ThemeCard v-if="themeOpenId" :id="themeOpenId" :depth="(props.depth ?? 0) + 1" @close="themeOpenId = null" />
  <StorylineCard
    v-if="storylineOpenTopicId"
    :id="storylineOpenTopicId"
    :depth="(props.depth ?? 0) + 1"
    @close="storylineOpenTopicId = null"
  />
</template>
