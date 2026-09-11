<script setup lang="ts">
/**
 * Person card BODY — the person-specific sections of the entity card: the external bio (photo +
 * "often appears with" + prose), a bio-less person's signals, hosted shows, the episode list,
 * related people/topics, and notes. The shell ({@link EntityCardBody}) owns the back-stack, the
 * header (kicker / title / follow / save / dismiss) and the load; this just renders the loaded
 * `PersonCard`. Graph navigation (tapping a related chip / a signal) emits `open`; `close` dismisses
 * the whole card (the shell re-emits it upward).
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"
import type { Entity, EpisodeSummary, PersonCard, PersonShow, Topic } from "../services/types"
import EntitySignals from "./EntitySignals.vue"
import ProfileAvatar from "./ProfileAvatar.vue"
import NoteComposer from "./NoteComposer.vue"
import EpisodeRow from "./EpisodeRow.vue"

const props = defineProps<{ person: PersonCard }>()
const emit = defineEmits<{
  (e: "open", payload: { kind: "person" | "topic"; id: string }): void
  (e: "close"): void
}>()
const { t } = useI18n()
const router = useRouter()

const label = computed(() => props.person.label ?? "")
// External bio (wave-G, person_web enricher). Extractive + attributed.
const personWeb = computed(() => props.person.web ?? null)
// Wikimedia's image "Artist" field can carry HTML — render the visible TEXT only.
const photoArtist = computed(() =>
  (personWeb.value?.image_artist ?? "")
    .replace(/<[^>]*>/g, "")
    .replace(/\s+/g, " ")
    .trim()
)
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
const relatedTopics = computed<Topic[]>(() => props.person.related_topics ?? [])

function searchLibrary(): void {
  const term = label.value.trim()
  emit("close")
  if (term) void router.push({ name: "search", query: { q: term } })
}
</script>

<template>
  <!-- Bio block (wave-G): LARGE photo + "often appears with" on the left, prose on the right;
       stacks to one column on narrow screens. Only when the web enricher matched. -->
  <section
    v-if="personWeb"
    class="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-4"
    data-testid="ec-person-bio"
  >
    <div class="sm:w-1/3 sm:shrink-0">
      <ProfileAvatar
        :name="label"
        :src="personWeb.image_url"
        :size="176"
        shape="square"
        data-testid="ec-person-photo"
      />
      <!-- "Often appears with" + signals, set off from the photo so it reads as its own section. -->
      <div class="mt-5">
        <EntitySignals kind="person" :id="person.id" @open="(p) => emit('open', p)" />
      </div>
    </div>
    <div class="min-w-0 sm:flex-1">
      <!-- Pull the first line up by the paragraph's half-leading so the bio's CAP height aligns
           with the TOP of the photo on the 2-col (sm+) layout, not a few px below it. -->
      <p class="text-sm leading-relaxed text-canvas-foreground sm:-mt-1">{{ personWeb.bio }}</p>
      <p class="lp-kicker mt-1">
        <a
          v-if="personWeb.source_url"
          :href="personWeb.source_url"
          target="_blank"
          rel="noopener"
          class="underline"
          >{{ t("ec.bioVia", { source: personWeb.source }) }}</a
        >
        <span v-else>{{ t("ec.bioVia", { source: personWeb.source }) }}</span>
        <span v-if="personWeb.license"> · {{ personWeb.license }}</span>
        <!-- The photo carries its OWN license/credit, distinct from the bio text's. -->
        <span v-if="personWeb.image_license">
          · {{ t("ec.photoLicense", { license: personWeb.image_license }) }}</span
        >
        <span v-if="photoArtist" data-testid="ec-photo-artist">
          · {{ t("ec.photoBy", { artist: photoArtist }) }}</span
        >
      </p>
    </div>
  </section>

  <!-- A bio-less person's signals (co-appearance + consensus) render here rather than in the
       left column above. -->
  <EntitySignals v-if="!personWeb" kind="person" :id="person.id" @open="(p) => emit('open', p)" />

  <!-- Search transcripts for this person. Content-width, never full-bleed. -->
  <button
    type="button"
    class="mb-4 block w-fit max-w-full rounded-full border border-border px-4 py-2 text-left text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
    data-testid="ec-search-library"
    @click="searchLibrary"
  >
    {{ t("ec.searchLibrary", { term: label }) }}
  </button>

  <!-- Shows this person hosts — kept distinct from guest appearances in the episode list below. -->
  <section v-if="hostShows.length" class="mb-4" data-testid="ec-host-shows">
    <h3 class="lp-section mb-2">{{ t("ec.hostOf") }}</h3>
    <div class="flex flex-col">
      <RouterLink
        v-for="s in hostShows"
        :key="s.feed_id"
        :to="{ name: 'podcast', params: { feedId: s.feed_id } }"
        class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
        @click="emit('close')"
      >
        <span class="min-w-0 flex-1 truncate text-sm font-semibold">{{ s.title }}</span>
        <span class="lp-kicker shrink-0">{{
          t("ec.showEpisodeCount", s.episode_count, { named: { count: s.episode_count } })
        }}</span>
      </RouterLink>
    </div>
  </section>

  <!-- Episodes (newest-first, STATED not offered as a control — #2004 item 11). Host-show
       back-catalogue is dropped above, so this is "also appears in" when they host anything. -->
  <section v-if="shownEpisodes.length" class="mb-4">
    <h3 class="lp-section mb-2 flex flex-wrap items-baseline gap-x-2">
      <span>{{
        hostShows.length
          ? t("ec.personOtherEpisodes", shownEpisodes.length, {
              named: { count: shownEpisodes.length },
            })
          : t("ec.personEpisodes", episodeCount, { named: { count: episodeCount } })
      }}</span>
      <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
    </h3>
    <ul class="flex flex-col">
      <li v-for="e in shownEpisodes" :key="e.slug">
        <EpisodeRow :episode="e" @navigate="emit('close')" />
      </li>
    </ul>
  </section>

  <section v-if="relatedPeople.length" class="mb-4">
    <h3 class="lp-section mb-2">{{ t("ec.relatedPeople") }}</h3>
    <div class="flex flex-wrap gap-1.5">
      <button
        v-for="p in relatedPeople"
        :key="p.id"
        type="button"
        class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
        @click="emit('open', { kind: 'person', id: p.id })"
      >
        {{ p.name }}
      </button>
    </div>
  </section>

  <section v-if="relatedTopics.length">
    <h3 class="lp-section mb-2">{{ t("ec.relatedTopics") }}</h3>
    <div class="flex flex-wrap gap-1.5">
      <button
        v-for="tp in relatedTopics"
        :key="tp.id"
        type="button"
        class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
        @click="emit('open', { kind: 'topic', id: tp.id })"
      >
        {{ tp.label }}
      </button>
    </div>
  </section>

  <!-- Notes on this person (PD.4). -->
  <NoteComposer target="person" :target-id="person.id" />
</template>
