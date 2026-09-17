<script setup lang="ts">
/**
 * Person card BODY — the person-specific sections of the entity card: the external bio (photo +
 * "often appears with" + prose), a bio-less person's signals, hosted shows, the episode list,
 * related people/topics, and notes. The shell ({@link EntityCardBody}) owns the back-stack, the
 * header (kicker / title / follow / save / dismiss) and the load; this just renders the loaded
 * `PersonCard`. Graph navigation (tapping a related chip / a signal) emits `open`; `close` dismisses
 * the whole card (the shell re-emits it upward).
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue"
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
// Bio clamp, driven by the photo column (lp-media-*). Defaults to clipped so a "Show more" is
// never hidden before layout has happened.
const bioEl = ref<HTMLElement | null>(null)
const bioExpanded = ref(false)
const bioClipped = ref(true)

function measureBio(): void {
  const el = bioEl.value
  if (!el || bioExpanded.value) return // expanded: the window no longer constrains anything
  const prose = el.firstElementChild
  if (!prose || el.clientHeight === 0) return
  // The PROSE against the WINDOW — see EpisodeCard.measureSummary for why the reverse never fired.
  bioClipped.value = prose.scrollHeight - el.clientHeight > 1
}

onMounted(() => {
  measureBio()
  if (typeof ResizeObserver !== "undefined" && bioEl.value) {
    const ro = new ResizeObserver(() => measureBio())
    ro.observe(bioEl.value)
    onBeforeUnmount(() => ro.disconnect())
  }
})

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
const relatedTopics = computed<Topic[]>(() => props.person.related_topics ?? [])

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
            ><span v-else-if="i === hostShows.length - 2"> {{ t("ec.andJoin") }} </span>
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
      ><span v-else-if="i === hostShows.length - 2"> {{ t("ec.andJoin") }} </span>
    </template>
  </p>


  <!-- "Often appears with" + signals — our own derived data, so it follows the sourced block and
       the shows rather than sitting between them. Rendered once for both the bio and bio-less
       cases; it used to be duplicated across two branches. -->
  <EntitySignals kind="person" :id="person.id" @open="(p) => emit('open', p)" />


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
        <EpisodeRow :episode="e" />
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

  <!-- Search transcripts — placed BETWEEN related people and related topics so it separates the two
       chip groups (operator 2026-09-17). They are both rows of pills and ran together visually;
       the pill-shaped button breaks them apart while staying useful where it sits. Content-width,
       never full-bleed. -->
  <button
    type="button"
    class="mb-4 block w-fit max-w-full rounded-full border border-border px-4 py-2 text-left text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
    data-testid="ec-search-library"
    @click="searchLibrary"
  >
    {{ t("ec.searchLibrary", { term: label }) }}
  </button>

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
