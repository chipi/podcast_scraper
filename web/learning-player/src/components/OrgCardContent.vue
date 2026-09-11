<script setup lang="ts">
/**
 * Organization card BODY (#2031) — the org-specific sections of the entity card: where the org is
 * mentioned (episodes) and who/what co-occurs with it (people / other orgs / topics). Leaner than
 * the person body by design — orgs have no web bio/photo. The shell ({@link EntityCardBody}) owns
 * the back-stack, header and load; this renders the loaded `OrgCard`. Graph navigation (tapping a
 * chip) emits `open`; `close` dismisses the whole card.
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import { useRouter } from "vue-router"
import type { Entity, EpisodeSummary, OrgCard, Topic } from "../services/types"
import EpisodeRow from "./EpisodeRow.vue"

const props = defineProps<{ org: OrgCard }>()
const emit = defineEmits<{
  (e: "open", payload: { kind: "person" | "topic" | "organization"; id: string }): void
  (e: "close"): void
}>()
const { t } = useI18n()
const router = useRouter()

const label = computed(() => props.org.label ?? "")
const episodes = computed<EpisodeSummary[]>(() => props.org.episodes ?? [])
const relatedPeople = computed<Entity[]>(() => props.org.related_people ?? [])
const relatedOrgs = computed<Entity[]>(() => props.org.related_orgs ?? [])
const relatedTopics = computed<Topic[]>(() => props.org.related_topics ?? [])

function searchLibrary(): void {
  const term = label.value.trim()
  emit("close")
  if (term) void router.push({ name: "search", query: { q: term } })
}
</script>

<template>
  <!-- Search transcripts for this org. Content-width, never full-bleed. -->
  <button
    type="button"
    class="mb-4 block w-fit max-w-full rounded-full border border-border px-4 py-2 text-left text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
    data-testid="ec-search-library"
    @click="searchLibrary"
  >
    {{ t("ec.searchLibrary", { term: label }) }}
  </button>

  <!-- Episodes mentioning this org (newest-first). -->
  <section v-if="episodes.length" class="mb-4">
    <h3 class="lp-section mb-2 flex flex-wrap items-baseline gap-x-2">
      <span>{{ t("ec.orgEpisodes", episodes.length, { named: { count: episodes.length } }) }}</span>
      <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
    </h3>
    <ul class="flex flex-col">
      <li v-for="e in episodes" :key="e.slug">
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

  <!-- Other organizations mentioned alongside this one — the org↔org co-occurrence the person card
       has no analog for; each chip drills into that org's card in place. -->
  <section v-if="relatedOrgs.length" class="mb-4" data-testid="ec-related-orgs">
    <h3 class="lp-section mb-2">{{ t("ec.relatedOrgs") }}</h3>
    <div class="flex flex-wrap gap-1.5">
      <button
        v-for="o in relatedOrgs"
        :key="o.id"
        type="button"
        class="rounded-full bg-overlay px-2.5 py-1 text-xs text-canvas-foreground transition hover:bg-elevated"
        data-testid="ec-related-org"
        @click="emit('open', { kind: 'organization', id: o.id })"
      >
        {{ o.name }}
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
</template>
