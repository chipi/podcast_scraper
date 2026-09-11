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
// External enrichment (org_web, #2035): description + logo + facts. Absent → the lean card.
const web = computed(() => props.org.web ?? null)
const facts = computed(() =>
  [
    web.value?.founded ? { k: t("ec.orgFounded"), v: web.value.founded } : null,
    web.value?.industry ? { k: t("ec.orgIndustry"), v: web.value.industry } : null,
  ].filter((f): f is { k: string; v: string } => f !== null)
)
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
  <!-- External enrichment (org_web, #2035): logo + description + basic facts + attribution. Only
       shown when the enricher matched; the logo appears only when we host one (often non-free). -->
  <section
    v-if="web"
    class="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-4"
    data-testid="ec-org-web"
  >
    <img
      v-if="web.logo_url"
      :src="web.logo_url"
      :alt="label"
      class="h-16 w-16 shrink-0 rounded-lg object-contain"
      data-testid="ec-org-logo"
    />
    <div class="min-w-0 sm:flex-1">
      <p v-if="web.summary || web.description" class="text-sm leading-relaxed text-canvas-foreground">
        {{ web.summary || web.description }}
      </p>
      <p v-if="facts.length" class="lp-kicker mt-1">
        <span v-for="(f, i) in facts" :key="f.k">
          <span v-if="i > 0"> · </span>{{ f.k }}: {{ f.v }}
        </span>
      </p>
      <p class="lp-kicker mt-1">
        <a
          v-if="web.website"
          :href="web.website"
          target="_blank"
          rel="noopener"
          class="underline"
          data-testid="ec-org-website"
          >{{ t("ec.orgWebsite") }}</a
        >
        <a
          v-if="web.source_url"
          :href="web.source_url"
          target="_blank"
          rel="noopener"
          class="underline"
          ><span v-if="web.website"> · </span>{{ t("ec.bioVia", { source: web.source }) }}</a
        >
        <span v-if="web.logo_license"> · {{ t("ec.photoLicense", { license: web.logo_license }) }}</span>
      </p>
    </div>
  </section>

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
