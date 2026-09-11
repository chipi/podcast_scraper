<script setup lang="ts">
/**
 * Entity card SHELL (PRD-043 FR2/FR3; UXS-014) — the shared person/topic frame: the header (kicker
 * / role / title / one-line descriptor / follow / save / collection / dismiss / open-in-page), the
 * re-entrant back stack (walk the graph, step back), and the load. The kind-specific body is
 * delegated to {@link PersonCardContent} / {@link TopicCardContent}. The shell holds only what BOTH
 * need — so the ONE stack can carry a mixed person↔topic walk within a single panel, which is why
 * this stays one component rather than two full cards.
 *
 * Rendered two ways:
 *   • `inline`  — replaces a panel's content with a ‹ Back (Insights → entity); no new layer.
 *   • `overlay` — wrapped in EntityCard's modal (Search → entity, a page-level surface).
 */
import { computed, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
import { getOrgCard, getPersonCard, getTopicCard } from "../services/api"
import type { OrgCard, PersonCard, TopicCard } from "../services/types"
import AddToCollectionButton from "./AddToCollectionButton.vue"
import FavoriteButton from "./FavoriteButton.vue"
import PersonCardContent from "./PersonCardContent.vue"
import TopicCardContent from "./TopicCardContent.vue"
import OrgCardContent from "./OrgCardContent.vue"
import ShareMenu from "./ShareMenu.vue"
import type { EntityCardModel } from "../composables/entityShareCard"
import { useAuthStore } from "../stores/auth"
import { useInterestsStore } from "../stores/interests"
import { useFavoritesStore } from "../stores/favorites"

type EntityKind = "person" | "topic" | "organization"
type Target = { kind: EntityKind; id: string }

const props = withDefaults(
  defineProps<{
    kind: EntityKind
    id: string
    variant?: "inline" | "overlay"
    /**
     * What the control means when there is nothing left on the card's own back stack.
     *
     * `back` — the card DRILLED DOWN inside something you were already reading, and dismissing it
     * returns you to that thing. The Knowledge Panel works this way: tapping a person chip replaces
     * the panel body, and the panel is still what you are in.
     *
     * `close` — the card is the whole destination. An overlay sheet you opened, or the standalone
     * topic / person route, where nothing contains it. A back arrow here is a back arrow whose only
     * job is to close, which is what it looked like on the full-page route.
     */
    rootControl?: "back" | "close"
  }>(),
  { variant: "overlay", rootControl: undefined }
)
const emit = defineEmits<{ (e: "close"): void }>()

const { t } = useI18n()
const auth = useAuthStore()
const interests = useInterestsStore()
const favorites = useFavoritesStore()
// Load follow + favourite state once we know the user is signed in — auth may resolve after mount.
watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (authed) {
      void interests.ensureLoaded()
      void favorites.ensureLoaded()
    }
  },
  { immediate: true }
)

// Follow this person/topic → its id is the interest token (person:… / topic:…), which feeds
// personalized discovery. Following shapes "Recommended for you" on Home.
const following = computed(() => interests.has(current.value.id))
function toggleFollow(): void {
  void interests.toggle(current.value.id)
}

// Back stack — the bottom is the entity opened on; the top is what's shown.
const stack = ref<Target[]>([{ kind: props.kind, id: props.id }])
const current = computed<Target>(() => stack.value[stack.value.length - 1])
const atRoot = computed(() => stack.value.length === 1)
// An overlay is a thing you opened; an inline card is, by default, a drill-down inside its host.
// A host that is itself the destination (the standalone routes) says so with `rootControl`.
const dismissAtRoot = computed(
  () =>
    atRoot.value &&
    (props.rootControl ?? (props.variant === "overlay" ? "close" : "back")) === "close"
)

const person = ref<PersonCard | null>(null)
const topic = ref<TopicCard | null>(null)
const org = ref<OrgCard | null>(null)
const loading = ref(false)
const failed = ref(false)

async function load(target: Target): Promise<void> {
  loading.value = true
  failed.value = false
  person.value = null
  topic.value = null
  org.value = null
  try {
    if (target.kind === "person") person.value = await getPersonCard(target.id)
    else if (target.kind === "organization") org.value = await getOrgCard(target.id)
    else topic.value = await getTopicCard(target.id)
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
}

// Re-open on a brand-new target (parent opened a different chip) — reset the stack.
watch(
  () => [props.kind, props.id] as const,
  ([kind, id]) => {
    stack.value = [{ kind, id }]
  }
)
watch(current, (target) => void load(target), { immediate: true })

function open(kind: EntityKind, id: string): void {
  stack.value = [...stack.value, { kind, id }]
}
// Left control: pop the stack if deeper, else dismiss the whole card (back to panel / close modal).
function onBack(): void {
  if (stack.value.length > 1) stack.value = stack.value.slice(0, -1)
  else emit("close")
}

const label = computed(() => person.value?.label ?? topic.value?.label ?? org.value?.label ?? "")

// #2036 — the shareable card model for the current entity. Lean v1: kicker + title + an
// episode-count stat + canonical link (person/topic have pages; org is overlay-only → no link).
const shareModel = computed<EntityCardModel>(() => {
  const kind = current.value.kind
  const kicker =
    kind === "person" ? t("ec.person") : kind === "organization" ? t("ec.organization") : t("ec.topic")
  const card = person.value ?? topic.value ?? org.value
  const eps = card?.episode_count ?? 0
  const origin = typeof window !== "undefined" ? window.location.origin : ""
  const path =
    kind === "topic"
      ? `/topic/${current.value.id}`
      : kind === "person"
        ? `/person/${current.value.id}`
        : "" // org has no standalone page yet
  return {
    kicker,
    title: label.value || current.value.id,
    stats: eps ? `${eps} ${eps === 1 ? "episode" : "episodes"}` : null,
    // accent omitted → the engine's DEFAULT_ACCENT (a token-mirrored hex in the .ts) applies;
    // the card's colour lives in one place, not as a literal in this component.
    url: path && origin ? origin + path : null,
  }
})

// Speaker role badge (host / guest / mentioned) — KG-grounded from the person node's aggregate
// role. Empty for topics / unknown role.
const ROLE_LABEL_KEYS: Record<string, string> = {
  host: "ec.roleHost",
  guest: "ec.roleGuest",
  mentioned: "ec.roleMentioned",
}
const personRole = computed(() =>
  current.value.kind === "person" ? (person.value?.role ?? "").toLowerCase() : ""
)
const personRoleLabel = computed(() => {
  const key = ROLE_LABEL_KEYS[personRole.value]
  return key ? t(key) : ""
})
// The external bio's one-line descriptor rides in the header ("who is this"); the rest of the bio
// lives in the person body. Kept in the shell because it sits beside the title.
const personWeb = computed(() => person.value?.web ?? null)
const isTopic = computed(() => current.value.kind === "topic")
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col bg-surface">
    <!-- Header mirrors the episode-detail masthead (UXS-014): kicker + role, then title + all
         actions (follow / save / collection + the close-or-back control) on one row. -->
    <header class="border-b border-border px-4 py-3">
      <span class="flex items-center gap-2">
        <span class="lp-kicker">{{
          current.kind === "person"
            ? t("ec.person")
            : current.kind === "organization"
              ? t("ec.organization")
              : t("ec.topic")
        }}</span>
        <!-- Host / guest / mentioned — the person's aggregate speaker role. Host gets the ringed
             emphasis idiom used for the "current" chip elsewhere. -->
        <span
          v-if="personRoleLabel"
          data-testid="ec-person-role"
          :data-role="personRole"
          class="rounded-full bg-overlay px-2 py-0.5 text-[0.65rem] font-bold uppercase tracking-wide text-person"
          :class="personRole === 'host' ? 'ring-1 ring-person' : ''"
          >{{ personRoleLabel }}</span
        >
      </span>
      <div class="mt-1 flex items-start justify-between gap-3">
        <div class="min-w-0 flex-1">
          <span class="block truncate font-display text-xl font-extrabold">{{ label || "…" }}</span>
          <!-- One-line "who is this" descriptor under the name (person_web) — glanceable identity
               without reading the bio. e.g. "American financier and politician". -->
          <span
            v-if="!isTopic && personWeb?.description"
            class="mt-0.5 block truncate text-sm text-muted"
            data-testid="ec-person-descriptor"
            >{{ personWeb.description }}</span
          >
        </div>
        <div class="flex shrink-0 items-center gap-2">
          <template v-if="label">
            <button
              v-if="auth.isAuthenticated"
              type="button"
              data-testid="ec-follow"
              class="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition"
              :class="
                following
                  ? 'bg-accent text-accent-foreground'
                  : 'bg-overlay text-canvas-foreground hover:bg-elevated'
              "
              :aria-pressed="following"
              :title="t('ec.followHint')"
              @click="toggleFollow"
            >
              <span aria-hidden="true">{{ following ? "✓" : "+" }}</span>
              {{ following ? t("ec.following") : t("ec.follow") }}
            </button>
            <!-- Save + collection are person/topic only — the org card is deliberately lean
                 (#2031: a name + where it's mentioned + who co-occurs), no save/collection. -->
            <template v-if="current.kind !== 'organization'">
              <!-- Save (heart) — the ONE save affordance; distinct from Follow (F2.2). -->
              <FavoriteButton :item="{ kind: current.kind, ref: current.id, label }" />
              <!-- Pin this topic/person into a collection (RFC-119) — self-gates when signed out. -->
              <AddToCollectionButton :item="{ kind: current.kind, ref: current.id }" variant="pill" />
            </template>
          </template>
          <!-- Share (card / link / text) — #2036. Present for every kind once the card has loaded. -->
          <ShareMenu v-if="label" :model="shareModel" />
          <!-- Close (✕) at the card root, Back (‹) when deeper in the walk. -->
          <button
            type="button"
            class="lp-nav shrink-0"
            :aria-label="dismissAtRoot ? t('ec.close') : t('ec.back')"
            data-testid="ec-dismiss"
            @click="onBack"
          >
            <span aria-hidden="true" class="text-base leading-none">{{
              dismissAtRoot ? "✕" : "‹"
            }}</span>
          </button>
        </div>
      </div>
      <!-- #1261-9: escape hatch from the modal to the standalone page. Overlay only — inline is
           already the standalone page or an embedded panel where a link would go nowhere useful. -->
      <RouterLink
        v-if="variant === 'overlay' && label && current.kind !== 'organization'"
        :to="{ name: current.kind === 'topic' ? 'topic' : 'person', params: { id: current.id } }"
        class="mt-2 ml-2 inline-flex items-center gap-1 rounded-full bg-overlay px-3 py-1 text-xs font-bold text-canvas-foreground transition hover:bg-elevated"
        data-testid="ec-open-in-page"
        @click="emit('close')"
      >
        {{ t("ec.openInPage") }} ›
      </RouterLink>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <p v-if="loading" class="text-sm text-muted">{{ t("ec.loading") }}</p>
      <p v-else-if="failed || (!person && !topic && !org)" class="text-sm text-muted">
        {{ t("ec.notFound") }}
      </p>
      <!-- Kind-specific body. `person`/`topic`/`org` are set XOR by `load()` on the current target. -->
      <PersonCardContent
        v-else-if="person"
        :person="person"
        @open="(p) => open(p.kind, p.id)"
        @close="emit('close')"
      />
      <TopicCardContent
        v-else-if="topic"
        :topic="topic"
        @open="(p) => open(p.kind, p.id)"
        @close="emit('close')"
      />
      <OrgCardContent
        v-else-if="org"
        :org="org"
        @open="(p) => open(p.kind, p.id)"
        @close="emit('close')"
      />
    </div>
  </div>
</template>
