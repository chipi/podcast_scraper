<script setup lang="ts">
/**
 * Entity card SHELL (PRD-043 FR2/FR3; UXS-014) — the shared person/topic frame: the header (kicker
 * / role / title / one-line descriptor / follow / save / collection / dismiss), the
 * re-entrant back stack (walk the graph, step back), and the load. The kind-specific body is
 * delegated to {@link PersonCardContent} / {@link TopicCardContent}. The shell holds only what BOTH
 * need — so the ONE stack can carry a mixed person↔topic walk within a single panel, which is why
 * this stays one component rather than two full cards.
 *
 * Rendered two ways:
 *   • `inline`  — replaces a panel's content with a ‹ Back (Insights → entity); no new layer.
 *   • `overlay` — wrapped in EntityCard's modal (Search → entity, a page-level surface).
 */
import { computed, nextTick, ref, watch } from "vue"
import BackIcon from "./BackIcon.vue"
import CloseIcon from "./CloseIcon.vue"
import { anchorWithin, restoreAnchorWithin, type ClickAnchor } from "../utils/backAnchor"
import { restoreScroll } from "../utils/scrollRestore"
import { useI18n } from "vue-i18n"
import { getOrgCard, getPersonCard, getTopicCard } from "../services/api"
import type { OrgCard, PersonCard, TopicCard } from "../services/types"
import AddToCollectionButton from "./AddToCollectionButton.vue"
import FavoriteButton from "./FavoriteButton.vue"
import FollowButton from "./FollowButton.vue"
import PersonCardContent from "./PersonCardContent.vue"
import TopicCardContent from "./TopicCardContent.vue"
import OrgCardContent from "./OrgCardContent.vue"
import ShareMenu from "./ShareMenu.vue"
import { useAuthStore } from "../stores/auth"
import { useInterestsStore } from "../stores/interests"
import { useFavoritesStore } from "../stores/favorites"

type EntityKind = "person" | "topic" | "organization"
/**
 * `scroll` is where the reader was on this entity when they walked on to the next one, so Back can
 * return them there. Two offsets because the card scrolls in its own body inside a sheet or panel,
 * but the PAGE scrolls on the standalone /person and /topic routes, where the body is unbounded.
 */
type Target = {
  kind: EntityKind
  id: string
  scroll?: { body: number; page: number }
  /** The control the reader tapped to walk on — Back puts THAT back where it sat (backAnchor). */
  anchor?: ClickAnchor | null
}

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
    /** Stack depth of the sheet hosting this body; forwarded so what it opens sits one deeper. */
    depth?: number
    /**
     * May this card open a storyline / person as a sheet ON TOP, rather than routing away?
     *
     * Defaults to `dismissAtRoot`, which ties the answer to the back-stack — so a card stopped
     * layering as soon as you drilled one level inside it, for no reason a user could perceive.
     * Hosts that know better say so: the Insights panel is a full-height bottom sheet, so a card
     * opened in it CAN be stacked on even though it renders `inline` (operator 2026-09-16 — the
     * requirement is topic in the background, storyline on it, person on that).
     */
    canLayer?: boolean
    /**
     * The card IS the page (the standalone /topic and /person routes): no side padding of its own,
     * so its content starts at the page's left edge like every other page (operator 2026-10-05,
     * one page width). In a sheet or a panel the card keeps `px-4`, because there the card's
     * padding is the only gutter.
     */
    flush?: boolean
  }>(),
  { variant: "overlay", rootControl: undefined, depth: 0, canLayer: undefined, flush: false }
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

const bodyEl = ref<HTMLElement | null>(null)
// Set by Back, applied once the entity it returned to has loaded (operator 2026-10-04).
let pendingScroll: Target["scroll"] | null = null
let pendingAnchor: ClickAnchor | null = null
// The standalone /person and /topic routes, where the card IS the page and the page is what
// scrolls. Anywhere else (a panel, a sheet) the page behind belongs to someone else — on desktop
// the episode-notes rail sits beside the episode — and the card must not move it.
const cardIsPage = computed(() => props.variant === "inline" && props.rootControl === "close")

function open(kind: EntityKind, id: string): void {
  const here = stack.value.slice(0, -1)
  const leaving: Target = {
    ...current.value,
    scroll: { body: bodyEl.value?.scrollTop ?? 0, page: cardIsPage.value ? window.scrollY : 0 },
    anchor: bodyEl.value ? anchorWithin(bodyEl.value) : null,
  }
  stack.value = [...here, leaving, { kind, id }]
  // A new entity starts at its top, not at the offset the reader had reached on the last one.
  if (bodyEl.value) bodyEl.value.scrollTop = 0
  if (cardIsPage.value) window.scrollTo({ top: 0 })
}
// Left control: pop the stack if deeper, else dismiss the whole card (back to panel / close modal).
function onBack(): void {
  if (stack.value.length > 1) {
    stack.value = stack.value.slice(0, -1)
    pendingScroll = current.value.scroll ?? null
    pendingAnchor = current.value.anchor ?? null
  } else emit("close")
}
watch(loading, (isLoading) => {
  if (isLoading || !pendingScroll) return
  const { body, page } = pendingScroll
  const anchor = pendingAnchor
  pendingScroll = null
  pendingAnchor = null
  void nextTick(async () => {
    // The tapped control, back where it sat — robust to sections above it loading at a different
    // pace than last time. The offsets are the fallback when it cannot be found again.
    const body_ = bodyEl.value
    if (anchor && body_ && (await restoreAnchorWithin(body_, anchor))) return
    void restoreScroll(body_, body)
    if (cardIsPage.value) void restoreScroll(null, page)
  })
})

const label = computed(() => person.value?.label ?? topic.value?.label ?? org.value?.label ?? "")

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
// The external bio's one-line descriptor rides in the header ("who is this"); the bio prose itself
// sits beside the photo in the person body.
const personWeb = computed(() => person.value?.web ?? null)
const isTopic = computed(() => current.value.kind === "topic")
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col bg-surface">
    <!-- Header (UXS-014, unified across topic / person / storyline): the actions ride the top row
         WITH the kicker, so the TITLE owns its own full-width row and can run to two lines instead
         of being crushed to "agent in…" beside the icons. Same structure in StorylineView. -->
    <header class="border-b border-border py-3" :class="props.flush ? '' : 'px-4'">
      <div class="flex items-start justify-between gap-3">
        <span class="flex min-w-0 flex-wrap items-center gap-2">
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
        <!-- Only the close/back control rides the kicker row; the primary actions get their OWN
             row AFTER the title (operator: the kicker+actions row was too cramped and misaligned). -->
        <button
          type="button"
          class="lp-nav shrink-0"
          :aria-label="dismissAtRoot ? t('ec.close') : t('ec.back')"
          data-testid="ec-dismiss"
          @click="onBack"
        >
          <!-- Both drawn, same box and stroke: ✕ because U+2715 is a tofu box in the iOS UI font
               (see CloseIcon), ‹ because as a character it was a sliver beside that ✕ (BackIcon). -->
          <CloseIcon v-if="dismissAtRoot" />
          <BackIcon v-else />
        </button>
      </div>

      <!-- TITLE on its own full-width row — up to two lines, never sliced against the actions. -->
      <h2 class="mt-2 font-display text-xl font-extrabold leading-snug line-clamp-2">
        {{ label || "…" }}
      </h2>
      <!-- One-line "who is this" descriptor under the name (person_web) — glanceable identity
           without reading the bio. e.g. "American financier and politician". The BIO itself sits
           beside the photo in PersonCardContent; this is the short subtitle, not that. -->
      <span
        v-if="!isTopic && personWeb?.description"
        class="mt-0.5 block truncate text-sm text-muted"
        data-testid="ec-person-descriptor"
        >{{ personWeb.description }}</span
      >

      <!-- Actions on their OWN aligned row, AFTER the title (operator). Follow / save / collection
           are person·topic; org is deliberately lean (#2031). Share is present for every kind. -->
      <div v-if="label" class="mt-3 flex flex-wrap items-center gap-2">
        <FollowButton
          v-if="auth.isAuthenticated"
          variant="ec"
          :following="following"
          :label="label"
          @toggle="toggleFollow"
        />
        <template v-if="current.kind !== 'organization'">
          <!-- Save (heart) — the ONE save affordance; distinct from Follow (F2.2). -->
          <FavoriteButton :item="{ kind: current.kind, ref: current.id, label }" />
          <!-- Pin this topic/person into a collection (RFC-119) — self-gates when signed out. -->
          <AddToCollectionButton :item="{ kind: current.kind, ref: current.id }" variant="pill" />
        </template>
        <!-- Share (card / link / text) — #2036. -->
        <!-- The server's card for this entity (operator 2026-10-05) — the menu needs only what it is. -->
        <ShareMenu
          :kind="current.kind"
          :id="current.id"
          :title="label || current.id"
          :context="person?.web?.description ?? org?.web?.description ?? null"
          :target-kind="current.kind"
        />
      </div>

      <!-- REMOVED (operator 2026-09-16): the "Open in page ›" escape hatch (#1261-9).
           The sheet already shows everything the standalone page does, so the link asked the user
           to make a navigation decision that changes nothing they can see — and it sat directly
           under the action row, competing with Follow / favourite / Collection / Share for the one
           position the eye lands on first. Topics and people remain reachable as pages by deep
           link and from search; nothing else pointed here.
           The `router.back()` trap this link once documented now lives on `EpisodeRow`, which is
           where it actually bit (tapping an episode landed on Home). -->
    </header>

    <div ref="bodyEl" class="min-h-0 flex-1 overflow-y-auto py-4" :class="props.flush ? '' : 'px-4'">
      <p v-if="loading" class="text-sm text-muted">{{ t("ec.loading") }}</p>
      <p v-else-if="failed || (!person && !topic && !org)" class="text-sm text-muted">
        {{ t("ec.notFound") }}
      </p>
      <!-- Kind-specific body. `person`/`topic`/`org` are set XOR by `load()` on the current target. -->
      <PersonCardContent
        v-else-if="person"
        :person="person"
        :depth="depth"
        @open="(p) => open(p.kind, p.id)"
        @close="emit('close')"
      />
      <!-- `can-layer`: may this card open a person/storyline as a sheet ON TOP, or must it replace
           in place? The rule app-wide is that a sheet may layer over another SHEET or over a PAGE,
           but never over an inline PANEL — inside the Knowledge Panel the panel is already the
           layer, and a modal on top would put two dismissables on screen with two different Back
           meanings (the replace-in-panel rule, UXS-014).
           `dismissAtRoot` already draws exactly that line: true when this card is the whole
           destination (overlay sheet, or a standalone /topic/:id page), false when it is a
           drill-down inside a host panel. So the policy is one existing condition, not a new
           concept (operator 2026-09-16). -->
      <TopicCardContent
        v-else-if="topic"
        :topic="topic"
        :can-layer="canLayer ?? dismissAtRoot"
        :depth="depth"
        :wide="props.flush"
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
