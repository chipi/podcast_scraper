<script setup lang="ts">
/**
 * Profile / account — where the signed-in user sees who they are and edits their personalization,
 * starting with their interest topics (chosen at sign-in via the onboarding card). Auth-gated.
 */
import { computed, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
defineOptions({ name: "ProfileView" }) // stable name for <keep-alive :include> (App.vue)
import {
  getComms,
  getMyStats,
  getTopClusters,
  getUserInterests,
  putComms,
  uploadAvatar,
} from "../services/api"
import type {
  CommsChannel,
  CommsSettings,
  CommsType,
  InterestCluster,
  UserStats,
} from "../services/types"
import { disablePush, enablePush } from "../composables/usePushSubscription"
import { useRouter } from "vue-router"
import { CACHE_KEYS, clearCached } from "../services/contentCache"
import { useAuthStore } from "../stores/auth"
import { useUserPreferencesStore } from "../stores/userPreferences"
import Tabs from "../components/Tabs.vue"
import { panelAttrs, type TabSpec } from "../components/tabs"
import InterestsPicker from "../components/InterestsPicker.vue"
import Sparkline from "../components/Sparkline.vue"
import ListeningRecap from "../components/ListeningRecap.vue"
import ProfileAvatar from "../components/ProfileAvatar.vue"
import AvatarCropModal from "../components/AvatarCropModal.vue"

const { t } = useI18n()
const auth = useAuthStore()
const userPrefs = useUserPreferencesStore()

// Avatar upload (Area E) — a narrow control on the identity header; on success we re-fetch /me so
// the new photo shows everywhere ProfileAvatar reads auth.user.image.
const avatarInput = ref<HTMLInputElement | null>(null)
const avatarError = ref<string | null>(null)
const avatarBusy = ref(false)
// A picked file is composed in the crop modal first (pan + zoom to a square) — the modal emits the
// cropped Blob we actually upload, so portraits/landscapes fill the circle cleanly.
const cropFile = ref<File | null>(null)

function onAvatarPicked(e: Event): void {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (avatarInput.value) avatarInput.value.value = "" // allow re-picking the same file
  if (!file) return
  avatarError.value = null
  cropFile.value = file // open the crop modal
}

async function onCropConfirm(blob: Blob): Promise<void> {
  cropFile.value = null
  avatarBusy.value = true
  try {
    await uploadAvatar(blob)
    await auth.refresh() // /me now carries the uploaded image
  } catch {
    avatarError.value = t("profile.avatarUploadFailed")
  } finally {
    avatarBusy.value = false
  }
}

// Profile is tabbed (Account / Topics / Stats) so the identity, personalization and analytics are
// three destinations rather than one long scroll. About/version/help live in Settings (the gear).
type ProfileTab = "account" | "topics" | "stats"
const tab = ref<ProfileTab>("account")
const profileTabs = computed<TabSpec<ProfileTab>[]>(() => [
  { key: "account", label: t("profile.tabAccount") },
  { key: "topics", label: t("profile.tabTopics") },
  { key: "stats", label: t("profile.tabStats") },
])

// How "Your Week" lays out on the home page — a synced per-user preference, shared with the inline
// "Show more / Show less" toggle on the home section (#1412). Independent of the email toggle below.
const YOUR_WEEK_LAYOUT_KEY = "lp.yourweek.layout"
const yourWeekLayout = ref<"compact" | "full">("compact")
const yourWeekLayoutOptions = computed<TabSpec<"compact" | "full">[]>(() => [
  { key: "compact", label: t("profile.yourWeekCompact") },
  { key: "full", label: t("profile.yourWeekFull") },
])
function setYourWeekLayout(v: "compact" | "full"): void {
  yourWeekLayout.value = v
  void userPrefs.set(YOUR_WEEK_LAYOUT_KEY, v)
}

const interests = ref<string[]>([])
const clusters = ref<InterestCluster[]>([])
const pickerOpen = ref(false)

// Listening analytics (UXS-014) — the user's own play history, summarized.
const router = useRouter()

/**
 * Sign out (#1962) — moved here from the masthead.
 *
 * The cached content belongs to the identity being discarded (#1909), so it is cleared BEFORE the
 * identity goes: a signed-out device must not keep another session's library readable.
 */
async function onSignOut(): Promise<void> {
  await clearCached(CACHE_KEYS)
  await auth.logout()
  // Home, not Catalog (#1594). Catalog is a flat index of every episode in the corpus — a
  // reasonable place to browse and the wrong place to LAND. Home is the app's front door: it
  // renders a signed-out hero explaining what the app is for, which is the only thing a person who
  // just signed out might want next. Sending them to a bare list instead reads like a session that
  // half-broke rather than one they deliberately ended.
  await router.push({ name: "home" })
}

const stats = ref<UserStats | null>(null)
/** A failed load is not an empty one — see the note in `hydrate()`. */
const statsFailed = ref(false)
const interestsFailed = ref(false)
// NO hours tile here any more (#1914). `/me/stats` reports `listening_seconds` as
// `sum(position_seconds)` — a lifetime snapshot of furthest position reached, which rises when
// you seek forward without hearing anything and does not move when you re-listen. It was rendered
// as a headline "Xh". ListeningRecap shows time actually accrued instead, with its coverage.
const series = computed(() => stats.value?.daily.map((d) => d.count) ?? [])
const hasStats = computed(() => !!stats.value && stats.value.episodes > 0)

// Map saved interest tokens → human labels. Clusters resolve via the top-cluster set; topics and
// people (followed from entity cards) de-slug from their id (`topic:personal-growth` → "personal
// growth"). `kind` drives the chip hue so people read distinct from topics.
const interestLabels = computed(() => {
  const byId = new Map(clusters.value.map((c) => [c.id, c.label]))
  return interests.value.map((id) => ({
    id,
    kind: id.startsWith("person:") ? "person" : "topic",
    label: byId.get(id) ?? id.replace(/^(tc|topic|person):/, "").replace(/-/g, " "),
  }))
})

// Delivery consent (PRD-046 FR1 / #1414) — the "Your Week" digest + push nudges.
const comms = ref<CommsSettings | null>(null)

async function load(): Promise<void> {
  // Each catch used to collapse a FAILURE into an empty value, and the template then read that
  // emptiness as fact: with no network the page said "Start listening to build your stats" and
  // "No interests chosen yet" to a user with both. That is #1591's defect — "a cold corpus and a
  // total API outage rendered the same page" — recurring here, where nothing was watching for it.
  statsFailed.value = false
  interestsFailed.value = false
  const [ints, tops, st, cm] = await Promise.all([
    getUserInterests().catch(() => {
      interestsFailed.value = true
      return [] as string[]
    }),
    getTopClusters(50).catch(() => [] as InterestCluster[]),
    getMyStats().catch(() => {
      statsFailed.value = true
      return null
    }),
    getComms().catch(() => null),
  ])
  interests.value = ints
  clusters.value = tops
  stats.value = st
  comms.value = cm
  await userPrefs.hydrate()
  const layoutPref = userPrefs.get<string>(YOUR_WEEK_LAYOUT_KEY)
  if (layoutPref === "full" || layoutPref === "compact") yourWeekLayout.value = layoutPref
}

function onSaved(ids: string[]): void {
  interests.value = ids
}

// The per-type × per-channel matrix (wave-I). Types down, channels across.
const NOTIFICATION_TYPES: CommsType[] = ["digest", "new_episodes", "product"]
const NOTIFICATION_CHANNELS: CommsChannel[] = ["email", "push", "in_app"]

const anyPushOn = computed(
  () => !!comms.value && NOTIFICATION_TYPES.some((tp) => comms.value!.types[tp].push)
)

// Send the FULL matrix (the server merges known cells, so a partial would reset the rest).
async function saveMatrix(): Promise<void> {
  if (comms.value) comms.value = await putComms({ types: comms.value.types })
}

// The digest email cadence lives outside the matrix.
async function saveSchedule(): Promise<void> {
  if (comms.value) comms.value = await putComms({ digest_schedule: comms.value.digest_schedule })
}

// A push cell needs a real browser subscription behind it, not just a flag.
// - Turning a push cell ON: ensure a subscription exists FIRST; if the browser refuses, revert the
//   cell and don't persist.
// - Turning the LAST push cell OFF: unregister the subscription BEFORE the server save, so the
//   user's intent is honored browser-side even if the save call then fails (a save-after-unsub
//   ordering avoids leaving the browser subscribed to pushes the user just turned off).
async function onChannelToggle(type: CommsType, channel: CommsChannel): Promise<void> {
  if (!comms.value) return
  const cell = comms.value.types[type]
  if (channel === "push") {
    if (cell.push) {
      let ok = false
      try {
        ok = await enablePush()
      } catch {
        ok = false
      }
      if (!ok) {
        cell.push = false // browser refused → the UI must not claim push is on
        return
      }
    } else if (!anyPushOn.value) {
      // last push cell off → unsubscribe before persisting the matrix. Best-effort: a failed
      // unsubscribe must NOT skip the save (the server matrix would then stay push-on).
      try {
        await disablePush()
      } catch {
        /* keep going — persist the user's choice regardless */
      }
    }
  }
  await saveMatrix()
}

onMounted(load)
</script>

<template>
  <section class="max-w-2xl">
    <!-- Identity header: avatar + name + @handle + email, with the Settings gear on the right. -->
    <div class="mb-5 flex items-center justify-between gap-3">
      <div class="flex min-w-0 items-center gap-3">
        <button
          type="button"
          class="relative shrink-0 rounded-full"
          :aria-label="t('profile.changePhoto')"
          :aria-busy="avatarBusy"
          data-testid="avatar-upload-trigger"
          @click="avatarInput?.click()"
        >
          <ProfileAvatar
            :name="auth.user?.name"
            :email="auth.user?.email"
            :src="auth.user?.image"
            :size="48"
          />
          <span
            class="absolute -bottom-0.5 -right-0.5 flex h-5 w-5 items-center justify-center rounded-full border border-canvas bg-elevated text-[10px] text-canvas-foreground"
            aria-hidden="true"
            >✎</span
          >
        </button>
        <input
          ref="avatarInput"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          class="hidden"
          data-testid="avatar-file-input"
          @change="onAvatarPicked"
        />
        <div class="min-w-0">
          <h1 class="truncate font-display text-2xl font-extrabold tracking-tight">
            {{ auth.user?.name || t("profile.title") }}
          </h1>
          <p
            v-if="auth.user?.username"
            class="truncate text-sm text-muted"
            data-testid="profile-handle"
          >
            @{{ auth.user.username }}
          </p>
          <p v-if="auth.user?.email" class="truncate text-sm text-muted">{{ auth.user?.email }}</p>
        </div>
      </div>
      <RouterLink
        :to="{ name: 'settings' }"
        class="shrink-0 rounded-full border border-border p-2 text-muted no-underline transition hover:bg-overlay hover:text-canvas-foreground"
        :aria-label="t('settings.title')"
        :title="t('settings.title')"
        data-testid="profile-settings-link"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="h-5 w-5"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="3" />
          <path
            d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
          />
        </svg>
      </RouterLink>
    </div>
    <p
      v-if="avatarError"
      class="mb-4 text-sm font-semibold text-danger"
      role="alert"
      data-testid="avatar-error"
    >
      {{ avatarError }}
    </p>

    <AvatarCropModal
      v-if="cropFile"
      :file="cropFile"
      @confirm="onCropConfirm"
      @cancel="cropFile = null"
    />

    <Tabs
      v-model="tab"
      :tabs="profileTabs"
      :label="t('profile.title')"
      id-prefix="profile"
      variant="pill"
      class="mb-5"
    />

    <!-- STATS tab: listening analytics + the recap. -->
    <div v-show="tab === 'stats'" v-bind="panelAttrs('profile', 'stats')">
      <!-- Listening analytics (UXS-014) — derived entirely from this user's own play history. -->
      <section class="rounded-2xl border border-border p-5">
        <h2 class="lp-section mb-4">{{ t("stats.title") }}</h2>
        <template v-if="hasStats">
          <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div class="rounded-xl bg-overlay p-4">
              <div class="flex items-baseline gap-1">
                <span class="font-display text-3xl font-extrabold leading-none">{{
                  stats!.day_streak
                }}</span>
                <span v-if="stats!.day_streak > 0" aria-hidden="true">🔥</span>
              </div>
              <div class="mt-2 text-xs font-medium text-muted">{{ t("stats.streak") }}</div>
            </div>
            <div class="rounded-xl bg-overlay p-4">
              <span class="font-display text-3xl font-extrabold leading-none">{{
                stats!.episodes
              }}</span>
              <div class="mt-2 text-xs font-medium text-muted">{{ t("stats.episodes") }}</div>
            </div>
            <div class="rounded-xl bg-overlay p-4">
              <span class="font-display text-3xl font-extrabold leading-none">{{
                stats!.shows
              }}</span>
              <div class="mt-2 text-xs font-medium text-muted">{{ t("stats.shows") }}</div>
            </div>
          </div>
          <div class="mt-3 rounded-xl bg-overlay p-4">
            <div class="mb-2 flex items-baseline justify-between">
              <span class="text-xs font-medium text-muted">{{ t("stats.overTime") }}</span>
              <span class="text-xs text-muted">{{
                t("stats.activeDays", stats!.active_days, { named: { count: stats!.active_days } })
              }}</span>
            </div>
            <Sparkline
              :values="series"
              :width="320"
              :height="44"
              class="block w-full text-canvas-foreground"
            />
          </div>
        </template>
        <p v-else-if="statsFailed" class="text-sm text-muted" data-testid="stats-unavailable">
          {{ t("profile.unavailable") }}
        </p>
        <p v-else class="text-sm text-muted">{{ t("stats.empty") }}</p>
      </section>

      <!-- The recap (#1914): time actually listened, the listener's own days, what recurred, and the
         line they kept. -->
      <ListeningRecap class="mt-6" />
    </div>

    <!-- TOPICS tab: the interest topics driving personalization. -->
    <div v-show="tab === 'topics'" v-bind="panelAttrs('profile', 'topics')">
      <section class="rounded-2xl border border-border p-5">
        <div class="mb-3 flex items-center justify-between gap-2">
          <h2 class="lp-section">{{ t("profile.interests") }}</h2>
          <button
            type="button"
            class="text-sm font-bold text-accent"
            data-testid="profile-edit-interests"
            @click="pickerOpen = true"
          >
            {{ t("profile.editInterests") }}
          </button>
        </div>
        <p class="mb-3 text-sm text-muted">{{ t("profile.interestsHelp") }}</p>
        <div v-if="interestLabels.length" class="flex flex-wrap gap-1.5">
          <span
            v-for="i in interestLabels"
            :key="i.id"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs"
            :class="i.kind === 'person' ? 'text-person' : 'text-topic'"
            >{{ i.label }}</span
          >
        </div>
        <p
          v-else-if="interestsFailed"
          class="text-sm text-muted"
          data-testid="interests-unavailable"
        >
          {{ t("profile.unavailable") }}
        </p>
        <p v-else class="text-sm text-muted">{{ t("profile.noInterests") }}</p>
      </section>
    </div>

    <!-- ACCOUNT tab: delivery/notifications + sign out. -->
    <div v-show="tab === 'account'" v-bind="panelAttrs('profile', 'account')">
      <!-- Delivery consent (PRD-046 FR1 / #1414) — the "Your Week" digest + push nudges. -->
      <section v-if="comms" class="rounded-2xl border border-border p-5">
        <h2 class="lp-section mb-1">{{ t("profile.notifications") }}</h2>
        <p class="mb-3 text-sm text-muted">{{ t("profile.notificationsHelp") }}</p>

        <!-- How Your Week lays out on your home — the in-app view is the primary surface. -->
        <div class="flex items-center justify-between gap-3 py-2">
          <span class="text-sm font-medium">{{ t("profile.yourWeekLayout") }}</span>
          <!-- Same shared control as Search's scope switch (#1959). It was hand-rolled here with
             one-off Tailwind and NO ARIA, so the identical interaction announced itself as two
             unrelated buttons to a screen reader and looked like two mismatched halves glued
             together. Selected state now rides on aria-selected, so the accessible state and the
             visible state cannot drift apart again. -->
          <!--
          A radiogroup, not a tablist (#1594 item 7). This sets a saved preference and switches no
          region, so there is nothing for a tab to control; "tab" was simply the wrong role.
        -->
          <Tabs
            :model-value="yourWeekLayout"
            :tabs="yourWeekLayoutOptions"
            :label="t('profile.yourWeekLayout')"
            id-prefix="yourweek-layout"
            variant="segment"
            pattern="radio"
            @update:model-value="setYourWeekLayout"
          />
        </div>
        <p class="mb-1 text-xs text-muted">{{ t("profile.yourWeekLayoutHelp") }}</p>

        <!-- Per-type × per-channel matrix (wave-I): each notification TYPE delivers independently on
           each CHANNEL. Channels differ — email/push reach you away, in-app waits in the inbox. -->
        <div
          class="mt-2 grid grid-cols-[1fr_3.2rem_3.2rem_3.2rem] items-center gap-x-2 gap-y-1 border-t border-border pt-3"
          role="group"
          :aria-label="t('profile.notifications')"
        >
          <span aria-hidden="true"></span>
          <span
            v-for="ch in NOTIFICATION_CHANNELS"
            :key="`hdr-${ch}`"
            class="text-center text-xs font-medium text-muted"
            >{{ t(`profile.channel.${ch}`) }}</span
          >

          <template v-for="nt in NOTIFICATION_TYPES" :key="nt">
            <div class="py-1.5">
              <div class="text-sm font-medium">{{ t(`profile.notifType.${nt}`) }}</div>
              <p class="text-xs text-muted">{{ t(`profile.notifTypeHelp.${nt}`) }}</p>
            </div>
            <div
              v-for="ch in NOTIFICATION_CHANNELS"
              :key="`${nt}-${ch}`"
              class="flex justify-center"
            >
              <input
                v-model="comms.types[nt][ch]"
                type="checkbox"
                class="lp-check"
                :data-testid="`notif-${nt}-${ch}`"
                :aria-label="`${t(`profile.notifType.${nt}`)} — ${t(`profile.channel.${ch}`)}`"
                @change="onChannelToggle(nt, ch)"
              />
            </div>
          </template>
        </div>

        <!-- The digest email cadence lives outside the matrix (not a per-channel thing). -->
        <template v-if="comms.types.digest.email">
          <label
            class="mt-2 flex items-center justify-between gap-3 border-t border-border py-2 pt-3"
          >
            <span class="text-sm text-muted">{{ t("profile.cadence") }}</span>
            <select
              v-model="comms.digest_schedule.cadence"
              class="rounded-lg border border-border bg-overlay px-2 py-1 text-sm"
              @change="saveSchedule"
            >
              <option value="weekly">{{ t("profile.cadenceWeekly") }}</option>
              <option value="daily">{{ t("profile.cadenceDaily") }}</option>
            </select>
          </label>
          <label class="flex items-center justify-between gap-3 py-2">
            <span class="text-sm text-muted">{{ t("profile.pauseDigest") }}</span>
            <input
              v-model="comms.digest_schedule.paused"
              type="checkbox"
              class="lp-check"
              @change="saveSchedule"
            />
          </label>
          <p v-if="!comms.email_verified" class="mt-1 text-xs text-muted">
            {{ t("profile.emailUnverified") }}
          </p>
        </template>
      </section>

      <!-- Sign out (#1962): quiet, last, least weight — the last thing you'd do here. -->
      <button
        v-if="auth.isAuthenticated"
        type="button"
        class="mt-8 w-full rounded-2xl border border-border py-3 text-sm font-bold text-muted transition hover:text-canvas-foreground"
        @click="onSignOut"
      >
        {{ t("auth.signOut") }}
      </button>
    </div>

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onSaved" />
  </section>
</template>
