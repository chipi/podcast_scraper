<script setup lang="ts">
/**
 * Profile / account — where the signed-in user sees who they are and edits their personalization,
 * starting with their interest topics (chosen at sign-in via the onboarding card). Auth-gated.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
defineOptions({ name: 'ProfileView' }) // stable name for <keep-alive :include> (App.vue)
import { getComms, getMyStats, getTopClusters, getUserInterests, putComms } from '../services/api'
import type { CommsSettings, InterestCluster, UserStats } from '../services/types'
import { disablePush, enablePush } from '../composables/usePushSubscription'
import { useAuthStore } from '../stores/auth'
import { useUserPreferencesStore } from '../stores/userPreferences'
import InterestsPicker from '../components/InterestsPicker.vue'
import Sparkline from '../components/Sparkline.vue'
import ConnectedAgents from '../components/ConnectedAgents.vue'

const { t } = useI18n()
const auth = useAuthStore()
const userPrefs = useUserPreferencesStore()

// How "Your Week" lays out on the home page — a synced per-user preference, shared with the inline
// "Show more / Show less" toggle on the home section (#1412). Independent of the email toggle below.
const YOUR_WEEK_LAYOUT_KEY = 'lp.yourweek.layout'
const yourWeekLayout = ref<'compact' | 'full'>('compact')
function setYourWeekLayout(v: 'compact' | 'full'): void {
  yourWeekLayout.value = v
  void userPrefs.set(YOUR_WEEK_LAYOUT_KEY, v)
}

const interests = ref<string[]>([])
const clusters = ref<InterestCluster[]>([])
const pickerOpen = ref(false)

// Listening analytics (UXS-014) — the user's own play history, summarized.
const stats = ref<UserStats | null>(null)
const hours = computed(() => (stats.value ? stats.value.listening_seconds / 3600 : 0))
const hoursLabel = computed(() => (hours.value >= 10 ? Math.round(hours.value) : hours.value.toFixed(1)))
const series = computed(() => stats.value?.daily.map((d) => d.count) ?? [])
const hasStats = computed(() => !!stats.value && stats.value.episodes > 0)

// Map saved interest tokens → human labels. Clusters resolve via the top-cluster set; topics and
// people (followed from entity cards) de-slug from their id (`topic:personal-growth` → "personal
// growth"). `kind` drives the chip hue so people read distinct from topics.
const interestLabels = computed(() => {
  const byId = new Map(clusters.value.map((c) => [c.id, c.label]))
  return interests.value.map((id) => ({
    id,
    kind: id.startsWith('person:') ? 'person' : 'topic',
    label: byId.get(id) ?? id.replace(/^(tc|topic|person):/, '').replace(/-/g, ' '),
  }))
})

// Delivery consent (PRD-046 FR1 / #1414) — the "Your Week" digest + push nudges.
const comms = ref<CommsSettings | null>(null)

async function load(): Promise<void> {
  const [ints, tops, st, cm] = await Promise.all([
    getUserInterests().catch(() => [] as string[]),
    getTopClusters(50).catch(() => [] as InterestCluster[]),
    getMyStats().catch(() => null),
    getComms().catch(() => null),
  ])
  interests.value = ints
  clusters.value = tops
  stats.value = st
  comms.value = cm
  await userPrefs.hydrate()
  const layoutPref = userPrefs.get<string>(YOUR_WEEK_LAYOUT_KEY)
  if (layoutPref === 'full' || layoutPref === 'compact') yourWeekLayout.value = layoutPref
}

function onSaved(ids: string[]): void {
  interests.value = ids
}

// Persist a whole section (server fills defaults on unset fields, so never send a partial).
async function saveDigest(): Promise<void> {
  if (comms.value) comms.value = await putComms({ digest: comms.value.digest })
}

// Push needs a real browser subscription, not just a flag. Enabling registers the subscription
// (which enables the channel server-side); if the browser can't, revert the toggle. Disabling
// unregisters + clears the flag.
async function onPushToggle(): Promise<void> {
  if (!comms.value) return
  if (comms.value.push.enabled) {
    // enablePush registers the subscription (which enables the channel server-side). If the browser
    // can't (false) OR the register POST throws, revert the toggle so the UI never claims "on"
    // without a real subscription.
    let ok = false
    try {
      ok = await enablePush()
    } catch {
      ok = false
    }
    if (!ok) comms.value = await putComms({ push: { enabled: false } })
  } else {
    await disablePush()
    comms.value = await putComms({ push: { enabled: false } })
  }
}

onMounted(load)
</script>

<template>
  <section class="max-w-2xl">
    <div class="mb-1 flex items-start justify-between gap-3">
      <h1 class="font-display text-3xl font-extrabold tracking-tight">{{ t('profile.title') }}</h1>
      <RouterLink
        :to="{ name: 'settings' }"
        class="shrink-0 rounded-full border border-border p-2 text-muted no-underline transition hover:bg-overlay hover:text-canvas-foreground"
        :aria-label="t('settings.title')"
        :title="t('settings.title')"
        data-testid="profile-settings-link"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </RouterLink>
    </div>
    <p class="mb-6 text-muted">{{ auth.user?.name }}<span v-if="auth.user?.email"> · {{ auth.user?.email }}</span></p>

    <section class="rounded-2xl border border-border p-5">
      <div class="mb-3 flex items-center justify-between gap-2">
        <h2 class="lp-section">{{ t('profile.interests') }}</h2>
        <button type="button" class="text-sm font-bold text-accent" @click="pickerOpen = true">
          {{ t('profile.editInterests') }}
        </button>
      </div>
      <p class="mb-3 text-sm text-muted">{{ t('profile.interestsHelp') }}</p>
      <div v-if="interestLabels.length" class="flex flex-wrap gap-1.5">
        <span
          v-for="i in interestLabels"
          :key="i.id"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs"
          :class="i.kind === 'person' ? 'text-person' : 'text-topic'"
        >{{ i.label }}</span>
      </div>
      <p v-else class="text-sm text-muted">{{ t('profile.noInterests') }}</p>
    </section>

    <!-- Delivery consent (PRD-046 FR1 / #1414) — the "Your Week" digest + push nudges. -->
    <section v-if="comms" class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-1">{{ t('profile.notifications') }}</h2>
      <p class="mb-3 text-sm text-muted">{{ t('profile.notificationsHelp') }}</p>

      <!-- How Your Week lays out on your home — the in-app view is the primary surface. -->
      <div class="flex items-center justify-between gap-3 py-2">
        <span class="text-sm font-medium">{{ t('profile.yourWeekLayout') }}</span>
        <div class="flex overflow-hidden rounded-full border border-border">
          <button
            type="button"
            class="px-3 py-1 text-sm font-semibold"
            :class="yourWeekLayout === 'compact' ? 'bg-accent text-accent-foreground' : 'text-muted'"
            @click="setYourWeekLayout('compact')"
          >
            {{ t('profile.yourWeekCompact') }}
          </button>
          <button
            type="button"
            class="px-3 py-1 text-sm font-semibold"
            :class="yourWeekLayout === 'full' ? 'bg-accent text-accent-foreground' : 'text-muted'"
            @click="setYourWeekLayout('full')"
          >
            {{ t('profile.yourWeekFull') }}
          </button>
        </div>
      </div>
      <p class="mb-1 text-xs text-muted">{{ t('profile.yourWeekLayoutHelp') }}</p>

      <!-- The email edge: Your Week in your inbox for when you don't open the app. -->
      <label class="mt-2 flex items-center justify-between gap-3 border-t border-border py-2 pt-3">
        <span class="text-sm font-medium">{{ t('profile.digestEmail') }}</span>
        <input
          v-model="comms.digest.enabled"
          type="checkbox"
          class="h-5 w-5"
          @change="saveDigest"
        />
      </label>

      <template v-if="comms.digest.enabled">
        <label class="flex items-center justify-between gap-3 py-2">
          <span class="text-sm text-muted">{{ t('profile.cadence') }}</span>
          <select
            v-model="comms.digest.cadence"
            class="rounded-lg border border-border bg-overlay px-2 py-1 text-sm"
            @change="saveDigest"
          >
            <option value="weekly">{{ t('profile.cadenceWeekly') }}</option>
            <option value="daily">{{ t('profile.cadenceDaily') }}</option>
          </select>
        </label>
        <label class="flex items-center justify-between gap-3 py-2">
          <span class="text-sm text-muted">{{ t('profile.pauseDigest') }}</span>
          <input
            v-model="comms.digest.paused"
            type="checkbox"
            class="h-5 w-5"
            @change="saveDigest"
          />
        </label>
        <p v-if="!comms.email_verified" class="mt-1 text-xs text-muted">
          {{ t('profile.emailUnverified') }}
        </p>
      </template>

      <label class="mt-2 flex items-center justify-between gap-3 border-t border-border py-2 pt-3">
        <span class="text-sm font-medium">{{ t('profile.pushNudges') }}</span>
        <input v-model="comms.push.enabled" type="checkbox" class="h-5 w-5" @change="onPushToggle" />
      </label>
    </section>

    <!-- Listening analytics (UXS-014) — derived entirely from this user's own play history. -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('stats.title') }}</h2>
      <template v-if="hasStats">
        <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div class="rounded-xl bg-overlay p-4">
            <div class="flex items-baseline gap-1">
              <span class="font-display text-3xl font-extrabold leading-none text-accent">{{ stats!.day_streak }}</span>
              <span v-if="stats!.day_streak > 0" aria-hidden="true">🔥</span>
            </div>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.streak') }}</div>
          </div>
          <div class="rounded-xl bg-overlay p-4">
            <span class="font-display text-3xl font-extrabold leading-none">{{ stats!.episodes }}</span>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.episodes') }}</div>
          </div>
          <div class="rounded-xl bg-overlay p-4">
            <span class="font-display text-3xl font-extrabold leading-none">{{ stats!.shows }}</span>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.shows') }}</div>
          </div>
          <div class="rounded-xl bg-overlay p-4">
            <span class="font-display text-3xl font-extrabold leading-none">{{ hoursLabel }}<span class="text-lg">h</span></span>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.hours') }}</div>
          </div>
        </div>
        <div class="mt-3 rounded-xl bg-overlay p-4">
          <div class="mb-2 flex items-baseline justify-between">
            <span class="text-xs font-medium text-muted">{{ t('stats.overTime') }}</span>
            <span class="text-xs text-muted">{{ t('stats.activeDays', stats!.active_days, { named: { count: stats!.active_days } }) }}</span>
          </div>
          <Sparkline :values="series" :width="320" :height="44" class="block w-full text-accent" />
        </div>
      </template>
      <p v-else class="text-sm text-muted">{{ t('stats.empty') }}</p>
    </section>

    <!-- Connected agents (RFC-112 §5) — only for users with the mcp_access entitlement. -->
    <ConnectedAgents v-if="auth.user?.mcp_access" />

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onSaved" />
  </section>
</template>
