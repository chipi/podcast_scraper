<script setup lang="ts">
/**
 * "On this device" — the one surface that works offline AND signed out (operator 2026-09-23).
 *
 * The gap it closes, in the operator's words: *"what if I was not logged in? I could not log in
 * because I was offline. In that case offline mode is useless for playing things I anyway
 * downloaded already."* Every other surface is behind the login-first guard (RFC-120), and signing
 * in needs a network — so a lapsed session on a plane made the episodes on the device unreachable
 * from the device holding them.
 *
 * ## Why it is narrow, and stays narrow
 *
 * Downloads are namespaced per account precisely so a shared phone cannot show one person's
 * listening history to the next (#1905). This route reads the LAST signed-in account's registry,
 * which is a real exposure of that history to whoever is holding the unlocked phone. It was taken
 * knowingly, so the blast radius is fixed by construction rather than by intent:
 *
 * - It renders ONLY when offline AND signed out. Online it redirects to sign-in — where you can
 *   actually sign in — and signed in it redirects Home, where the full Library already is.
 * - It is READ-ONLY. No delete, no queue, no favourite, no account information, no library. The
 *   only action is play.
 * - It reads the registry and the files. It never reaches the API, which is the point.
 *
 * The alternative — keeping the session alive across an offline stretch — is ALSO implemented, and
 * is the path a returning user actually takes: `auth.hydrateFromDevice()` repaints the cached
 * identity and a transport failure never clears it. This is the fallback for when that is not
 * enough: a genuine sign-out, a cleared snapshot, a session too old to trust.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import DownloadedList from '../components/DownloadedList.vue'
import { useAuthStore } from '../stores/auth'
import { useDownloadsStore } from '../stores/downloads'
import { useOnline } from '../composables/useOnline'

const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const downloads = useDownloadsStore()
const { isOnline } = useOnline()

const checked = ref(false)
const found = ref(false)

onMounted(async () => {
  // Signed in, or online: this page has no reason to exist. Redirect rather than render, so it
  // cannot become a back-door into another account's history while the network is up.
  if (auth.hasSession) {
    void router.replace({ name: 'home' })
    return
  }
  if (isOnline.value) {
    void router.replace({ name: 'landing' })
    return
  }
  found.value = await downloads.adoptLastAccount()
  checked.value = true
})

const hasAny = computed(() => checked.value && found.value)
</script>

<template>
  <main class="mx-auto max-w-3xl px-4 py-6" data-testid="offline-downloads">
    <h1 class="font-display text-2xl font-extrabold tracking-tight text-canvas-foreground">
      {{ t('offlineDownloads.title') }}
    </h1>
    <!-- Says WHY this page looks different from the app, so it reads as a deliberate fallback
         rather than as the app having lost everything. -->
    <p class="mt-1 text-sm text-muted">{{ t('offlineDownloads.subtitle') }}</p>

    <div v-if="!checked" class="mt-6 text-sm text-muted">{{ t('catalog.loading') }}</div>
    <DownloadedList v-else-if="hasAny" class="mt-6" />
    <p v-else class="mt-6 text-sm text-muted" data-testid="offline-downloads-empty">
      {{ t('offlineDownloads.empty') }}
    </p>
  </main>
</template>
