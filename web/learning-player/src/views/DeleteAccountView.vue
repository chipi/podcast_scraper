<script setup lang="ts">
/**
 * Delete account (#2273) — App Store 5.1.1(v) and Google Play's data-deletion requirement.
 *
 * Decisions (operator, 2026-10-04): delete IMMEDIATELY (no grace period), confirm by TYPING
 * "DELETE", and keep the address on the invite allowlist so the person can come back.
 *
 * Public route, two modes:
 *   - signed in: says WHICH account goes (a person can hold separate Google, Apple and email
 *     accounts for one address — only the signed-in one is deleted), lists what is and is not
 *     removed, and deletes on a typed confirmation;
 *   - signed out: the web address Play's listing points at — how to delete, and who to email
 *     when signing in is not possible.
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import { deleteAccount } from '../services/api'
import { CACHE_KEYS, clearCached } from '../services/contentCache'
import { useAuthStore } from '../stores/auth'

const { t, te } = useI18n()
const router = useRouter()
const auth = useAuthStore()

const SUPPORT_EMAIL = 'info@closelistening.app'

const typed = ref('')
const deleting = ref(false)
const failed = ref(false)

const confirmed = computed(() => typed.value.trim() === t('deleteAccount.confirmWord'))
const providerLabel = computed(() => {
  const key = `deleteAccount.providers.${auth.user?.provider ?? ''}`
  return te(key) ? t(key) : (auth.user?.provider ?? '')
})

async function onDelete(): Promise<void> {
  if (!confirmed.value || deleting.value) return
  deleting.value = true
  failed.value = false
  try {
    await deleteAccount(t('deleteAccount.confirmWord'))
  } catch {
    deleting.value = false
    failed.value = true
    return
  }
  // The account is gone server-side; now drop everything this device holds for it — the same
  // order as sign-out (#1909): cached content first, then the identity.
  await clearCached(CACHE_KEYS)
  await auth.logout()
  await router.replace({ name: 'landing', query: { deleted: '1' } })
}
</script>

<template>
  <section class="lp-page lp-focus" data-testid="delete-account-view">
    <h1 class="mb-3 font-display text-3xl font-extrabold tracking-tight">
      {{ t('deleteAccount.title') }}
    </h1>

    <template v-if="auth.isAuthenticated && auth.user">
      <p class="mb-5 text-sm" data-testid="delete-account-who">
        {{ t('deleteAccount.signedInAs', { provider: providerLabel, email: auth.user.email }) }}
      </p>

      <h2 class="mb-1 text-sm font-bold">{{ t('deleteAccount.removedTitle') }}</h2>
      <p class="mb-5 text-sm text-muted">{{ t('deleteAccount.removed') }}</p>

      <h2 class="mb-1 text-sm font-bold">{{ t('deleteAccount.keptTitle') }}</h2>
      <ul class="mb-6 list-disc space-y-1 pl-5 text-sm text-muted">
        <li>{{ t('deleteAccount.keptAnalytics') }}</li>
        <li>{{ t('deleteAccount.keptAccess') }}</li>
        <li>{{ t('deleteAccount.keptOther') }}</li>
      </ul>

      <h2 class="mb-1 text-sm font-bold">{{ t('deleteAccount.partialTitle') }}</h2>
      <p class="mb-6 text-sm text-muted" data-testid="delete-account-partial">{{ t('deleteAccount.partialBody') }}</p>

      <form @submit.prevent="onDelete">
        <label class="mb-1 block text-sm font-bold" for="delete-confirm">
          {{ t('deleteAccount.confirmLabel') }}
        </label>
        <input
          id="delete-confirm"
          v-model="typed"
          type="text"
          autocomplete="off"
          autocapitalize="characters"
          spellcheck="false"
          class="mb-4 w-full rounded-full border border-border bg-canvas px-4 py-2 text-sm"
          data-testid="delete-account-confirm"
        />
        <button
          type="submit"
          :disabled="!confirmed || deleting"
          class="w-full rounded-full bg-danger py-3 font-bold text-canvas disabled:opacity-40"
          data-testid="delete-account-submit"
        >
          {{ deleting ? t('deleteAccount.deleting') : t('deleteAccount.submit') }}
        </button>
        <p v-if="failed" class="mt-3 text-sm text-danger" role="alert" data-testid="delete-account-error">
          {{ t('deleteAccount.error') }}
        </p>
      </form>
    </template>

    <template v-else>
      <p class="mb-5 text-sm" data-testid="delete-account-signed-out">
        {{ t('deleteAccount.signedOutBody', { email: SUPPORT_EMAIL }) }}
      </p>
      <h2 class="mb-1 text-sm font-bold">{{ t('deleteAccount.partialTitle') }}</h2>
      <p class="mb-5 text-sm text-muted" data-testid="delete-account-partial">{{ t('deleteAccount.partialBody') }}</p>
      <RouterLink
        :to="{ name: 'login', query: { redirect: '/account/delete' } }"
        class="inline-block rounded-full border border-border px-5 py-2 text-sm font-bold no-underline"
      >
        {{ t('deleteAccount.signIn') }}
      </RouterLink>
    </template>
  </section>
</template>
