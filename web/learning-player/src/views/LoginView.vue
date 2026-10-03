<script setup lang="ts">
/**
 * Auth entry (C2/#1081). One view, two modes (`?mode=signup` vs sign-in) — both drive the
 * same OAuth flow: with open signup the provider get-or-creates the account, so "sign up" and
 * "sign in" converge on the same redirect. A link toggles between the two framings.
 *
 * Dev (mock provider): a picker lets you sign in as a seeded user or a custom name (#1128).
 */
import { Capacitor } from '@capacitor/core'
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { getDevUsers, requestMagicLink, type DevUser } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { safeInternalPath } from '../utils/redirect'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const isSignup = computed(() => route.query.mode === 'signup')
const isIos = Capacitor.getPlatform() === 'ios'
// Same-origin post-login target (login-first #2009); null when absent/unsafe.
const redirectTarget = computed(() => safeInternalPath(route.query.redirect) ?? undefined)

// Leave the login view once the session resolves. The web flow navigates away via a full-page OAuth
// redirect, but the native deep-link flow (#1310) returns IN-app — the auth store updates but the
// router doesn't move — so LoginView must route itself to the guard's `redirect` target (or home).
// `immediate` also bounces an already-signed-in visitor who lands on /login. Only same-origin paths.
watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (!authed) return
    const dest = redirectTarget.value ?? { name: 'home' }
    void router.replace(dest)
  },
  { immediate: true },
)

const devEnabled = ref(false)
const devUsers = ref<DevUser[]>([])
const custom = ref('')

onMounted(async () => {
  const { enabled, users } = await getDevUsers()
  devEnabled.value = enabled
  devUsers.value = users
})

// --- Email magic link (#2272) -----------------------------------------------------------------
//
// A second way in, for people who will not create a Google account. Same mechanism for both
// framings on this screen: the link creates the account on first use and signs in afterwards, so
// "create account" and "sign in" differ only in the copy and in where the server lands you.
const emailMode = ref(false)
const email = ref('')
const sending = ref(false)
//: Set once a request has been accepted. Deliberately NOT cleared by a failure path that would
//: reveal anything about the address — see `emailSent` in the template.
const emailSent = ref(false)
const emailError = ref(false)

const emailLooksValid = computed(() => {
  const value = email.value.trim()
  // Shape only. Anything stricter rejects real addresses, and the server is the one that decides
  // whether an address is deliverable.
  return value.length >= 3 && value.includes('@') && !value.startsWith('@') && !value.endsWith('@')
})

async function sendMagicLink(): Promise<void> {
  if (!emailLooksValid.value || sending.value) return
  sending.value = true
  emailError.value = false
  const ok = await requestMagicLink(email.value.trim(), redirectTarget.value)
  sending.value = false
  // `ok` is true for ANY accepted request — the server answers identically for an address it has
  // never seen and one on the allowlist. The UI must not add a distinction the API refuses to make:
  // "check your inbox" is the honest message even when nothing was sent, because telling the person
  // otherwise would tell a stranger whether an address has an account here.
  if (ok) emailSent.value = true
  else emailError.value = true
}

function resetEmail(): void {
  emailSent.value = false
  emailError.value = false
  email.value = ''
}

function signInCustom(): void {
  const name = custom.value.trim()
  if (name) auth.login(name, redirectTarget.value)
}
</script>

<template>
  <section class="max-w-md">
    <span class="lp-kicker">{{ t('app.tagline') }}</span>
    <h1 class="mb-2 mt-1 font-display text-3xl font-extrabold tracking-tight">
      {{ isSignup ? t('auth.signupTitle') : t('auth.loginTitle') }}
    </h1>
    <p class="mb-6 text-sm text-muted">
      {{ isSignup ? t('auth.signupTagline') : t('auth.loginTagline') }}
    </p>

    <!-- Dev (mock provider): pick a predefined user, or type a custom one. -->
    <div v-if="devEnabled">
      <div v-if="devUsers.length" class="space-y-1.5" data-testid="dev-user-list">
        <p class="text-xs font-medium uppercase tracking-wide text-muted">Sign in as</p>
        <button
          v-for="u in devUsers"
          :key="u.hint"
          type="button"
          class="flex w-full items-center justify-between rounded-xl border border-border px-4 py-2.5 text-left hover:bg-surface"
          :data-testid="`dev-user-${u.hint}`"
          @click="auth.login(u.hint, redirectTarget)"
        >
          <span class="font-bold">{{ u.name }}</span>
          <span class="rounded-full bg-surface px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted">{{ u.role }}</span>
        </button>
      </div>

      <form class="mt-4 flex gap-2" @submit.prevent="signInCustom">
        <input
          v-model="custom"
          type="text"
          placeholder="or a custom name…"
          class="min-w-0 flex-1 rounded-full border border-border bg-canvas px-4 py-2 text-sm"
          data-testid="dev-custom-input"
        />
        <button
          type="submit"
          :disabled="!custom.trim()"
          class="rounded-full bg-accent px-5 py-2 font-bold text-accent-foreground disabled:opacity-50"
          data-testid="dev-custom-submit"
        >
          {{ t('auth.signIn') }}
        </button>
      </form>
      <p class="mt-3 text-xs text-muted">Dev sign-in (mock OAuth) — picked users keep their role.</p>
    </div>

    <!-- Real provider: Google. Built to Google's "Sign in with Google" branding guidelines, dark
         theme — Google's fixed fill, 1px inside stroke and text colours, 14/20 medium, pill shape,
         the official gradient G at 20px (cropped unmodified from Google's signin-assets.zip), and the
         platform padding (Android/Web 12·10·12, iOS 16·12·16). The text is one of the three strings
         Google allows; "Sign in" alone sat above "Email me a sign-in link" and did not say which
         door it was. -->
    <button
      v-else
      type="button"
      class="lp-gsi inline-flex h-10 items-center rounded-full"
      :class="isIos ? 'pl-4 pr-4' : 'pl-3 pr-3'"
      data-testid="signin-button"
      @click="auth.login(undefined, redirectTarget)"
    >
      <img
        src="/brand/google-g-dark.png"
        alt=""
        width="20"
        height="20"
        class="size-5 shrink-0"
        :class="isIos ? 'mr-3' : 'mr-2.5'"
      />
      {{ isSignup ? t('auth.signUpWithGoogle') : t('auth.signInWithGoogle') }}
    </button>

    <!-- Email magic link: the second front door (#2272). Offered on BOTH framings, because it
         creates an account just as readily as it signs one in. -->
    <div class="mt-6 border-t border-border pt-5">
      <template v-if="emailSent">
        <p class="font-bold" data-testid="magic-link-sent">{{ t('auth.magicLinkSentTitle') }}</p>
        <p class="mt-1 text-sm text-muted">
          {{ t('auth.magicLinkSentBody', { email: email.trim() }) }}
        </p>
        <button
          type="button"
          class="mt-3 text-sm font-bold text-accent underline"
          data-testid="magic-link-reset"
          @click="resetEmail"
        >
          {{ t('auth.magicLinkUseAnother') }}
        </button>
      </template>

      <template v-else-if="emailMode">
        <form class="flex gap-2" @submit.prevent="sendMagicLink">
          <input
            v-model="email"
            type="email"
            autocomplete="email"
            inputmode="email"
            :placeholder="t('auth.magicLinkPlaceholder')"
            class="min-w-0 flex-1 rounded-full border border-border bg-canvas px-4 py-2 text-sm"
            data-testid="magic-link-input"
          />
          <button
            type="submit"
            :disabled="!emailLooksValid || sending"
            class="rounded-full bg-accent px-5 py-2 font-bold text-accent-foreground disabled:opacity-50"
            data-testid="magic-link-submit"
          >
            {{ sending ? t('auth.magicLinkSending') : t('auth.magicLinkSend') }}
          </button>
        </form>
        <p v-if="emailError" class="mt-2 text-sm text-muted" data-testid="magic-link-error">
          {{ t('auth.magicLinkError') }}
        </p>
        <p class="mt-2 text-xs text-muted">{{ t('auth.magicLinkHint') }}</p>
      </template>

      <button
        v-else
        type="button"
        class="w-full rounded-full border border-border px-6 py-3 font-bold"
        data-testid="magic-link-button"
        @click="emailMode = true"
      >
        {{ t('auth.magicLinkCta') }}
      </button>
    </div>

    <p class="mt-5 text-sm text-muted">
      <template v-if="isSignup">
        {{ t('auth.haveAccount') }}
        <RouterLink :to="{ name: 'login' }" class="font-bold text-accent no-underline">
          {{ t('auth.signIn') }}
        </RouterLink>
      </template>
      <template v-else>
        {{ t('auth.newHere') }}
        <RouterLink :to="{ name: 'login', query: { mode: 'signup' } }" class="font-bold text-accent no-underline">
          {{ t('auth.signUp') }}
        </RouterLink>
      </template>
    </p>
  </section>
</template>

<style scoped>
/* Google's dark-theme values (tokens.css `--lp-gsi-*`), fixed by the branding guidelines. */
.lp-gsi {
  background: var(--lp-gsi-fill);
  box-shadow: inset 0 0 0 1px var(--lp-gsi-stroke);
  color: var(--lp-gsi-text);
  font-family: 'Google Sans', Roboto, system-ui, sans-serif;
  font-size: 14px;
  line-height: 20px;
  font-weight: 500;
}
</style>
