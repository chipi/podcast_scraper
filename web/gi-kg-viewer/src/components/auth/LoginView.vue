<script setup lang="ts">
/**
 * Login landing (#1128). When the MOCK provider is on (dev), a picker lets you sign in as a seeded
 * user (their role is preserved) or type a custom name (lands as creator). With a real provider, it
 * shows the normal Sign-in button. New users get creator access; ask an admin for elevated roles.
 */
import { onMounted, ref } from 'vue'
import { getDevUsers, type DevUser } from '../../api/authApi'
import { useAuthStore } from '../../stores/auth'

const auth = useAuthStore()
const devEnabled = ref(false)
const devUsers = ref<DevUser[]>([])
const custom = ref('')

onMounted(async () => {
  const { enabled, users } = await getDevUsers()
  devEnabled.value = enabled
  devUsers.value = users
})

function signInCustom(): void {
  const name = custom.value.trim()
  if (name) auth.loginAs(name, 'creator') // a fresh custom identity → creator
}
</script>

<template>
  <div class="flex h-dvh items-center justify-center bg-canvas px-4 text-canvas-foreground">
    <div class="w-full max-w-sm rounded-lg border border-border bg-surface p-8 text-center shadow-sm">
      <h1 class="text-xl font-semibold tracking-tight text-surface-foreground">
        Podcast Intelligence Platform
        <span class="ml-1 align-top text-xs font-normal text-muted">v2</span>
      </h1>
      <p class="mt-2 text-sm text-muted">Sign in to explore the knowledge graph.</p>

      <!-- Dev (mock provider): pick a predefined user, or type a custom one. -->
      <template v-if="devEnabled">
        <div v-if="devUsers.length" class="mt-5 space-y-1.5 text-left" data-testid="dev-user-list">
          <p class="text-xs font-medium uppercase tracking-wide text-muted">Sign in as</p>
          <button
            v-for="u in devUsers"
            :key="u.hint"
            type="button"
            class="flex w-full items-center justify-between rounded border border-border px-3 py-2 text-sm hover:bg-overlay"
            :data-testid="`dev-user-${u.hint}`"
            @click="auth.loginAs(u.hint)"
          >
            <span class="font-medium text-surface-foreground">{{ u.name }}</span>
            <span class="rounded-full bg-elevated px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted">{{ u.role }}</span>
          </button>
        </div>

        <form class="mt-4 flex gap-2" @submit.prevent="signInCustom">
          <input
            v-model="custom"
            type="text"
            placeholder="or a custom name…"
            class="min-w-0 flex-1 rounded border border-border bg-canvas px-2 py-1.5 text-sm text-canvas-foreground"
            data-testid="dev-custom-input"
          />
          <button
            type="submit"
            :disabled="!custom.trim()"
            class="rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            data-testid="dev-custom-submit"
          >
            Go
          </button>
        </form>
        <p class="mt-3 text-[11px] text-muted">Dev sign-in (mock OAuth). Picked users keep their role; a custom name lands as creator.</p>
      </template>

      <!-- Real provider: the normal sign-in button. -->
      <template v-else>
        <button
          type="button"
          class="mt-6 w-full rounded bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          data-testid="login-button"
          @click="auth.login()"
        >
          Sign in
        </button>
        <p class="mt-4 text-xs text-muted">New here? You'll get creator access. Ask an admin for elevated roles.</p>
      </template>
    </div>
  </div>
</template>
