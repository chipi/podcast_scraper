<script setup lang="ts">
/**
 * No-access screen (#1128) — shown when a signed-in user is a `listener` (e.g. a Learning Player
 * user) who has no viewer access yet. They can sign out; an admin must grant them a viewer role.
 */
import { useAuthStore } from '../../stores/auth'

const auth = useAuthStore()
</script>

<template>
  <div class="flex h-dvh items-center justify-center bg-canvas px-4 text-canvas-foreground">
    <div class="w-full max-w-sm rounded-lg border border-border bg-surface p-8 text-center shadow-sm">
      <h1 class="text-lg font-semibold tracking-tight text-surface-foreground">
        Awaiting access
      </h1>
      <p class="mt-3 text-sm text-muted" data-testid="no-access-message">
        You're signed in as
        <span class="font-medium text-surface-foreground">{{ auth.user?.email }}</span
        >, but your account doesn't have viewer access yet. Ask an admin to grant you a role.
      </p>
      <button
        type="button"
        class="mt-6 w-full rounded border border-border px-4 py-2 text-sm text-muted hover:bg-overlay hover:text-surface-foreground"
        data-testid="no-access-signout"
        @click="auth.logout()"
      >
        Sign out
      </button>
    </div>
  </div>
</template>
