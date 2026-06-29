<script setup lang="ts">
/**
 * Login landing (#1128) — shown when no one is signed in. Full-page redirect into the shared OAuth
 * flow; new users land as `creator` (viewer access). There is no self-service sign-up of roles.
 */
import { useAuthStore } from '../../stores/auth'

const auth = useAuthStore()
</script>

<template>
  <div class="flex h-dvh items-center justify-center bg-canvas px-4 text-canvas-foreground">
    <div class="w-full max-w-sm rounded-lg border border-border bg-surface p-8 text-center shadow-sm">
      <h1 class="text-xl font-semibold tracking-tight text-surface-foreground">
        Podcast Intelligence Platform
        <span class="ml-1 align-top text-xs font-normal text-muted">v2</span>
      </h1>
      <p class="mt-2 text-sm text-muted">Sign in to explore the knowledge graph.</p>
      <button
        type="button"
        class="mt-6 w-full rounded bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        data-testid="login-button"
        @click="auth.login()"
      >
        Sign in
      </button>
      <p class="mt-4 text-xs text-muted">
        New here? You'll get creator access. Ask an admin for elevated roles.
      </p>
    </div>
  </div>
</template>
