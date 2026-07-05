<script setup lang="ts">
/**
 * Header user menu (#1128) — the avatar/icon next to the theme toggle. Shows the signed-in user's
 * name, email, and role, with a Sign out action. Lives in the header's existing `gap-3` cluster.
 */
import { computed, ref } from 'vue'
import { useAuthStore } from '../../stores/auth'

const auth = useAuthStore()
const open = ref(false)

const initial = computed(() => {
  const src = auth.user?.name || auth.user?.email || '?'
  return src.trim().charAt(0).toUpperCase() || '?'
})

const roleLabel = computed(() => {
  const r = auth.role
  return r ? r.charAt(0).toUpperCase() + r.slice(1) : ''
})

function close(): void {
  open.value = false
}
</script>

<template>
  <div class="relative">
    <button
      type="button"
      class="flex h-7 w-7 items-center justify-center rounded-full border border-border bg-elevated text-xs font-semibold text-surface-foreground hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
      :title="auth.user?.email ?? 'Account'"
      aria-haspopup="menu"
      :aria-expanded="open"
      data-testid="user-menu-button"
      @click="open = !open"
    >
      {{ initial }}
    </button>

    <!-- click-away backdrop (single, transparent) -->
    <div v-if="open" class="fixed inset-0 z-30" data-testid="user-menu-backdrop" @click="close" />

    <div
      v-if="open"
      class="absolute right-0 z-40 mt-1 w-56 rounded border border-border bg-surface p-1 text-sm shadow-lg"
      role="menu"
      data-testid="user-menu"
    >
      <div class="px-3 py-2">
        <p class="truncate font-medium text-surface-foreground">{{ auth.user?.name }}</p>
        <p class="truncate text-xs text-muted">{{ auth.user?.email }}</p>
        <span
          class="mt-1 inline-block rounded-full bg-elevated px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted"
          data-testid="user-menu-role"
        >
          {{ roleLabel }}
        </span>
      </div>
      <hr class="my-1 border-border" />
      <button
        type="button"
        class="block w-full rounded px-3 py-1.5 text-left text-muted hover:bg-overlay hover:text-surface-foreground"
        role="menuitem"
        data-testid="user-menu-signout"
        @click="auth.logout()"
      >
        Sign out
      </button>
    </div>
  </div>
</template>
