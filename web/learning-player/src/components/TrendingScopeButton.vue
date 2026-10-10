<script setup lang="ts">
/**
 * Mine ⇄ Everyone — the ONE "mine" switch (operator 2026-10-09). Same remembered choice everywhere
 * (`useTrendingScope`): Discover's page header, Home's Trends and the Search results read and write
 * it, so flipping it on one flips it on all. "Mine" means the listener's own world — one meaning
 * per kind of item (ADR-162).
 *
 * Icon circle; active (accent) = Mine. Signed out there is no "mine", so the switch hides.
 */
import { useI18n } from "vue-i18n"
import { useTrendingScope } from "../composables/useTrendingScope"
import { useAuthStore } from "../stores/auth"

withDefaults(defineProps<{ testid?: string }>(), { testid: "discover-scope" })

const { t } = useI18n()
const auth = useAuthStore()
const { scope, setScope } = useTrendingScope()
</script>

<template>
  <button
    v-if="auth.isAuthenticated"
    type="button"
    :data-testid="testid"
    class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border transition"
    :class="scope === 'mine' ? 'border-accent bg-accent text-accent-foreground' : 'border-border text-muted hover:text-canvas-foreground'"
    :aria-pressed="scope === 'mine'"
    :aria-label="t('home.trendingScopeLabel')"
    :title="scope === 'mine' ? t('home.trendingScopeMine') : t('home.trendingScopeAll')"
    @click="setScope(scope === 'mine' ? 'corpus' : 'mine')"
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></svg>
    <!-- A NAME inside the button: the only other child is a hidden svg, and Android System WebView
         then announces it as an unnamed "Button" (2026-09-25, AccessibleNameAuditTests). -->
    <span class="sr-only">{{ scope === 'mine' ? t('home.trendingScopeMine') : t('home.trendingScopeAll') }}</span>
  </button>
</template>
