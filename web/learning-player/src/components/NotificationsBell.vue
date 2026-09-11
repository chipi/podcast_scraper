<script setup lang="ts">
/**
 * Header notification bell (UXS-011) — the in-app inbox surface (wave-I). A round icon button with
 * an unread badge; clicking opens a dropdown listing recent notifications newest-first. Opening
 * refreshes the list; clicking an item navigates to its deep link and marks it read; a "mark all
 * read" action clears the badge. This is the `in_app` delivery channel — what's waiting when you
 * open the app, as opposed to OS push which reaches you while it's closed.
 */
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'

import { useNotificationsStore } from '../stores/notifications'
import type { NotificationItem } from '../services/types'

const { t } = useI18n()
const router = useRouter()
const store = useNotificationsStore()

const open = ref(false)
const root = ref<HTMLElement | null>(null)

async function toggle(): Promise<void> {
  open.value = !open.value
  if (open.value) await store.load() // never show a stale list when the panel opens
}

function close(): void {
  open.value = false
}

// Close on an outside click or Escape — a dropdown that traps focus/click is worse than none.
function onDocClick(e: MouseEvent): void {
  if (open.value && root.value && !root.value.contains(e.target as Node)) close()
}
function onKey(e: KeyboardEvent): void {
  if (e.key === 'Escape') close()
}
onMounted(() => {
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('keydown', onKey)
})

async function openItem(n: NotificationItem): Promise<void> {
  await store.markRead(n.id)
  close()
  if (n.deep_link) void router.push(n.deep_link)
}

/** A compact relative time ("now", "3h", "2d") — the panel is a glance surface, not a log. */
function ago(createdAt: number, now = Math.floor(Date.now() / 1000)): string {
  const s = Math.max(0, now - createdAt)
  if (s < 60) return t('notifications.now')
  if (s < 3600) return t('notifications.minutesAgo', { n: Math.floor(s / 60) })
  if (s < 86400) return t('notifications.hoursAgo', { n: Math.floor(s / 3600) })
  return t('notifications.daysAgo', { n: Math.floor(s / 86400) })
}
</script>

<template>
  <div ref="root" class="relative">
    <button
      type="button"
      class="group relative inline-flex h-9 w-9 items-center justify-center rounded-full text-muted transition-colors hover:bg-overlay hover:text-canvas-foreground focus-visible:text-canvas-foreground"
      :aria-label="store.unread ? t('notifications.bellCounted', { n: store.unread }) : t('notifications.bell')"
      :aria-expanded="open"
      aria-haspopup="true"
      data-testid="notifications-bell"
      @click="toggle"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
        <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
        <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
      </svg>
      <span
        v-if="store.unread"
        aria-hidden="true"
        data-testid="notifications-badge"
        class="absolute -right-0.5 -top-0.5 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-accent px-1 text-[10px] font-bold text-accent-foreground"
      >{{ store.unread > 9 ? '9+' : store.unread }}</span>
    </button>

    <!-- The dropdown. A right-anchored panel so it never runs off a phone's right edge. -->
    <div
      v-if="open"
      data-testid="notifications-panel"
      class="absolute right-0 z-50 mt-2 w-80 max-w-[calc(100vw-1.5rem)] overflow-hidden rounded-2xl border border-border bg-elevated shadow-xl"
      role="group"
      :aria-label="t('notifications.title')"
    >
      <div class="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <span class="text-sm font-bold">{{ t('notifications.title') }}</span>
        <button
          v-if="store.unread"
          type="button"
          class="text-xs font-bold text-accent"
          data-testid="notifications-mark-all"
          @click="store.markAllRead()"
        >{{ t('notifications.markAllRead') }}</button>
      </div>

      <p v-if="!store.items.length" class="px-4 py-6 text-center text-sm text-muted">
        {{ t('notifications.empty') }}
      </p>

      <ul v-else class="max-h-[22rem] overflow-y-auto">
        <li v-for="n in store.items" :key="n.id">
          <button
            type="button"
            class="flex w-full items-start gap-2 px-4 py-3 text-left transition-colors hover:bg-overlay"
            :class="n.read ? 'opacity-70' : ''"
            data-testid="notification-item"
            @click="openItem(n)"
          >
            <span
              aria-hidden="true"
              class="mt-1.5 h-2 w-2 shrink-0 rounded-full"
              :class="n.read ? 'bg-transparent' : 'bg-accent'"
            ></span>
            <span class="min-w-0 flex-1">
              <span class="block text-sm font-medium leading-snug">{{ n.title }}</span>
              <span v-if="n.body" class="mt-0.5 block truncate text-xs text-muted">{{ n.body }}</span>
              <span class="mt-0.5 block text-[11px] text-muted">{{ ago(n.created_at) }}</span>
            </span>
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>
