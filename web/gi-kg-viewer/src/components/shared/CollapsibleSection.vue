<script setup lang="ts">
import { ref } from 'vue'

const props = withDefaults(
  defineProps<{
    title: string
    summary?: string
    defaultOpen?: boolean
  }>(),
  { summary: '', defaultOpen: true },
)

const open = ref(props.defaultOpen)
</script>

<template>
  <div class="rounded-lg border border-border bg-surface">
    <div class="flex w-full min-w-0 items-center gap-1.5 px-3 py-2 hover:bg-overlay/50">
      <button
        type="button"
        class="flex shrink-0 items-center gap-2 text-left text-sm font-medium text-surface-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
        :aria-expanded="open"
        @click="open = !open"
      >
        <svg
          class="h-3 w-3 shrink-0 text-muted transition-transform"
          :class="{ 'rotate-90': open }"
          viewBox="0 0 12 12"
          fill="currentColor"
        >
          <path d="M4 2l4 4-4 4z" />
        </svg>
        <span>{{ title }}</span>
      </button>
      <div v-if="$slots.actions" class="shrink-0" @click.stop>
        <slot name="actions" />
      </div>
      <button
        v-if="!open && summary"
        type="button"
        class="ml-auto min-w-0 max-w-[min(24rem,55vw)] truncate text-left text-[10px] font-normal text-muted hover:text-surface-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
        aria-expanded="false"
        :aria-label="`Expand section: ${summary}`"
        @click="open = true"
      >
        {{ summary }}
      </button>
    </div>
    <div
      v-if="$slots.subtitle"
      class="border-t border-border px-3 py-1.5 text-xs leading-snug text-muted"
    >
      <slot name="subtitle" />
    </div>
    <div
      v-show="open"
      class="border-t border-border px-3 pb-3 pt-2"
    >
      <slot />
    </div>
  </div>
</template>
