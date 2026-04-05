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
    <button
      type="button"
      class="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium text-surface-foreground hover:bg-overlay/50"
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
      <span
        v-if="!open && summary"
        class="ml-auto truncate text-[10px] font-normal text-muted"
      >
        {{ summary }}
      </span>
    </button>
    <div
      v-show="open"
      class="border-t border-border px-3 pb-3 pt-2"
    >
      <slot />
    </div>
  </div>
</template>
