<script setup lang="ts">
/**
 * List ⇄ grid view toggle — ONE control shared by every browsable list (Catalog / Browse ›
 * Episodes, Browse › Shows). Was hand-copied verbatim (same two buttons, same inline SVGs, same
 * class strings) in `CatalogView` and `ShowBrowseView`; a copy is how they drift apart. `v-model`
 * carries the mode; `lp-tap` keeps the 44px hit box the design system enforces.
 *
 * testids default to `view-list` / `view-grid`; a surface with its own e2e selectors (ShowBrowse
 * uses `show-view-*`) overrides them via props, so adopting this changes no test.
 */
import { useI18n } from "vue-i18n"

withDefaults(
  defineProps<{ modelValue: "list" | "grid"; testidList?: string; testidGrid?: string }>(),
  { testidList: "view-list", testidGrid: "view-grid" }
)
const emit = defineEmits<{ (e: "update:modelValue", v: "list" | "grid"): void }>()
const { t } = useI18n()
</script>

<template>
  <div class="flex shrink-0 gap-1" role="group" :aria-label="t('list.view')">
    <button
      type="button"
      :data-testid="testidList"
      class="lp-tap flex h-9 w-9 items-center justify-center rounded-full border transition"
      :class="
        modelValue === 'list'
          ? 'border-accent text-accent'
          : 'border-border text-muted hover:text-canvas-foreground'
      "
      :aria-pressed="modelValue === 'list'"
      :aria-label="t('list.viewList')"
      :title="t('list.viewList')"
      @click="emit('update:modelValue', 'list')"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        class="h-4 w-4"
        aria-hidden="true"
      >
        <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
      </svg>
    </button>
    <button
      type="button"
      :data-testid="testidGrid"
      class="lp-tap flex h-9 w-9 items-center justify-center rounded-full border transition"
      :class="
        modelValue === 'grid'
          ? 'border-accent text-accent'
          : 'border-border text-muted hover:text-canvas-foreground'
      "
      :aria-pressed="modelValue === 'grid'"
      :aria-label="t('list.viewGrid')"
      :title="t('list.viewGrid')"
      @click="emit('update:modelValue', 'grid')"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        class="h-4 w-4"
        aria-hidden="true"
      >
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
    </button>
  </div>
</template>
