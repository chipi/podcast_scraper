<script setup lang="ts">
/**
 * List toolbar (UXS-014) — the ONE shared filter / sort / search header for big lists. Collapsed by
 * default to a compact pill (keeps lists clean); expands to small, on-brand pill controls. An accent
 * dot on the pill signals active filters while collapsed. Presentational — two-way-binds via v-model.
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const search = defineModel<string>('search', { default: '' })
const sort = defineModel<string>('sort', { default: 'newest' })
const filter = defineModel<string>('filter', { default: 'all' })
const show = defineModel<string>('show', { default: '' })

const { t } = useI18n()
withDefaults(
  defineProps<{ showFilter?: boolean; count?: string; shows?: { id: string; label: string }[] }>(),
  { showFilter: true, shows: () => [] },
)

const open = ref(false)
const active = computed(
  () =>
    search.value.trim() !== '' ||
    sort.value !== 'newest' ||
    filter.value !== 'all' ||
    show.value !== '',
)
</script>

<template>
  <div class="mb-4">
    <div class="flex items-center justify-end gap-2">
      <span v-if="count" class="text-xs text-muted">{{ count }}</span>
      <button
        type="button"
        class="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-bold transition-colors"
        :class="open || active ? 'border-accent text-accent' : 'border-border text-muted hover:text-canvas-foreground'"
        :aria-expanded="open"
        @click="open = !open"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-3.5 w-3.5" aria-hidden="true">
          <line x1="21" x2="14" y1="4" y2="4" /><line x1="10" x2="3" y1="4" y2="4" />
          <line x1="21" x2="12" y1="12" y2="12" /><line x1="8" x2="3" y1="12" y2="12" />
          <line x1="21" x2="16" y1="20" y2="20" /><line x1="12" x2="3" y1="20" y2="20" />
          <line x1="14" x2="14" y1="2" y2="6" /><line x1="8" x2="8" y1="10" y2="14" /><line x1="16" x2="16" y1="18" y2="22" />
        </svg>
        {{ t('list.controls') }}
        <span v-if="active && !open" class="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden="true" />
      </button>
    </div>

    <transition name="lp-collapse">
      <div
        v-show="open"
        class="mt-2 flex flex-col gap-2 rounded-2xl border border-border bg-surface p-2 sm:flex-row sm:items-center"
      >
        <input
          v-model="search"
          type="search"
          :placeholder="t('list.search')"
          :aria-label="t('list.search')"
          class="min-w-0 flex-1 rounded-full bg-overlay px-3.5 py-1.5 text-xs text-canvas-foreground placeholder:text-muted"
        />
        <div class="flex shrink-0 gap-2">
          <select
            v-model="sort"
            class="rounded-full bg-overlay px-2.5 py-1.5 text-xs font-semibold text-canvas-foreground"
            :aria-label="t('list.sort')"
          >
            <option value="newest">{{ t('list.sortNewest') }}</option>
            <option value="oldest">{{ t('list.sortOldest') }}</option>
            <option value="title">{{ t('list.sortTitle') }}</option>
          </select>
          <select
            v-if="showFilter"
            v-model="filter"
            class="rounded-full bg-overlay px-2.5 py-1.5 text-xs font-semibold text-canvas-foreground"
            :aria-label="t('list.filter')"
          >
            <option value="all">{{ t('list.filterAll') }}</option>
            <option value="insights">{{ t('list.filterInsights') }}</option>
          </select>
          <select
            v-if="shows.length"
            v-model="show"
            class="max-w-[10rem] rounded-full bg-overlay px-2.5 py-1.5 text-xs font-semibold text-canvas-foreground"
            :aria-label="t('list.filterShow')"
          >
            <option value="">{{ t('list.allShows') }}</option>
            <option v-for="s in shows" :key="s.id" :value="s.id">{{ s.label }}</option>
          </select>
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.lp-collapse-enter-active,
.lp-collapse-leave-active {
  transition:
    opacity 0.15s ease,
    transform 0.15s ease;
}
.lp-collapse-enter-from,
.lp-collapse-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
