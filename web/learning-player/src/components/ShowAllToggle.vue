<script setup lang="ts">
/**
 * The reveal control under a capped Library section (#2042 follow-up). Kept as one component so
 * every section reads identically and the label stays translated in one place.
 *
 * DELIBERATELY not the full-width bordered bar that `CatalogView` / `ShowBrowseView` /
 * `PodcastView` / `EntityEpisodeList` use (operator 2026-09-19). Those sit at the END of a list and
 * mean "this page continues"; this one sits INSIDE a section of a multi-section hub and means "this
 * section has more". A bar here would draw a hard rule between sections that are meant to read as
 * one scannable page, and Library → Saved stacks four of them. Same job, different scope — so if
 * you are here to unify the two, this is the note saying the difference was chosen.
 *
 * Two labels, because the sections now page two different ways (operator 2026-09-18):
 *
 * - **expand-all** (`remaining` unset): "Show all (N)" / "Show less" — the whole list in one press.
 * - **incremental** (`remaining` passed): "Show more (N)" while items are still hidden, then
 *   "Show less" once everything is out. The count is what is STILL HIDDEN rather than the total,
 *   because that is the question the control answers — whether pressing again is worth it.
 */
import { useI18n } from 'vue-i18n'

const props = defineProps<{
  expanded: boolean
  count: number
  /** Items still hidden. Passing this is what puts the control in incremental mode. */
  remaining?: number
}>()
defineEmits<{ toggle: [] }>()
const { t } = useI18n()

function label(): string {
  if (props.remaining == null) {
    return props.expanded ? t('library.showLess') : t('library.showAll', { count: props.count })
  }
  return props.remaining > 0
    ? t('library.showMore', { count: props.remaining })
    : t('library.showLess')
}
</script>

<template>
  <button
    type="button"
    class="mt-2 text-xs font-bold text-accent transition hover:opacity-80"
    data-testid="show-all-toggle"
    :aria-expanded="expanded"
    @click="$emit('toggle')"
  >
    {{ label() }}
  </button>
</template>
