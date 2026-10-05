import { computed, ref, type Ref } from "vue"

/**
 * Show a long list a page at a time: the first `step` items, then `step` more per "Show more"
 * (operator 2026-10-05: "Who agrees" and the perspectives lists, in chunks of 5). Once everything
 * is showing, `reset()` folds it back to the first page.
 */
export function usePaged<T>(items: Ref<readonly T[]>, step = 5) {
  const shown = ref(step)
  const visible = computed(() => items.value.slice(0, shown.value))
  const hidden = computed(() => Math.max(0, items.value.length - shown.value))
  /** How many the next "Show more" reveals: a full page, or what is left. */
  const nextCount = computed(() => Math.min(step, hidden.value))
  /** Paged past the first page — offer "Show less" once nothing is hidden. */
  const canFold = computed(() => hidden.value === 0 && items.value.length > step)
  function more(): void {
    shown.value += step
  }
  function reset(): void {
    shown.value = step
  }
  return { visible, hidden, nextCount, canFold, more, reset }
}
