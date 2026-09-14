<script setup lang="ts">
/**
 * Compact toolbar control (operator 2026-09-14) — a small trigger that opens a vertical menu of
 * options, replacing the native `<select>` on the list toolbar.
 *
 * Why not a `<select>`: a native select reserves width for its WIDEST option, so "All" rendered as
 * a massive pill and the sort select ate the row. This shows only the ACTIVE state (an icon, or the
 * current label on the `pill` variant) and reveals the choices on tap — so sort + view collapse to
 * little circles and the search gets the width back.
 *
 * Presentational: `v-model` is the selected value; the caller owns the options + labels.
 */
import { onBeforeUnmount, ref, watch } from "vue"

const model = defineModel<string>({ default: "" })
const props = withDefaults(
  defineProps<{
    options: { value: string; label: string }[]
    menuLabel: string
    /** `icon` = a 36px circle (sort/view); `pill` = a compact chip showing the current label (filter). */
    variant?: "icon" | "pill"
    testid?: string
    /** Which edge the popover aligns to — `right` for controls near the row's end. */
    align?: "left" | "right"
  }>(),
  { variant: "icon", align: "left" }
)

const open = ref(false)
const root = ref<HTMLElement | null>(null)

function currentLabel(): string {
  return props.options.find((o) => o.value === model.value)?.label ?? ""
}
function pick(value: string): void {
  model.value = value
  open.value = false
}
// Dismiss on an outside press or Escape (a menu you can't click away from is a trap). `pointerdown`,
// not `click`, so a sibling control that stops click propagation can't wedge it open.
function onDocPointerDown(e: PointerEvent): void {
  if (root.value && !root.value.contains(e.target as Node)) open.value = false
}
function onKey(e: KeyboardEvent): void {
  if (e.key === "Escape") open.value = false
}
watch(open, (isOpen) => {
  if (isOpen) {
    document.addEventListener("pointerdown", onDocPointerDown)
    document.addEventListener("keydown", onKey)
  } else {
    document.removeEventListener("pointerdown", onDocPointerDown)
    document.removeEventListener("keydown", onKey)
  }
})
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onDocPointerDown)
  document.removeEventListener("keydown", onKey)
})
</script>

<template>
  <div ref="root" class="relative shrink-0">
    <button
      type="button"
      :aria-label="menuLabel"
      :aria-expanded="open"
      aria-haspopup="menu"
      :data-testid="testid"
      class="lp-tap flex h-9 items-center justify-center gap-1 rounded-full border transition"
      :class="[
        variant === 'pill' ? 'px-3' : 'w-9',
        open
          ? 'border-accent text-accent'
          : 'border-border text-muted hover:text-canvas-foreground',
      ]"
      @click.stop="open = !open"
    >
      <slot name="icon" />
      <span v-if="variant === 'pill'" class="max-w-[6rem] truncate text-xs font-semibold">{{
        currentLabel()
      }}</span>
    </button>
    <div
      v-if="open"
      class="absolute z-40 mt-1 min-w-[9rem] max-w-[calc(100vw-2rem)] rounded-xl border border-border bg-surface p-1 shadow-lg"
      :class="align === 'right' ? 'right-0' : 'left-0'"
      role="menu"
      :aria-label="menuLabel"
    >
      <button
        v-for="o in options"
        :key="o.value"
        type="button"
        role="menuitemradio"
        :aria-checked="o.value === model"
        :data-testid="testid ? `${testid}-opt-${o.value}` : undefined"
        class="flex w-full items-center justify-between gap-3 rounded-lg px-2 py-1.5 text-left text-sm transition hover:bg-overlay"
        :class="o.value === model ? 'font-semibold text-canvas-foreground' : 'text-muted'"
        @click="pick(o.value)"
      >
        <span class="min-w-0 truncate">{{ o.label }}</span>
        <span v-if="o.value === model" aria-hidden="true" class="shrink-0 text-accent">✓</span>
      </button>
    </div>
  </div>
</template>
