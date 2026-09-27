import { onBeforeUnmount, ref, watch, type Ref } from 'vue'

/**
 * The prose-window measurement — ONCE, for every surface that clips text under a "Read more".
 *
 * Four components had hand-copied versions of this: `EpisodeCard`, `ShowRow`, `PersonCardContent`
 * and `PodcastView`. They were written from each other, so they shared the bugs and then drifted:
 * two observed the element via `watch` and two via `onMounted` (which misses a window that appears
 * in a second render), and the threshold was `> 1` in three of them and `> 2` in the fourth.
 *
 * The drift is the reason this exists rather than four more patches. When the operator reported
 * half-cut lines on the queue (2026-09-23) the same fault was live on Browse › Episodes, Browse ›
 * Shows and every show page — fixing one copy would have left three, which is how it got here.
 *
 * ## What it does
 *
 * The WINDOW (`.lp-media-clip`) takes its height from the neighbouring artwork column and hides the
 * overflow; the prose is absolutely positioned inside it, so the prose can never grow the row.
 *
 * 1. **Is it actually cut off?** Measures the PROSE against the WINDOW. Measuring the window against
 *    itself was the original bug — it stretches to fit, so the two always matched and no toggle ever
 *    appeared. Defaults to `true`: before layout both heights read 0, and treating that as "it fits"
 *    HIDES a "Read more" the text needs, which is the worse of the two failures.
 * 2. **Cut it on a LINE.** The window's height has no reason to be a whole multiple of the prose's
 *    line-height, so `overflow: hidden` sliced the last line through the middle of the glyphs. The
 *    prose is clamped to the number of lines that FIT, so the text ends on a real line with an
 *    ellipsis — a deliberate cut rather than a rendering fault.
 *
 * The clamp is applied to the PROSE, which is absolutely positioned, so it changes what is drawn and
 * never any height — it cannot feed back into the ResizeObserver that triggered it.
 *
 * The observer watches the element REF, not `onMounted`: a window behind `v-if="description"` does
 * not exist on the first render when the data arrives in two phases, and a mount-time guard then
 * skipped observer creation for the life of that instance. `immediate: true` makes this a strict
 * superset of `onMounted`; `flush: 'post'` guarantees the DOM exists; ResizeObserver's initial
 * callback delivers the first size and fires again across `display: none` → visible.
 *
 * @param el       the WINDOW element (the `.lp-media-clip`), as a template ref
 * @param expanded whether the caller is currently showing the full text
 */
export function useClampedProse(
  el: Ref<HTMLElement | null>,
  expanded: Ref<boolean>,
): { clipped: Ref<boolean>; measure: () => void } {
  const clipped = ref(true)

  function measure(): void {
    const win = el.value
    if (!win) return
    const prose = win.firstElementChild as HTMLElement | null
    if (!prose) return

    if (expanded.value) {
      // Expanded: the window no longer constrains anything, so the clamp must come OFF — left on,
      // "Read more" would open onto text still cut at the line it was cut at before.
      prose.style.webkitLineClamp = ''
      return
    }

    // Measure UNCLAMPED: a clamp left from the previous pass caps `scrollHeight`, so the element
    // would report "it fits" about text it had itself cut short.
    prose.style.webkitLineClamp = ''
    if (win.clientHeight === 0) return // not laid out yet — keep the safe default
    clipped.value = prose.scrollHeight - win.clientHeight > 1

    const lineHeight = parseFloat(getComputedStyle(prose).lineHeight)
    if (!Number.isFinite(lineHeight) || lineHeight <= 0) return
    prose.style.webkitLineClamp = clipped.value
      ? String(Math.max(1, Math.floor(win.clientHeight / lineHeight)))
      : ''
  }

  // `onBeforeUnmount` stays at setup top level: Vue does not set `currentInstance` for watcher
  // callbacks, so registering it inside would warn and not bind.
  let ro: ResizeObserver | null = null
  watch(
    el,
    (next) => {
      ro?.disconnect()
      ro = null
      if (!next || typeof ResizeObserver === 'undefined') return
      ro = new ResizeObserver(() => measure())
      ro.observe(next)
    },
    { flush: 'post', immediate: true },
  )
  onBeforeUnmount(() => ro?.disconnect())

  // Toggling does not necessarily resize the window (the row keeps the artwork's height), so the
  // observer may never fire — re-measure explicitly rather than relying on it.
  watch(expanded, () => measure(), { flush: 'post' })

  return { clipped, measure }
}
