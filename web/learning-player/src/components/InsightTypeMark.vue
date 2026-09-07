<script setup lang="ts">
/**
 * The mark that tells one insight type from another (#2004 item 8, second pass).
 *
 * ## What was wrong with the first attempt
 *
 * The complaint was that every insight in the panel looked identical. The first answer was a text
 * glyph — `◆ ○ → ?` — set at kicker size, and it did not land:
 *
 * * it sat immediately after a green dot that rendered on EVERY grounded row, so the row still
 *   opened with the same mark every time and the one differentiating character competed with it
 * * a text glyph inherits the kicker's ~10px muted styling, so the shapes were small and washed out
 * * `→` and `?` are punctuation, not marks: they read as typography, not as a visual identity
 *
 * The dot is gone (it duplicated the timestamp on the same row). These are the replacement.
 *
 * ## Why SVG shapes with their own colour
 *
 * Size and weight are exact and identical across the four, which a text character cannot promise —
 * `◆` and `?` have completely different optical weights in any font. Colour is back deliberately:
 * shape alone at 10px is a weak signal, and Marko asked for the presence the old dot had. It is
 * carried by the MARK only, never by the label, so `.lp-kicker` stays mono + muted (#2013) and the
 * accent stays reserved for things a finger can act on.
 *
 * The shapes are a family — diamond, ring, triangle, square — chosen to be separable at a glance
 * and in greyscale, since colour is a second channel here and never the only one. The type WORD is
 * always beside them, so the mark's job is fast differentiation, not self-explanation, which is why
 * an arbitrary-but-distinct shape (square for `question`) is fine.
 *
 * `aria-hidden`: the word next to it already says the same thing, and a screen reader should not
 * hear "diamond claim".
 */
const props = defineProps<{ type: string }>()

/** Same four keys as `INSIGHT_TYPE_GLYPHS` replaced — the closed vocabulary from `gi/pipeline.py`. */
const MARKS: Record<string, { path: string; fill: boolean; token: string }> = {
  // Diamond, filled — an assertion, the most "solid" of the four.
  claim: { path: 'M6 1.2 10.8 6 6 10.8 1.2 6Z', fill: true, token: 'claim' },
  // Ring — noticing something, lighter than asserting it.
  observation: { path: 'M6 1.6A4.4 4.4 0 1 1 6 10.4 4.4 4.4 0 1 1 6 1.6Z', fill: false, token: 'observation' },
  // Triangle pointing right — a nudge toward doing something.
  recommendation: { path: 'M2.4 1.4 10.6 6 2.4 10.6Z', fill: true, token: 'recommendation' },
  // Square — arbitrary, and deliberately so: it only has to be unmistakably NOT the other three.
  question: { path: 'M1.8 1.8h8.4v8.4H1.8Z', fill: false, token: 'question' },
}

/**
 * A type outside the closed vocabulary still gets a mark — a small neutral dot in the kicker's own
 * muted colour.
 *
 * The server's vocabulary is closed, but a corpus built by an older pipeline (or a future value)
 * can carry something else. Rendering nothing would leave that row without a mark column while
 * every neighbour has one, so the list stops lining up for the one row that is already unusual. A
 * neutral dot says "a type we have no identity for" without inventing one.
 */
const FALLBACK = { path: 'M6 3.6A2.4 2.4 0 1 1 6 8.4 2.4 2.4 0 1 1 6 3.6Z', fill: true, token: '' }

const mark = (): { path: string; fill: boolean; token: string } => MARKS[props.type] ?? FALLBACK
</script>

<template>
  <svg
    viewBox="0 0 12 12"
    class="h-[0.7rem] w-[0.7rem] shrink-0"
    :style="{ color: mark().token ? `var(--lp-insight-${mark().token})` : 'var(--lp-muted)' }"
    aria-hidden="true"
  >
    <path
      :d="mark().path"
      :fill="mark().fill ? 'currentColor' : 'none'"
      :stroke="mark().fill ? 'none' : 'currentColor'"
      stroke-width="1.6"
      stroke-linejoin="round"
    />
  </svg>
</template>
