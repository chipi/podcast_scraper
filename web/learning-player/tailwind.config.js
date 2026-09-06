/**
 * UXS-011 consumer-app design system (Editorial Bold). Tailwind utilities resolve to the
 * CSS custom properties defined in src/theme/tokens.css — same tokens→tailwind bridge as
 * the operator viewer, but the consumer app owns its OWN tokens (--lp-* vs the viewer's
 * --ps-*). Per-show accent (`--lp-accent`) is set at runtime; components reference it via
 * the `accent` color key, never a hard-coded hue.
 *
 * @type {import('tailwindcss').Config}
 */
/**
 * Bridge one `--lp-*` token into a Tailwind colour that can carry an alpha.
 *
 * Tailwind v3 builds slash-opacity by substituting `<alpha-value>` into the colour, which only
 * works for a channel triple (`rgb(… / <alpha-value>)`). Our tokens hold colour LITERALS, so a
 * bare `var(--lp-x)` produces an invalid value for `bg-x/80` that computes to **transparent** —
 * silently, with no build warning. `bg-accent/70` blanked the activity chart that way, and
 * `bg-canvas/80` was erasing the legibility scrim behind the player's "Insight now" card, leaving
 * white text directly on artwork.
 *
 * Routing every token through `color-mix` fixes the whole class of bug rather than one token at a
 * time. Tailwind substitutes `1` for `<alpha-value>` on non-opacity utilities, so plain `bg-canvas`
 * is unchanged. `color-mix` is already a baseline assumption here (see PlayerControls.vue).
 */
const alphaToken = (name) =>
  `color-mix(in srgb, var(--lp-${name}) calc(<alpha-value> * 100%), transparent)`

/**
 * Scale one spacing ladder by `--lp-density` (#1949).
 *
 * Applied to padding / margin / gap / space ONLY — deliberately not to Tailwind's `spacing` scale
 * as a whole. `spacing` also feeds `width`, `height` and `inset`, so scaling it would shrink every
 * `h-5 w-5` icon and every sized control along with the whitespace. Density is meant to change how
 * much AIR the layout has, not how big its objects are; an icon that shrinks with the padding just
 * makes a tighter direction look zoomed-out rather than denser.
 *
 * `0` and `px` pass through untouched: nothing is gained by scaling a zero, and `px` is the
 * hairline used for borders and 1px offsets (`-mb-px` under the tab underline) — scaling it to
 * 0.8px would blur or drop the rule entirely.
 *
 * Tap targets are unaffected by construction: they are written as arbitrary values
 * (`min-h-[3rem]` on the bottom-nav links), which are literals Tailwind never routes through a
 * theme scale. A dense direction therefore cannot shrink a touch target below its minimum.
 */
const dense = (scale) =>
  Object.fromEntries(
    Object.entries(scale).map(([k, v]) =>
      k === '0' || k === 'px' ? [k, v] : [k, `calc(${v} * var(--lp-density))`],
    ),
  )

/** As `dense`, for any ladder whose zero entry is a duration rather than a length. */
const scaled = (scale, token) =>
  Object.fromEntries(
    Object.entries(scale).map(([k, v]) =>
      k === '0' ? [k, v] : [k, `calc(${v} * var(--lp-${token}))`],
    ),
  )

import defaultTheme from 'tailwindcss/defaultTheme'

export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      // Posture bridge, part two: whitespace follows `--lp-density`. Same trick as the radius
      // ladder — redefine the SCALE so every `p-*`, `gap-*` and `m-*` already in the app responds,
      // rather than touching call sites. At the default `--lp-density: 1` every value computes to
      // its stock rem, so introducing this moves nothing.
      padding: dense(defaultTheme.spacing),
      gap: dense(defaultTheme.spacing),
      space: dense(defaultTheme.spacing),
      margin: { auto: 'auto', ...dense(defaultTheme.spacing) },
      /**
       * Posture bridge, part three: how fast the app moves.
       *
       * 114 of the app's ~124 transition utilities are a bare `transition`, i.e. the DEFAULT
       * duration — so steering that one entry steers nearly all of the app's motion from a single
       * token, with no call site touched.
       *
       * A direction may set `--lp-motion: 0` for an interface that snaps rather than glides.
       * Users who asked for less motion are unaffected either way: the `prefers-reduced-motion`
       * block in style.css already pins `transition-duration` to 0.01ms with `!important`, so it
       * outranks whatever a direction chooses. Motion is a direction's to spend, never theirs to
       * impose.
       */
      transitionDuration: scaled(defaultTheme.transitionDuration, 'motion'),
      colors: {
        canvas: alphaToken('canvas'),
        'canvas-foreground': alphaToken('canvas-foreground'),
        surface: alphaToken('surface'),
        'surface-foreground': alphaToken('surface-foreground'),
        elevated: alphaToken('elevated'),
        overlay: alphaToken('overlay'),
        border: alphaToken('border'),
        muted: alphaToken('muted'),
        disabled: alphaToken('disabled'),
        link: alphaToken('link'),
        // Per-show adaptive accent (contrast-clamped; falls back to brand "Ember"). Keeping
        // `--lp-accent` a plain colour means `tokens.css`, `style.css` and `setShowAccent` are
        // untouched by the alpha bridge above.
        accent: alphaToken('accent'),
        'accent-foreground': alphaToken('accent-foreground'),
        'brand-default': alphaToken('brand-default'),
        success: alphaToken('success'),
        warning: alphaToken('warning'),
        danger: alphaToken('danger'),
        // Knowledge-layer domain tokens (provenance cues — separate from UI intents).
        grounded: alphaToken('grounded'),
        topic: alphaToken('topic'),
        person: alphaToken('person'),
        theme: alphaToken('theme'),
      },
      fontFamily: {
        display: 'var(--lp-font-display)',
        sans: 'var(--lp-font-ui)',
        mono: 'var(--lp-font-mono)',
      },
      /**
       * Posture bridge (#1949) — the corner scale is steered by one token.
       *
       * A visual direction is meant to be able to change how the app CARRIES itself, not just
       * how it is painted, and to do that with no component edits. So rather than asking 200+
       * `rounded-*` call sites to reference a variable, the SCALE itself is redefined in terms of
       * `--lp-radius`: every class already written in the app becomes direction-aware for free.
       *
       * The multipliers reproduce Tailwind's stock ladder exactly at the default
       * `--lp-radius: 0.25rem` (sm .125 / DEFAULT .25 / md .375 / lg .5 / xl .75 / 2xl 1 / 3xl
       * 1.5rem), so adding this indirection is a no-op for the shipping look — verified by
       * re-shooting every surface and diffing against the pre-change captures.
       *
       * `full` stays 9999px on purpose: see the note in tokens.css.
       */
      borderRadius: {
        none: '0px',
        sm: 'calc(var(--lp-radius) * 0.5)',
        DEFAULT: 'var(--lp-radius)',
        md: 'calc(var(--lp-radius) * 1.5)',
        lg: 'calc(var(--lp-radius) * 2)',
        xl: 'calc(var(--lp-radius) * 3)',
        '2xl': 'calc(var(--lp-radius) * 4)',
        '3xl': 'calc(var(--lp-radius) * 6)',
        full: '9999px',
      },
    },
  },
  plugins: [],
}
