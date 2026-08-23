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

export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
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
    },
  },
  plugins: [],
}
