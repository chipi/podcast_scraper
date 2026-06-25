/**
 * UXS-011 consumer-app design system (Editorial Bold). Tailwind utilities resolve to the
 * CSS custom properties defined in src/theme/tokens.css — same tokens→tailwind bridge as
 * the operator viewer, but the consumer app owns its OWN tokens (--lp-* vs the viewer's
 * --ps-*). Per-show accent (`--lp-accent`) is set at runtime; components reference it via
 * the `accent` color key, never a hard-coded hue.
 *
 * @type {import('tailwindcss').Config}
 */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: 'var(--lp-canvas)',
        'canvas-foreground': 'var(--lp-canvas-foreground)',
        surface: 'var(--lp-surface)',
        'surface-foreground': 'var(--lp-surface-foreground)',
        elevated: 'var(--lp-elevated)',
        overlay: 'var(--lp-overlay)',
        border: 'var(--lp-border)',
        muted: 'var(--lp-muted)',
        disabled: 'var(--lp-disabled)',
        link: 'var(--lp-link)',
        // Per-show adaptive accent (contrast-clamped; falls back to brand "Ember").
        accent: 'var(--lp-accent)',
        'accent-foreground': 'var(--lp-accent-foreground)',
        'brand-default': 'var(--lp-brand-default)',
        success: 'var(--lp-success)',
        warning: 'var(--lp-warning)',
        danger: 'var(--lp-danger)',
        // Knowledge-layer domain tokens (provenance cues — separate from UI intents).
        grounded: 'var(--lp-grounded)',
        topic: 'var(--lp-topic)',
        person: 'var(--lp-person)',
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
