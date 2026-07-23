/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: 'var(--ps-canvas)',
        'canvas-foreground': 'var(--ps-canvas-foreground)',
        surface: 'var(--ps-surface)',
        'surface-foreground': 'var(--ps-surface-foreground)',
        elevated: 'var(--ps-elevated)',
        'elevated-foreground': 'var(--ps-elevated-foreground)',
        border: 'var(--ps-border)',
        overlay: 'var(--ps-overlay)',
        'overlay-foreground': 'var(--ps-overlay-foreground)',
        muted: 'var(--ps-muted)',
        disabled: 'var(--ps-disabled)',
        link: 'var(--ps-link)',
        primary: 'var(--ps-primary)',
        'primary-foreground': 'var(--ps-primary-foreground)',
        // `accent` is used interchangeably with `primary` across shell
        // surfaces (App.vue chrome, admin views); alias so the class
        // resolves to the same brand color without a broad rename.
        accent: 'var(--ps-primary)',
        'accent-foreground': 'var(--ps-primary-foreground)',
        success: 'var(--ps-success)',
        warning: 'var(--ps-warning)',
        danger: 'var(--ps-danger)',
        // Action-oriented error styling aliases to danger today. Split
        // by adding --ps-destructive later if the operator theme calls
        // for a distinct color.
        destructive: 'var(--ps-danger)',
        gi: 'var(--ps-gi)',
        kg: 'var(--ps-kg)',
        // Knowledge-domain aliases used across ShowRailPanel,
        // NodeDetail, GraphCanvas, admin surfaces, search-result
        // icons (UXS-013 + UXS-015). Backed by --ps-grounded /
        // --ps-topic / --ps-person in tokens.css; those alias to
        // GI/KG values today and can be split when the operator
        // theme calls for distinct hues.
        grounded: 'var(--ps-grounded)',
        topic: 'var(--ps-topic)',
        person: 'var(--ps-person)',
      },
    },
  },
  plugins: [],
}
