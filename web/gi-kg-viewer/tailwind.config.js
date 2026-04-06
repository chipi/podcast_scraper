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
        success: 'var(--ps-success)',
        warning: 'var(--ps-warning)',
        danger: 'var(--ps-danger)',
        gi: 'var(--ps-gi)',
        kg: 'var(--ps-kg)',
      },
    },
  },
  plugins: [],
}
