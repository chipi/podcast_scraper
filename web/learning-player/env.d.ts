/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />
/// <reference types="vite-plugin-pwa/vue" />

interface ImportMetaEnv {
  readonly VITE_API_TARGET?: string
  // GlitchTip/Sentry DSN for the player's browser error reporting. Build-time
  // (baked into the bundle by Vite); Sentry only initialises when set, so
  // no-DSN builds (dev/CI) stay a true no-op. Passed via the docker build-arg.
  readonly VITE_SENTRY_DSN_PLAYER?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

// Build-time identity — injected by vite `define:` block in vite.config.ts.
// Consumed by src/main.ts to expose window.__buildInfo for update-path debugging.
// NOTE: env.d.ts is intentionally NOT a module (no `import`/`export`), so
// these `declare const` bindings + the Window augmentation live in the
// global scope by default. Do not add `export {}` — it would flip this
// file to module mode and break the globals silently.
declare const __BUILD_SHA__: string
declare const __BUILD_TIME__: string

// Shape of window.__buildInfo — a stable minimal identity surface that
// operators / support can inspect via DevTools console when triaging
// "the PWA isn't updating" reports.
interface Window {
  __buildInfo?: {
    sha: string
    time: string
  }
}
