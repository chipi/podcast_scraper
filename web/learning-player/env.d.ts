/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />
/// <reference types="vite-plugin-pwa/vue" />

interface ImportMetaEnv {
  readonly VITE_API_TARGET?: string
  // Absolute API base for native (Capacitor) builds, e.g. https://api.example.com/api/app.
  // Web builds leave it unset → the app uses the origin-relative '/api/app' (same-origin).
  // Native builds MUST set it: the WebView origin is capacitor://localhost, so relative fails.
  readonly VITE_API_BASE_URL?: string
  // GlitchTip/Sentry DSN for the player's browser error reporting. Build-time
  // (baked into the bundle by Vite); Sentry only initialises when set, so
  // no-DSN builds (dev/CI) stay a true no-op. Passed via the docker build-arg.
  readonly VITE_SENTRY_DSN_PLAYER?: string
  // Dev-tier GlitchTip DSN for on-device (native) builds. The committed default
  // is plain http to the tailnet host `homelab`, which a native WebView blocks
  // (https origin → http = mixed-content / iOS ATS). Set this to the https DSN
  // your `tailscale serve` exposes so dev-tier errors reach GlitchTip on-device.
  readonly VITE_SENTRY_DSN_PLAYER_DEV?: string
  // Umami analytics (cookieless). Both baked at build time; the tracking script
  // is injected only when BOTH are set, so no-arg builds (dev/CI) stay a no-op.
  //   VITE_UMAMI_WEBSITE_ID — the Umami website UUID (public, ships in the bundle)
  //   VITE_UMAMI_SRC        — the tracking-script URL on the analytics ingest edge
  readonly VITE_UMAMI_WEBSITE_ID?: string
  readonly VITE_UMAMI_SRC?: string
  // Dev-tier Umami script URL for on-device builds — same native-https caveat as
  // VITE_SENTRY_DSN_PLAYER_DEV. Set to the https `tailscale serve` script URL.
  readonly VITE_UMAMI_SRC_DEV?: string
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
declare const __APP_VERSION__: string

// Shape of window.__buildInfo — a stable minimal identity surface that
// operators / support can inspect via DevTools console when triaging
// "the PWA isn't updating" reports.
interface Window {
  __buildInfo?: {
    sha: string
    time: string
  }
}
