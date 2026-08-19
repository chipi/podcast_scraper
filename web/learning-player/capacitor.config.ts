import type { CapacitorConfig } from '@capacitor/cli'

// Local-dev live-reload origin: set CAP_DEV_SERVER to your vite dev URL to load the whole app from
// it for on-device debugging. MUST be unset in every release build (guarded by NODE_ENV below).
const devServer = process.env.CAP_DEV_SERVER

// #0e0d10 is the app's real dark canvas (--lp-canvas, theme/tokens.css) — the native background +
// splash must match it exactly so the shell never shows a white flash on launch or between views.
const CANVAS = '#0e0d10'

const config: CapacitorConfig = {
  appId: 'app.closelistening.player',
  appName: 'Learning Player',
  webDir: 'dist',
  server: {
    androidScheme: 'https', // stable WebView origin for history routing
    ...(devServer && process.env.NODE_ENV !== 'production' ? { url: devServer, cleartext: true } : {}),
  },
  ios: {
    contentInset: 'always',
    backgroundColor: CANVAS,
    scrollEnabled: true,
    limitsNavigationsToAppBoundDomains: true, // App Store req; external links go via @capacitor/browser (#1310)
    allowsLinkPreview: false,
  },
  android: { backgroundColor: CANVAS, captureInput: true },
  // Route the WebView's fetch/XHR through the native HTTP stack. The API lives on a DIFFERENT
  // origin than the shell (capacitor://localhost → https://closelistening.app), so browser fetch
  // hits CORS: the cross-origin reads need Access-Control-* the coming-soon edge never sends, and an
  // `Authorization` header would trigger a preflight OPTIONS that carries no cookie/creds and so lands
  // on the coming-soon gate. Native requests skip CORS + preflight entirely, and let the dev-tier
  // Basic-auth header (services/tier.ts :: resolveGateAuthHeader) ride straight to the edge's @authed
  // fallback. Applies to both tiers (dev = laptop http, prod = gated https).
  plugins: {
    CapacitorHttp: { enabled: true },
  },
}

export default config
