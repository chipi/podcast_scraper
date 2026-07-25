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
}

export default config
