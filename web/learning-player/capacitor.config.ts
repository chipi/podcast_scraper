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
    /**
     * NEVER, not 'always' — the safe-area inset has exactly one owner, and it is CSS (#2004 item 1).
     *
     * `contentInset: 'always'` sets the WKWebView scroll view to inset its content for the safe area
     * natively. The app ALSO pays the inset in CSS: the header is
     * `pt-[max(0.55rem,env(safe-area-inset-top))]`, BottomNav and MiniPlayer use
     * `env(safe-area-inset-bottom)`, and `index.html` sets `viewport-fit=cover` precisely so those
     * `env()` values resolve. Both layers were applying it, so the notch clearance was paid TWICE —
     * roughly 59px + 59px of dead space above the brand bar on a Dynamic Island device.
     *
     * It read as "some pages have a gap, others don't" only because the header is not sticky: on a
     * scrolled page the doubled inset has already scrolled away. Measured across 12 routes, the web
     * layer is byte-identical everywhere — `headerTop=0`, `padTop=8.8px` — so the page-to-page
     * variation was never CSS.
     *
     * CSS owns it because `env()` is the only mechanism the PWA build also has; letting native own it
     * would leave the browser build with no notch handling at all.
     */
    contentInset: 'never',
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
    // Branded launch splash (the cinematic desk scene, generated into ios Splash.imageset). Held on
    // the #0e0d10 canvas so there's no white flash, and hidden explicitly from JS (App.vue) the moment
    // the SPA has mounted — launchAutoHide:false means we control the hand-off, not an arbitrary timer.
    SplashScreen: {
      launchAutoHide: false,
      backgroundColor: CANVAS,
      showSpinner: false,
      splashFullScreen: true,
      splashImmersive: true,
    },
  },
}

export default config
