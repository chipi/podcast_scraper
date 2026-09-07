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
     * ## What this actually does, measured on device
     *
     * A/B on an iPhone 17 Pro simulator: same build, only this value changed, screenshots diffed
     * row by row.
     *
     *   - rows 117-2269 (the header and the whole page body): **byte-identical**
     *   - rows 2270-2621 (the bottom nav and below): different
     *
     * So `contentInset` moves the BOTTOM, not the top. Under `'always'` the WKWebView scroll view
     * insets its content for the home indicator while `BottomNav` is ALSO paying
     * `env(safe-area-inset-bottom)`: the nav is lifted off the edge and a dead black band is left
     * beneath it that the app never paints. Under `'never'` the CSS owns the inset alone and the
     * nav reaches the screen edge, which is what the design intends.
     *
     * ## What it does NOT do - correcting the claim this comment used to make
     *
     * It said this fixed the TOP gap item 1 was opened about, on the reasoning that the notch
     * clearance was being paid twice. That reasoning was never tested on a device - the comment
     * said as much - and the measurement above falsifies it: the top does not move. The brand
     * mark's first painted row is 199px in BOTH builds. Whatever produced the gap in that
     * screenshot, it was not this setting. The change is kept because the bottom behaviour is real
     * and `'never'` is correct there.
     *
     * ## The rest of the original reasoning, which still holds
     *
     * The app pays the inset in CSS: the header is
     * `pt-[max(0.55rem,env(safe-area-inset-top))]`, BottomNav and MiniPlayer use
     * `env(safe-area-inset-bottom)`, and `index.html` sets `viewport-fit=cover` precisely so those
     * `env()` values resolve. Two owners for one inset —
     * roughly 59px + 59px of dead space above the brand bar on a Dynamic Island device.
     *
     * The 12-route sweep behind the original change measured the web layer as identical everywhere
     * (`headerTop=0`, `padTop=8.8px`). That is still true and still shows the page-to-page
     * variation was never CSS. It simply never supported the top-inset conclusion drawn from it.
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
