# Design / product gaps — 2026-08-08

Running list of design + product gaps to close. Append new gaps as `## GAP-N`.

---

## GAP-1 — Universal brand icon (player + MCP connector, one asset everywhere)

**Need.** A single, recognizable "Close Listening" icon that works **across every surface** —
the player app *and* the remote MCP connector — so the brand is consistent wherever it appears.

### Where it has to render (all from one master)
| Surface | Sizes / form | Notes |
|---|---|---|
| Player PWA install icon | 192², 512², **maskable** (≥20% safe padding) | `web/learning-player/public` manifest icons |
| Favicon (browser tab, claude.ai connector list often uses the domain favicon) | 16², 32², 48² (+ SVG) | must stay legible at 16px |
| iOS / Android app icon (Capacitor mobile shell, #1310) | 1024² master → platform sets | square, no transparency for iOS |
| **MCP connector logo in claude.ai** | small square, light + dark bg | how claude.ai sources it is an OPEN QUESTION (below) |
| Social / OG card (optional) | 1200×630 lockup | icon + wordmark |

### Requirements
- **One SVG master** → deterministic export to all raster sizes (script it, don't hand-cut).
- **Square, centered, safe margins** so the maskable/rounded variants don't clip.
- **Legible at 16px** (favicon + connector list) AND clean at 512px+ (app icon).
- **Light- and dark-background safe** (connector lists + tabs vary); consider a mono/flat variant
  for tiny sizes and a full-color variant for large.
- **No text inside the glyph** (text is unreadable at 16px) — wordmark is a separate lockup.

### Open questions (need answers before finalizing)
1. **How does claude.ai pick the connector icon?** Domain favicon of `mcp.closelistening.app`,
   an MCP server-metadata field, or the `/.well-known` doc? → determines what we must serve and
   where. (Action: confirm from the MCP/claude.ai connector docs.)
2. Concept direction — the glyph motif. Options to explore (pick one):
   - listening/audio: soundwave, headphones, an ear
   - knowledge/graph: nodes+edges (ties to the KG/GI product), a "listening → understanding" mark
   - a hybrid (waveform that resolves into a graph)
3. Palette + does it need to match the existing player theme tokens?

### Acceptance
- `assets/brand/icon.svg` master committed + an export script producing every size above.
- Player manifest + favicons wired to the new icon; mobile app icon set regenerated.
- MCP connector shows the icon in claude.ai (once we know how it's sourced).
- Renders cleanly at 16px and 512px on both light and dark backgrounds.

---

## GAP-2 — iOS app splash / launch screen (graphical, on-brand)

**Need.** A **graphical splash screen** for the iOS app (the Capacitor mobile shell, #1310),
shown on launch — something visually rich and **in tune with the player's design tokens**
(palette / theme), and coherent with the GAP-1 icon so launch → app feels like one brand.

### Requirements
- **On-brand + graphical**, driven by the **existing player design tokens** (colors/theme in
  `web/learning-player`), not a plain logo-on-white. Should feel like a designed screen.
- **Capacitor convention:** a single **2732×2732** master with all key content in the centered
  **safe zone** (the tooling center-crops it to every device size / aspect ratio), or a native
  launch storyboard. Wire via `@capacitor/splash-screen` + `capacitor.config`.
- **Light + dark variants** (iOS honours a dark launch screen).
- **Reuse the GAP-1 icon/glyph** as the focal element for brand continuity.
- Keep it lightweight (launch assets ship in the bundle).

### Open questions
1. Static image vs. a subtle branded gradient/pattern behind the icon? (No animation at true
   launch — the OS shows a static launch image; any motion is a post-launch in-app splash.)
2. Android splash too, or iOS-only for now? (Capacitor can do both from the same master —
   cheap to include; confirm scope.)

### Acceptance
- Splash master (2732², light + dark) committed under `assets/brand/`; export/config wired via
  Capacitor.
- Renders correctly on iOS launch across device sizes (centered safe zone, no clipping) in both
  light and dark mode.
- Visually consistent with the GAP-1 icon + the player theme tokens.

---

## GAP-3 — Build the Android app + validate on iOS **and** Android

**Need.** The mobile shell (Capacitor, #1310) has been built for **iOS**; **Android is missing**.
Add the Android target, build it, and validate the app on **both** platforms.

### Requirements
- Capacitor **Android** target: `npx cap add android`, gradle build → debug APK (dev) and a
  signed release AAB (store).
- **Both platforms tested:** app launches, the **Google OAuth deep-link** works (native callback
  uses the custom scheme `closelistening://auth` — `NATIVE_AUTH_SCHEME` in `app_auth.py`), and
  core flows (feed, player, Your Week) work on iOS + Android.
- Android must register the **`closelistening://` intent-filter** so the OAuth deep-link callback
  returns into the app (parity with the iOS custom-scheme handling).
- Reuse GAP-1 (app icon) + GAP-2 (splash) once those land.

### Dependency / handover
- Needs the **build runbook/handover**. If there is no existing iOS-build handover to adapt,
  the operator will supply an **Android-build handover from another app/worktree**. → this GAP is
  effectively **blocked on that handover doc** landing here.

### Open questions
- Android **signing**: keystore creation + where it's stored (never in git).
- min/target SDK; Android Studio/gradle locally vs. CI build.
- Emulator vs. real-device test matrix.
- Note (ops): a prior **iOS build hung ~5.5h** — set a realistic kill-deadline + live progress
  watch on the Android/gradle build too; don't let it run unbounded.

### Acceptance
- Android app builds (APK debug + signed AAB), launches, Google OAuth deep-link works, core flows
  validated; iOS re-validated green; the build steps documented as a runbook in `docs/`.
