# App links — what has to happen before the next app builds (2026-10-05)

**Goal.** One `https://closelistening.app/…` link for everything — Copy link, every email, every
push — that opens like YouTube or Spotify: in the app when it is installed, in the browser when it
is not, and through sign-in and back to the same page when the person is signed out.

**Why the console steps exist.** A phone only hands a web link to an app when BOTH sides agree:
the app says "I open closelistening.app links" (built into the app), and the website confirms "that
app is mine" with a file at `/.well-known/`. The website file names the app precisely — on iOS by
Team ID + bundle id, on Android by package name + the SHA-256 fingerprint of the key Google Play
signs it with. Without the check, any app could claim our links (including sign-in links).

## Done in code (local commits, not pushed)

| What | Where |
|---|---|
| Copy link / Copy text actually copy; links from the apps use `https://closelistening.app` (they were `capacitor://localhost/…` / `https://localhost/…`) | podcast_scraper `bc57ccacf` |
| The app opens storyline + theme links, accepts graph ids, keeps `?revisit=` | `bc57ccacf` |
| iOS `applinks:closelistening.app`, Android verified https filter — content paths only (`/episode /podcast /topic /person /storyline /theme`), so the sign-in email link stays in the browser | `cbaec6b4f` |
| `/.well-known/apple-app-site-association` + `assetlinks.json`, served as JSON by the player nginx (they returned index.html) | `cbaec6b4f` |
| Digest "trending" topic links carried a stripped id (`/topic/ai`) and opened an EMPTY page; now the full id | `e8fcc7380` |
| Email chips (topic/person) + daily-recap storyline links: same stripped-id bug, fixed; push fallback `/revisit` (not a route) → `/library?tab=revisit` | homelab `802351d` on `fix/email-links-full-graph-ids` |

## YOUR STEPS (browser) — in this order

### 1. Google Play — send me the fingerprint

1. <https://play.google.com/console> → **Close Listening** → **Test and release** → **App integrity**
   (older menus: **Setup → App signing**).
2. Under **App signing key certificate**, copy the **SHA-256 certificate fingerprint**
   (`AB:CD:…`, 32 pairs).
3. Also copy the **Upload key certificate** SHA-256 — it lets builds signed with the upload key
   (internal sideloads) verify too.
4. Send both. I put them in `web/learning-player/public/.well-known/assetlinks.json` — it holds a
   placeholder (`PLAY_APP_SIGNING_SHA256_FROM_PLAY_CONSOLE`) until then. The fingerprint is
   public, not a secret.

### 2. Apple Developer — enable Associated Domains

1. <https://developer.apple.com/account> → **Certificates, Identifiers & Profiles** →
   **Identifiers**.
2. Open **app.closelistening.player** → **Capabilities** → tick **Associated Domains** → **Save**
   (confirm the dialog about invalidating profiles).
3. Do the same for **app.closelistening.player.dev** (internal/device builds). The dev build does
   NOT open links — the website file names only the shipped app (operator 2026-10-05: links open
   in the production app only) — but it carries the same entitlements file, so without the
   capability a dev DEVICE build fails to sign.
4. **Profiles** → **Close Listening Player AppStore** → it now shows *Invalid* → **Edit** →
   **Save/Generate** → **Download**, and double-click it (or let me regenerate it via the App Store
   Connect API key). Release builds sign MANUALLY with this profile; without regenerating it the
   TestFlight build fails with a missing-entitlement signing error. Device dev builds sign
   automatically and pick the change up on their own.

### 3. Resend — make sure click tracking is OFF

<https://resend.com/domains> → the sending domain → **Configuration** → **Click tracking** must be
**off**. With it on, every email link is rewritten through Resend's tracking host, and a phone
never hands such a link to the app — the app-link work would not apply to email at all. (I could
not check: the local `infra/delivery/.env` has no key, and I did not read secrets off the hosts.)

### 4. Approve the deploy

After step 1, I fill the fingerprint, rebase, run the gates, show you the diff, and ask to push.
The **player deploy** puts the two `/.well-known` files live; the **delivery worker** deploy
ships the email templates. Each is a prod action — you trigger or approve each one.

### 5. Then: build the apps

iOS and Android builds AFTER the deploy (the phones fetch the `/.well-known` files when the app
is installed).

## How we verify (after the deploy)

- `curl -sI https://closelistening.app/.well-known/apple-app-site-association` → `200`,
  `content-type: application/json`; same for `assetlinks.json`.
- Apple's CDN copy: `https://app-site-association.cdn-apple.com/a/v1/closelistening.app` (can lag
  the site by up to a day).
- Google's check: `https://digitalassetlinks.googleapis.com/v1/statements:list?source.web.site=https://closelistening.app&relation=delegate_permission/common.handle_all_urls`.
- Devices, signed in and signed out, for an episode, a moment (`?t=`), a person, a topic, a
  storyline: link with the app installed → opens in the app on that page; without the app → the
  browser; signed out → sign-in → the same page. Plus one link from each email type.
