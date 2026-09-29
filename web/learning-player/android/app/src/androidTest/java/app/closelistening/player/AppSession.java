package app.closelistening.player;

import android.content.Context;
import android.graphics.Rect;
import android.content.Intent;
import android.net.Uri;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;
import androidx.test.uiautomator.Until;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Session helpers for the Android device tier (#2139) — the sibling of `AppSession.swift`.
 *
 * Both tiers need the same question answered — "is this app signed in, and as whom?" — and
 * answering it too early is what made the iOS harness non-deterministic for weeks. Boot PAINTS the
 * last known identity from the device first, so a launch looks signed in immediately, and only
 * then revalidates against the api. A token minted by a previous fixture-api container is refused
 * by the new one, so a session can sit on screen for several seconds and then vanish. Every check
 * here therefore reads the state AFTER that window, not during it.
 */
final class AppSession {

    private AppSession() {}

    /**
     * A session exists, read POSITIVELY from the masthead. No navigation, no sleeps.
     *
     * This used to be "the 'Sign in' link is absent", and that is the conflation that cost most of
     * 2026-09-28: `/login` carries its OWN "Sign in" SUBMIT button in the page body, so on that one
     * route a signed-IN app still shows a control by that name. The absence test therefore reports
     * SIGNED OUT for an app that is signed in, on exactly the page you are on right after signing
     * out — which is where every account switch passes through.
     *
     * The masthead itself was never ambiguous. `App.vue` gates Queue (:595), the notifications bell
     * (:622) and the profile link (:627) on `auth.hasSession`, and the "Sign in" link (:664) on its
     * negation. Reading a control that only exists WHEN SIGNED IN cannot be confused by a route
     * that happens to render a similarly-named button.
     *
     * The bell is the signal rather than the profile link because the profile link is labelled with
     * the ACCOUNT NAME (`auth.user?.name || t('profile.title')`), which a generic caller does not
     * know — that is the whole reason this method exists separately from {@link #isSignedIn(String)}.
     * `contains` so "Notifications (3 unread)" matches too.
     */
    static boolean hasAnySession() {
        return Journey.find(Arrays.asList("Notifications"), true, 8_000) != null;
    }

    /**
     * Wait for the masthead to hold one answer THROUGH the revalidation window, then report it.
     *
     * Boot paints the last known identity from the device and only then revalidates against the
     * api, so a single read can be confidently wrong: the session sits on screen and vanishes
     * seconds later when the token is refused. That is what the `sleep(6_000)`s in this file were
     * buying, and the first version of this method did not replace it properly.
     *
     * IT REQUIRED TWO CONSECUTIVE AGREEING READS, which proves STABILITY but not DURATION — two
     * reads a second apart both land inside the revalidation window. The old path was slow enough
     * (a Profile trip plus a 6s sleep, ~40s) to outlast it by accident, so removing the slack
     * exposed a race the slack had been hiding.
     *
     * Measured 2026-09-28, `AppJourneyTests.test01ProfileTabs` in a full tier run:
     *
     *     =====CALLBACK signed in as appjourneytests in 1906ms, no Custom Tab=====
     *     =====OFFLINE_SET round 1 observed=false wanted=false=====   (x3)
     *     AssertionError: profile tab 'Topics' not tappable. On screen: … Create your free
     *         account … Sign in …
     *
     * Sign-in succeeded, `startClean` believed it, and the app signed itself out a few steps later.
     *
     * So the answer must now hold CONTINUOUSLY for `STABLE_FOR_MS` — the same guarantee the sleep
     * gave, without the Profile navigation that made up most of the old cost. Roughly 6s instead of
     * ~40s: the speed came from deleting the navigation, not from shortening the wait, and only the
     * navigation was ever waste.
     */
    private static final long STABLE_FOR_MS = 6_000;

    private static boolean settles(String label, long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        long since = -1;
        while (System.currentTimeMillis() < deadline) {
            if (Journey.find(Arrays.asList(label), true, 1_000) != null) {
                if (since < 0) since = System.currentTimeMillis();
                if (System.currentTimeMillis() - since >= STABLE_FOR_MS) return true;
            } else {
                // Seen and then lost is the revalidation refusing the token. Start again rather
                // than counting it: a session that flickered is not a session.
                if (since >= 0) {
                    Journey.mark("=====SETTLE '" + label + "' appeared then vanished after "
                            + (System.currentTimeMillis() - since) + "ms — revalidation refused it"
                            + "=====");
                }
                since = -1;
            }
            Journey.sleep(500);
        }
        return false;
    }

    /**
     * A session exists, SETTLED — the same question as {@link #hasAnySession()}, waited out.
     *
     * The Profile trip and the `sleep(6_000)` are gone for the reason given on
     * {@link #isSignedIn(String)}: "Sign out exists on Profile" and "the masthead shows a session"
     * are the same fact, and the masthead one needs no navigation and no guess about how to reach
     * Profile. The bell (`App.vue:622`) renders only under `auth.hasSession`.
     *
     * Kept separate from `hasAnySession` because the callers differ in what they can tolerate:
     * `hasAnySession` is a glance used to decide whether a sign-out is even needed, while this is a
     * VERDICT — `signIn` returns it — and a verdict has to outlast the revalidation window rather
     * than catch the identity the boot painted a moment before the api refuses it.
     */
    static boolean isSignedIn() {
        return settles("Notifications", 20_000);
    }

    /**
     * Signed in AS {@code identity} specifically.
     *
     * With one shared account "is there a session" was a sufficient question. With per-suite
     * accounts it is actively wrong: a session left by the PREVIOUS suite satisfies it, the suite
     * proceeds against someone else's data, and the collisions per-suite isolation exists to remove
     * come straight back — now harder to see, because everything looks signed in.
     */
    static boolean isSignedIn(String identity) {
        // READ THE MASTHEAD. No Profile trip, no scroll-to-Sign-out, no fixed sleeps.
        //
        // This used to navigate to Profile (`openProfile`, which opens with TWELVE hand-rolled
        // swipeDowns), scroll to "Sign out", `sleep(6_000)`, and scroll again — around 40 seconds,
        // paid by every suite, to answer a question the masthead answers directly. The profile link
        // is labelled `auth.user?.name || t('profile.title')` (App.vue:630) with a real text node at
        // :658, which is why Android can read it at all; a failure inventory from this very tier
        // shows `appjourneytests[TextView] | appjourneytests[View,click]` sitting in the masthead.
        //
        // The Profile trip was never verifying anything extra. "Sign out exists on Profile" and
        // "the masthead shows this account" are the same fact reached two ways, and the expensive
        // way also had to guess how to GET to Profile — which is how the hard-coded
        // ["Your profile","simtest","uitest"] list came to exist and then rot.
        //
        // What the sleeps were for is kept, in `settles`: boot paints the last known identity before
        // revalidating, so the answer has to hold across two reads rather than be believed on the
        // first one.
        return settles(identity, 20_000);
    }

    /**
     * Sign in through the dev picker as {@code identity}.
     *
     * The identity is a PARAMETER because suites get their own account: a hard-coded name puts
     * every suite in one shared world, where one suite's favourites decide another suite's result.
     * The mock provider mints an account for any name, so the picker is the whole mechanism and
     * nothing server-side has to change.
     */
    static boolean signIn(String identity) {
        UiObject2 entry = Journey.find(Arrays.asList("Sign in"), false, 20_000);
        if (entry == null) {
            // Already in? `/login` correctly bounces an AUTHENTICATED visitor to Home, so an app
            // that is already signed in can never show a "Sign in" link — and reporting that as a
            // failure is how the iOS twin failed a perfectly good session (2026-09-24).
            return isSignedIn();
        }
        try {
            // RAW CLICK, not `Journey.tap`, and that is a known hazard rather than an oversight:
            // `find` falls back to the inert TextView twin when no clickable match exists, and
            // clicking that does nothing at all, silently. Left as-is for now because changing it is
            // a behaviour change to the one path that still drives the real OAuth flow; the marker
            // below at least makes a swallowed tap visible instead of surfacing three steps later.
            entry.click();
        } catch (Throwable t) {
            Journey.mark("=====SIGNIN entry.click threw " + t + "=====");
            return false;
        }

        // Dev picker: a text field placeholdered "or a custom name…", whose submit stays disabled
        // until the field has content.
        //
        // RETRY the navigation, do not just wait longer. On a cold emulator the first tap can land
        // before the router is ready and is simply swallowed, so waiting 20s then 40s on a page
        // that was never navigated to is waiting for the wrong thing. And because every later suite
        // depends on this sign-in, the flake does not stay local — it leaves the app SIGNED OUT and
        // downstream suites then report missing controls that are correctly absent.
        UiObject2 input = waitForField(20_000);
        if (input == null) {
            Journey.mark("=====SIGNIN no dev picker after the first tap; re-tapping 'Sign in'=====");
            Journey.tap("Sign in", false, 10_000);
            input = waitForField(20_000);
        }
        if (input == null) {
            if (isSignedIn()) return true;
            throw new AssertionError(
                    "no dev identity input after two attempts, and the app is not signed in either. "
                            + "On screen: " + Journey.labelledInventory(12));
        }

        try {
            input.click();
            input.setText(identity);
        } catch (Throwable t) {
            Journey.mark("=====SIGNIN could not type into the dev picker: " + t + "=====");
            return false;
        }

        // The SUBMIT, not the masthead.
        //
        // `/login` carries TWO controls named "Sign in": the masthead entry that is on every
        // signed-out page, and the picker's submit. A plain lookup returns the masthead one — it
        // comes first in tree order — and clicking it navigates to `/login`, which is where we
        // already are. So the form is never submitted, nothing visibly fails, and the failure
        // surfaces later as "sign-in did not complete" with a filled-in form still on screen.
        //
        // The submit is the one BELOW the identity field; the masthead is above it. Position is the
        // only thing that separates them, because their accessible names are identical — which is
        // correct markup, not a bug to fix in the app.
        if (!Journey.tapBelow("Sign in", input, 10_000)) {
            // The submit is BELOW the field; the masthead link is above it. A miss here means one of
            // them was not where this expects, and without a marker it surfaces only as
            // "sign-in did not complete" long after the fact.
            Journey.mark("=====SIGNIN the picker's submit was not below the field :: "
                    + Journey.labelledInventory(16) + "=====");
            return false;
        }

        // The OAuth hand-off leaves the app: Capacitor opens a Custom Tab, which may show a consent
        // or account screen owned by ANOTHER package. While it is up every app-scoped query returns
        // empty, which reads as "the page rendered nothing" — the Android twin of the SpringBoard
        // alert that cost the iOS tier a full diagnosis. So look for the continue control WITHOUT
        // the package filter every other lookup here uses.
        UiObject2 consent = Journey.device().wait(
                Until.findObject(By.textContains("Continue").clickable(true)), 8_000);
        if (consent == null) {
            consent = Journey.device().wait(
                    Until.findObject(By.textContains("Allow").clickable(true)), 2_000);
        }
        if (consent != null) {
            try {
                consent.click();
            } catch (Throwable ignored) {
                // Already dismissed itself; the session check below is the real verdict.
            }
        }

        // OAuth returns to HOME, and "Sign out" is on Profile. Asking Home reported "sign-in did
        // not complete" after sign-ins that had completed, with the minted token sitting in the
        // device's preferences. Ask the page that can actually answer.
        return isSignedIn();
    }

    /**
     * The dev picker's text field, found by ENUMERATION for the same reason as `Journey.find`.
     *
     * This used `Until.findObject(By.pkg(PKG).clazz("android.widget.EditText"))`. `BySelector`
     * matching is not reliable against WebView content — `By.desc` was measured to match nothing at
     * all for web nodes (see `Journey.find`) — and the class filter fails the same way here: the
     * field is on screen and the selector returns null, so sign-in reports "no dev identity input"
     * about a picker that is rendered and waiting (2026-09-25).
     *
     * Enumerating and reading `getClassName()` off each materialised node sees what is there.
     */
    private static UiObject2 waitForField(long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        do {
            // PROVE WE ARE ON /login BEFORE TAKING ANY EditText.
            //
            // This returned the first EditText anywhere in the package, which answers "is there a
            // text field on screen" when the question is "is the dev picker rendered". Those differ
            // on every page that has a search box — which is Home and Discover, i.e. exactly where
            // the app sits when the preceding tap to reach /login was swallowed.
            //
            // Measured 2026-09-28: a sign-in that never left Home typed the identity into HOME'S
            // SEARCH BOX and the suite failed as "sign-in did not complete" with
            // `simtest[EditText,click]` in the inventory beside "Ask across every episode".
            //
            // Worse, it DEFEATED the retry that exists for precisely that case. `signIn` re-taps
            // "Sign in" and calls this again only when it returns null (see the retry above); a
            // wrong field is non-null, so the healing path could never run. A wrong answer here is
            // not a slower failure, it is a silent one that disables the recovery.
            //
            // Gate on text only `/login` renders — `LoginView.vue:68` ("Sign in as") and `:99`
            // ("Dev sign-in (mock OAuth)"), both plain `<p>` copy that arrives as TextView text.
            // Deliberately NOT the field's own placeholder ("or a custom name…", `:86`): the input
            // carries no `aria-label`, so its accessible name depends on Chromium surfacing the
            // placeholder on an empty field, which is unverified on this WebView. The page gate
            // needs no such assumption.
            if (Journey.find(Arrays.asList("Sign in as", "Dev sign-in"), true, 1_000) != null) {
                try {
                    for (UiObject2 o : Journey.device().findObjects(By.pkg(Journey.PKG))) {
                        String cls = String.valueOf(o.getClassName());
                        if (cls.endsWith("EditText")) return o;
                    }
                } catch (Throwable ignored) {
                    // Tree mutated mid-walk; the loop covers it.
                }
            }
            Journey.sleep(400);
        } while (System.currentTimeMillis() < deadline);
        return null;
    }

    /**
     * Sign out and stay out — the precondition for `/offline`, which only renders when signed out.
     */
    static boolean signOut() {
        if (!hasAnySession()) return true; // already out; the caller's precondition holds

        // RETRY, and re-resolve the control each time.
        //
        // A single tap reported success and left the app signed in, on a Discover page it had
        // navigated to — so the click landed on SOMETHING, just not the button (2026-09-24). Two
        // ways that happens here and both are invisible from the return value: `find` prefers a
        // clickable ancestor, which for a full-width button row can be a much larger container; and
        // a node resolved before a scroll is stale afterwards, so clicking it quietly does nothing.
        //
        // The app is not the suspect: `auth.logout()` drops the local identity in a `finally`
        // precisely so a sign-out with no network still works.
        for (int attempt = 1; attempt <= 3; attempt++) {
            // HOME FIRST. `setOfflineMode` leaves the app on Settings, where "Sign out" does not
            // exist, and `signOut` runs inside `ensureSignedIn` — before `startClean` gets the app
            // back to Home. Measured on the iOS twin 2026-09-29 (all four NativeOnlySurfacesTests).
            Journey.openTab("Home");
            // OPEN PROFILE WITHOUT KNOWING WHOSE IT IS. `signOut` runs precisely when the session
            // belongs to SOMEONE ELSE, so the account name the masthead link carries is the one
            // thing it cannot be told. The label-less `openProfile()` answered with a hardcoded
            // ["Your profile", "simtest", "uitest"], right only while every suite shared `simtest`.
            // Read the name off the link NEXT TO THE BELL. The bell and the profile link render
            // together under `auth.hasSession` (`App.vue:622`, `:627`), so the nearest clickable to
            // the bell's right in its row is the account, whatever it is called. A name-based scan
            // of the top band — the iOS approach — does not port: Android's tree carries buttons
            // too ("Backend target DEV"), and page content scrolls into the band. Measured
            // 2026-09-29, three attempts tapping the wrong control.
            List<String> candidates = new ArrayList<>();
            String account = accountNextToBell();
            if (!account.isEmpty()) candidates.add(account);
            Journey.mark("=====SIGNOUT profile candidates " + candidates + "=====");
            List<String> labels = new ArrayList<>(candidates);
            labels.add("Your profile");
            Journey.openProfile(labels);
            // SELECT THE ACCOUNT TAB. Profile is tabbed (Account / Topics / Stats) and "Sign out"
            // lives in the Account panel (`ProfileView.vue:803`), so on any other tab it is not
            // below the fold — it is NOT RENDERED, and scrolling cannot produce it.
            //
            // The tab STICKS: ProfileView is kept alive (`ProfileView.vue:89`, KEEP_ALIVE_TABS), so
            // the last tab anything looked at is the tab the next `openProfile` lands on.
            //
            // FOUND ON iOS (2026-09-28), where it failed three identical attempts. Added here
            // UNFIRED, deliberately: Android's `relaunch` uses FLAG_ACTIVITY_CLEAR_TASK, which
            // rebuilds the WebView and discards the keep-alive state, so a sign-out that follows a
            // relaunch has always landed on Account by luck. Within a single test — walk
            // Profile ▸ Topics, then sign out — Android has the same defect iOS just demonstrated.
            // Recording it on the platform that has not paid for it yet is the point of the drift
            // ledger; the alternative is finding it here in six weeks and calling it new.
            Journey.tap(Arrays.asList("Account"), false, 5_000);
            UiObject2 out = Journey.scrollTo("Sign out", false);
            // NOTHING TO SIGN OUT OF: the masthead shows only the GENERIC "Your profile" AND there
            // is no "Sign out" — both say `auth.isAuthenticated` is false (`v-if` on the button,
            // `auth.user?.name || 'Your profile'` on the link), while `hasAnySession` still answers
            // yes off the bell, which renders under `auth.hasSession`. A stale local session.
            // Retrying the tap cannot help; a relaunch re-runs the boot revalidation, `/me` refuses
            // the token, and the app settles into a real signed-out state with a Sign in link.
            // Measured on the iOS twin 2026-09-29. `ensureSignedIn` verifies the identity after.
            if (out == null && candidates.equals(Arrays.asList("Your profile"))) {
                Journey.mark("=====SIGNOUT nothing to sign out of — no account name, no Sign out=====");
                relaunch();
                return true;
            }
            if (out == null) {
                Journey.mark("=====SIGNOUT attempt " + attempt
                        + ": no 'Sign out' control on Profile :: " + Journey.labelledInventory(80)
                        + "=====");
                continue;
            }
            // `Journey.tap`, NOT a raw `click()` on the node `scrollTo` returned.
            //
            // "Sign out" is the LAST control on Profile, so scrolling to it parks it at the bottom
            // of the screen — measured at Rect(47, 2207 - 1031, 2328) on a 2400-tall display, i.e.
            // UNDERNEATH the bottom nav. The click landed on "Discover", the app navigated there,
            // and the session was of course still present. `tap` exists precisely to lift a control
            // clear of that overlap before touching it, and this call site had skipped it — the
            // same trap the iOS `Journey` documents for the transport row.
            boolean tapped = Journey.tap("Sign out", false, 10_000);
            Journey.sleep(3_000);
            if (!hasAnySession()) return true;
            Journey.mark("=====SIGNOUT attempt " + attempt + " tapped=" + tapped
                    + " but a session is still present :: " + Journey.labelledInventory(80) + "=====");
        }
        return false;
    }

    /** Name of the nearest clickable to the right of the notifications bell, in its row. */
    private static String accountNextToBell() {
        UiObject2 bell = Journey.find(Arrays.asList("Notifications"), true, 3_000);
        if (bell == null) return "";
        try {
            Rect bb = Journey.attr(bell, UiObject2::getVisibleBounds);
            if (bb == null) return "";
            UiObject2 best = null;
            int bestLeft = Integer.MAX_VALUE;
            for (UiObject2 o : Journey.device().findObjects(By.pkg(Journey.PKG).clickable(true))) {
                Rect b = Journey.attr(o, UiObject2::getVisibleBounds);
                if (b == null || b.left <= bb.left) continue;
                if (b.centerY() < bb.top || b.centerY() > bb.bottom) continue;
                if (b.left < bestLeft) {
                    bestLeft = b.left;
                    best = o;
                }
            }
            return best == null ? "" : Journey.nameOf(best);
        } catch (Throwable ignored) {
            return ""; // tree mutated mid-walk; "Your profile" + openProfile's fallback cover it
        }
    }

    /**
     * Mint a native session token over HTTP, the way the Makefile's `ios-journey-signin` does.
     *
     * The mock provider answers `/api/app/auth/login?as=<id>&platform=native` with a redirect chain
     * ending at `closelistening://auth#token=<signed>`. Reachable from the emulator on 127.0.0.1
     * because the tier sets `adb reverse` for the origin port before any suite runs.
     *
     * Redirects are followed BY HAND (`setInstanceFollowRedirects(false)`) for two reasons: the last
     * hop is a custom scheme `HttpURLConnection` cannot fetch, and the token lives in the `Location`
     * header, not in any body. Cookies carry forward because the provider keeps flow state in one.
     *
     * Returns null rather than throwing, so a mint that cannot reach the origin surfaces as the
     * origin's problem and the caller can still fall back to the UI.
     */
    private static String mintNativeToken(String identity) {
        // URL-ENCODED. Identities are derived from class names today, so they are already
        // `[a-z0-9]` — but `accountIdentity()` is overridable and an identity with a `&` or a space
        // would silently truncate the query and mint a token for the WRONG account, which is the
        // one failure here that would not look like a failure.
        String as;
        try {
            as = java.net.URLEncoder.encode(identity, "UTF-8");
        } catch (java.io.UnsupportedEncodingException e) {
            return null; // UTF-8 is guaranteed; unreachable in practice
        }
        String url = "http://127.0.0.1:" + Journey.originPort()
                + "/api/app/auth/login?as=" + as + "&platform=native";
        java.util.Map<String, String> jar = new java.util.LinkedHashMap<>();
        for (int hop = 0; hop < 6; hop++) {
            try {
                java.net.HttpURLConnection c =
                        (java.net.HttpURLConnection) new java.net.URL(url).openConnection();
                c.setInstanceFollowRedirects(false);
                c.setConnectTimeout(5_000);
                c.setReadTimeout(5_000);
                if (!jar.isEmpty()) {
                    StringBuilder header = new StringBuilder();
                    for (java.util.Map.Entry<String, String> e : jar.entrySet()) {
                        if (header.length() > 0) header.append("; ");
                        header.append(e.getKey()).append('=').append(e.getValue());
                    }
                    c.setRequestProperty("Cookie", header.toString());
                }
                c.connect();
                // Cookies replaced BY NAME, not appended. Appending sent `sid=a; sid=b` once the
                // provider re-issued a cookie across hops — the server then picks whichever it likes
                // and the flow state is a coin toss. Keyed so the newest value for a name wins.
                java.util.List<String> set = c.getHeaderFields().get("Set-Cookie");
                if (set != null) {
                    for (String s : set) {
                        String pair = s.split(";", 2)[0].trim();
                        int eq = pair.indexOf('=');
                        if (eq <= 0) continue;
                        jar.put(pair.substring(0, eq), pair.substring(eq + 1));
                    }
                }
                String loc = c.getHeaderField("Location");
                c.disconnect();
                // EVERY null names the hop it stopped at, as the iOS twin does ("CALLBACK mint
                // stopped at <url>"); only the exception path used to, so a mint that ended on a
                // non-redirect or an unexpected Location failed with nothing to explain it.
                if (loc == null) {
                    Journey.mark("=====MINT stopped at " + url + ": HTTP " + c.getResponseCode()
                            + " with no Location=====");
                    return null;
                }
                if (loc.startsWith("closelistening://")) {
                    int at = loc.indexOf("#token=");
                    if (at < 0) Journey.mark("=====MINT callback carried no token: " + loc + "=====");
                    return at < 0 ? null : loc.substring(at + "#token=".length());
                }
                // PROTOCOL-RELATIVE FIRST. `//host/path` starts with "/" too, so the path branch
                // below would have turned it into `http://127.0.0.1:4174//host/path` — a URL that
                // resolves to nothing, reported as a mint failure pointing at the wrong thing.
                if (loc.startsWith("//")) {
                    url = "http:" + loc;
                } else if (loc.startsWith("/")) {
                    url = "http://127.0.0.1:" + Journey.originPort() + loc;
                } else if (loc.startsWith("http")) {
                    url = loc;
                } else {
                    Journey.mark("=====MINT stopped at " + url + ": unrecognised Location " + loc + "=====");
                    return null;
                }
            } catch (Throwable t) {
                Journey.mark("=====MINT failed at " + url + " :: " + t + "=====");
                return null;
            }
        }
        Journey.mark("=====MINT gave up after 6 redirects, last at " + url + "=====");
        return null;
    }

    /**
     * Sign in by delivering the OAuth callback directly — no Custom Tab, no consent screen.
     *
     * ## Why: the UI path is six races and one of them is in another process
     *
     * {@link #signIn} finds "Sign in", waits for the dev picker, types, disambiguates TWO controls
     * named "Sign in" by POSITION, then waits on a consent screen owned by `com.android.chrome`.
     * Any of those failing surfaces as the same line — "sign-in did not complete as <id>" — and the
     * tier pays it once per suite across 14 suites. Measured 2026-09-28 while iterating on the
     * accessible-name audit: 2 failures in 9 runs.
     *
     * The Custom Tab is the worst of the six because it is not ours. `Makefile:2327-2344` already
     * disables Android's cached-app freezer for precisely this, with the trace recorded: consent
     * page opens, freezer suspends Chrome ten seconds later, sign-in gives up, Chrome unfreezes one
     * second too late. Pinning the environment removed one cause; the race is structural.
     *
     * This removes the path instead of hardening it, and it is NOT a bypass. `native.ts:80` states
     * the mechanism — "Android: @capacitor/browser + the manifest intent-filter delivers the
     * callback via appUrlOpen" — so an `ACTION_VIEW` on the callback URL runs the same listener,
     * the same `storeAuthToken`, the same `/me` refetch that production does. The deleted
     * `defaults write` seeds were killed for bypassing the app; this deliberately does not.
     *
     * THE APP MUST ALREADY BE RUNNING. `initNativeAuth` listens on `appUrlOpen` ONLY, and
     * `native.ts:213-215` records that `appUrlOpen` does NOT fire for a link that launched the app
     * — that arrival is `getLaunchUrl`, which only the routing listener reads. A cold launch by this
     * URL would therefore drop the token in silence, which is the worst available outcome. The guard
     * below refuses to fire until web content is on screen.
     */
    static boolean signInViaCallback(String identity) {
        // "Sign in" PRESENT, not merely "some web content is up".
        //
        // The first version accepted "Sign in" OR "Your profile" OR the identity as proof the
        // WebView had painted. That makes the success postcondition below — "'Sign in' is gone" —
        // satisfiable by the precondition itself: called on an app that is ALREADY signed in, it
        // would return true at once, having fired nothing and proved nothing. Only one caller
        // exists today and it signs out first, so the hole is latent; the method is package-visible
        // and the trap is one new caller away.
        //
        // Requiring the signed-out masthead keeps both jobs honest: it still proves web content is
        // painted (which is what `appUrlOpen` needs, since `initNativeAuth` does not read
        // `getLaunchUrl`), and it makes the postcondition a real state CHANGE rather than a tautology.
        if (Journey.find(Arrays.asList("Sign in"), false, 20_000) == null) {
            Journey.mark("=====CALLBACK no signed-out 'Sign in' on screen: either no web "
                    + "content yet (appUrlOpen would never see the token) or a session already "
                    + "exists, and neither is a state this path can act on :: "
                    + Journey.labelledInventory(12) + "=====");
            return false;
        }
        String token = mintNativeToken(identity);
        if (token == null || token.isEmpty()) return false;

        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Intent cb = new Intent(Intent.ACTION_VIEW,
                Uri.parse("closelistening://auth#token=" + token));
        cb.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        cb.setPackage(Journey.PKG);
        try {
            ctx.startActivity(cb);
        } catch (Throwable t) {
            Journey.mark("=====CALLBACK startActivity threw " + t + "=====");
            return false;
        }
        // WAIT FOR THE SESSION TO APPEAR, not for "Sign in" to disappear.
        //
        // The first version polled for the absence of "Sign in", and that conflated two different
        // things: being SIGNED IN, and having NAVIGATED AWAY from the login page. `storeAuthToken`
        // calls `onAuthed()`, which refreshes the auth store, so the masthead re-renders to the
        // account name immediately — while the ROUTE is still `/login`, whose body carries its own
        // "Sign in" SUBMIT button. Signed in, and still showing a control named "Sign in".
        //
        // Measured 2026-09-28. The marker at the moment of failure read:
        //
        //   =====CALLBACK token delivered but 'Sign in' is still on screen ::
        //         … | Notifications[Button,click] | simtest[TextView] | simtest[View,click]=====
        //
        // `simtest` in the masthead — the callback had worked. This returned false anyway,
        // `ensureSignedIn` fell through to the UI path, `signIn` ran on an already-signed-in app,
        // and `waitForField` grabbed the first EditText on screen, which was Discover's SEARCH box.
        // The suite then failed as "sign-in did not complete" with the identity typed into search.
        //
        // It only bit after an account SWITCH, which is why a standalone run never showed it: from
        // a cleared device the app sits signed-out on Home, where the only "Sign in" is the masthead
        // link and it really does vanish. After `signOut()` the app is on `/login`, where a second
        // one exists in the page body.
        //
        // So: poll for the masthead's SIGNED-IN marker. The identity once `/me` resolves, or the
        // generic label before it does — `App.vue` labels the link `auth.user?.name ||
        // t('profile.title')`. A positive signal cannot be satisfied by a leftover route.
        long started = System.currentTimeMillis();
        long deadline = started + 20_000;
        while (System.currentTimeMillis() < deadline) {
            if (Journey.find(Arrays.asList(identity, "Your profile"), true, 1_000) != null) {
                // A POSITIVE marker, not merely the absence of a failure one. Which path signed the
                // app in is the thing being measured here, and "no error was printed" is exactly the
                // kind of evidence this tier has been fooled by before.
                Journey.mark("=====CALLBACK signed in as " + identity + " in "
                        + (System.currentTimeMillis() - started) + "ms, no Custom Tab=====");
                return true;
            }
            Journey.sleep(500);
        }
        Journey.mark("=====CALLBACK token delivered but no signed-in masthead appeared :: "
                + Journey.labelledInventory(12) + "=====");
        return false;
    }

    /** Leave the app signed in as {@code identity}, whatever it was signed in as before. */
    static boolean ensureSignedIn(String identity) {
        if (isSignedIn(identity)) return true;
        // Signed in as SOMEONE ELSE: the dev picker is unreachable while a session exists (there is
        // no "Sign in" link on a signed-in app), so without this the suite would silently keep the
        // previous suite's account — the exact collision per-suite identities exist to prevent.
        //
        // `hasAnySession`, not `isSignedIn(identity)`: this branch runs when the session belongs to
        // SOMEONE ELSE, so the one thing it cannot supply is the name to look for.
        if (hasAnySession() && !signOut()) return false;
        // THE CALLBACK PATH, AND ONLY IT. No silent fallback to the UI.
        //
        // There used to be one, defended as "so a broken callback cannot take the tier down", with
        // the UI flow's coverage supposedly preserved because HarnessSmokeTests "still drives it
        // FIRST". Both halves were wrong. The smoke test signed in through `startClean` like
        // everything else, so the real flow was exercised by NOTHING until a test was added for it
        // (1653a8a34) — and the fallback's own defence is the argument against it, written in this
        // repo by me, three files away: `Journey.originPort()` refuses to default precisely because
        // a silent fallback makes the tier "pass — slower, flakier, and with the reason invisible".
        //
        // That is exactly what it did. When the callback path broke this afternoon the tier did not
        // report a broken callback; it fell through to the UI, ran `signIn` against an
        // already-signed-in app, typed the identity into Home's search box, and failed four steps
        // later as "sign-in did not complete". The fallback converted a precise failure into a
        // confusing one.
        //
        // So it fails loudly here instead. The UI flow keeps its coverage where it belongs — one
        // dedicated test in HarnessSmokeTests that drives `signIn` directly, once per tier, and
        // whose failure means the thing it names.
        if (!signInViaCallback(identity)) {
            Journey.mark("=====SIGNIN callback path did not land, and there is no fallback :: "
                    + Journey.labelledInventory(16) + "=====");
            return false;
        }
        // VERIFY THE IDENTITY, not merely that a session exists — `ensureSignedIn(identity)`
        // promises an ACCOUNT, and certifying the previous suite's session as this one's is the
        // isolation #2091 exists to provide.
        return isSignedIn(identity);
    }

    /**
     * Open an episode by slug through the app's deep-link scheme.
     *
     * Replaces navigating by taps and blind swipes, which was the single flakiest thing in the iOS
     * tier: the lists keep their scroll position between visits, so a swipe-and-look search walks
     * past rows and on one run mis-tapped and downloaded an episode the test never asked for. A
     * deep link addresses the episode directly, which is what the link is FOR — the test exercises
     * a product capability rather than working around the lack of one.
     */
    static void openEpisode(String slug) {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Intent view = new Intent(Intent.ACTION_VIEW, Uri.parse("closelistening://episode/" + slug));
        view.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        view.setPackage(Journey.PKG);
        ctx.startActivity(view);
        Journey.device().wait(Until.hasObject(By.pkg(Journey.PKG).depth(0)), 15_000);
    }

    /** True as soon as the WebView's accessibility tree holds anything labelled; false at the deadline. */
    private static boolean awaitPainted(long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        do {
            if (!Journey.labelledInventory(1).startsWith("<nothing labelled")) return true;
            Journey.sleep(500);
        } while (System.currentTimeMillis() < deadline);
        return false;
    }

    /** Cold start — relaunch so what is on disk is re-read, not just re-activated. */
    static void relaunch() {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Journey.device().pressHome();
        Intent launch = ctx.getPackageManager().getLaunchIntentForPackage(Journey.PKG);
        if (launch == null) throw new AssertionError("the app under test is not installed");
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        ctx.startActivity(launch);
        Journey.device().wait(Until.hasObject(By.pkg(Journey.PKG).depth(0)), 30_000);
        // WAIT FOR THE WEB ACCESSIBILITY TREE, not just for the process.
        //
        // `Until.hasObject(By.pkg(...))` is satisfied by the native shell — the WebView container
        // exists long before Chromium has built the tree for its content, and Chromium builds that
        // tree lazily. A suite that looks immediately sees a foregrounded app with NOTHING in it,
        // and the failure reads "sign-in did not complete ... <nothing labelled; foreground window
        // = app.closelistening.player>", which points at the session rather than at the timing
        // (2026-09-25, ConfigOfflineToggleTests).
        //
        // A fixed sleep was what stood here, and a fixed sleep is a guess about the slowest machine
        // anyone will ever run this on. Poll for real content instead, and keep a ceiling so a
        // genuinely blank app still fails rather than hanging.
        boolean painted = awaitPainted(30_000);

        // ONE RELAUNCH IF THE WEBVIEW NEVER PAINTED (2026-09-27).
        //
        // Waiting longer is not the answer past this point: if Chromium has not built a tree in
        // 30s the load has FAILED rather than slowed, and another 30s of polling changes nothing.
        // A fresh launch does — it is what a person would do — and it is bounded to one attempt so
        // a genuinely blank app still fails instead of looping.
        //
        // MEASURED twice on 2026-09-27, in two different suites, with the BACKEND HEALTHY both
        // times: the api was serving 200s and recording playback within a minute of the failure,
        // so this is a client-side paint failure, not an outage.
        //     sign-in did not complete as personalisationtests
        //     On screen: <nothing labelled; foreground window = app.closelistening.player>
        // Each occurrence cost a full tier run (~70 min), landing on whichever suite came next.
        //
        // DISTINCT from the OAuth-tab freeze fixed in the Makefile (the cached-app freezer
        // suspending com.android.chrome mid-consent): there the browser was in front and stalled,
        // here the app's own WebView is foregrounded and empty.
        for (int attempt = 1; attempt <= 2 && !painted; attempt++) {
            // NO FORCE-STOP. It used to run `am force-stop app.closelistening.player` here, and
            // that cannot work: Android instrumentation runs INSIDE THE TARGET APP'S PROCESS, so
            // the command kills the app and the test issuing it in the same breath.
            //
            // That is why this recovery was recorded as never having fired in ~40 sign-ins and
            // therefore untested. The first time it did fire — 2026-09-28, this suite — the run
            // ended with `INSTRUMENTATION_RESULT: shortMsg=Process crashed.` and a single marker,
            // which says nothing about the WebView and everything about the harness shooting
            // itself. A recovery whose success case is indistinguishable from a crash is worse
            // than no recovery, because it also destroys the evidence.
            //
            // The original comment reasoned that re-issuing the launch intent "does not help", and
            // that observation stands — but the conclusion drawn from it does not, because the
            // alternative it chose is unavailable from in-process. So: try the relaunch, which is
            // free and occasionally works, and if the tree is still empty FAIL SAYING SO. A precise
            // failure is worth more than an impossible recovery.
            Journey.mark("=====RELAUNCH webview blank; re-issuing the launch intent, attempt "
                    + attempt + "/2 (no force-stop: it would kill this test's own process)=====");
            Journey.sleep(2_000);
            ctx.startActivity(launch);
            Journey.device().wait(Until.hasObject(By.pkg(Journey.PKG).depth(0)), 30_000);
            painted = awaitPainted(30_000);
        }

        // STILL BLANK AFTER TWO TRIES: say so, here, in the words of the thing that is wrong.
        //
        // Without this the caller carries on and fails somewhere downstream as "sign-in did not
        // complete" or "no Download control", naming a surface rather than the cause — which is
        // how this defect cost two full tier runs on 2026-09-27 and got recorded as a sign-in
        // problem. The WebView never painted; nothing after this point can succeed, and the first
        // assertion to notice will be about something else entirely.
        //
        // WAITED, NOT SAMPLED (2026-09-29). This was one read, and so were the loop conditions
        // above, so a tree that had painted and was empty for an instant at this line failed as
        // "never painted" — measured: `ConfigOfflineToggleTests` died here in 9.9s with NO
        // `RELAUNCH` marker, i.e. the loop saw content and this line saw none. Every check now
        // waits for content, so only a tree that stays empty fails.
        if (!painted && !awaitPainted(5_000)) {
            throw new AssertionError(
                    "the app's WebView never painted after two launches — the accessibility tree is "
                            + "empty while `" + Journey.PKG + "` is foregrounded. This is a "
                            + "client-side paint failure, not a backend outage: when it was measured "
                            + "on 2026-09-27 the api was serving 200s throughout. Nothing this suite "
                            + "does next can work, so it stops here rather than failing later about "
                            + "a control that was never going to be on screen.");
        }

        // Boot paints the device snapshot and then revalidates; assert after that lands.
        Journey.sleep(5_000);
    }
}
