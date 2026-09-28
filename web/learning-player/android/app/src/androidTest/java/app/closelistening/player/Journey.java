package app.closelistening.player;

import android.graphics.Rect;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.BySelector;
import androidx.test.uiautomator.UiDevice;
import androidx.test.uiautomator.UiObject2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Shared helpers for the Android device tier (#2139) — the sibling of the iOS `Journey.swift`.
 *
 * It is a port of that file's DESIGN, not its lines. The two tiers answer the same question ("is
 * this control reachable to someone who cannot see the screen?") against different bridges, so the
 * laws carry over and the plumbing does not.
 *
 * ## What carries over from iOS
 *
 * `data-testid` is NOT an accessibility identifier on either platform. Web content surfaces no
 * `resource-id`, so elements are addressed by their accessible NAME, taken from
 * `src/i18n/locales/en.json`. Scrolling terminates on the page STALLING, never on a swipe count —
 * a count encodes how long a page happens to be today and erodes silently as surfaces grow.
 * Diagnostics must not be able to fail: one that throws replaces the real failure with its own.
 *
 * ## What is different here, measured 2026-09-24 (WebViewA11yProbe)
 *
 * 1. **The name lands in one of TWO fields, and which one depends on the element.** Text-bearing
 *    leaves arrive as `TextView` with `text` set and `contentDescription` empty; interactive
 *    wrappers arrive as `View` with `contentDescription` set and `text` empty. On iOS both are
 *    `label`. Every lookup here therefore queries both.
 * 2. **Every control appears TWICE.** A button renders as a `TextView text=Sign in` AND a
 *    `View desc=Sign in`. Taking the first match lands half of all taps on an inert label, so
 *    `find` prefers the CLICKABLE node and falls back to the other only when there is none.
 * 3. **The tree is built only while an accessibility client is connected.** `adb shell uiautomator
 *    dump` reported the WebView as NAF with zero children four times running; the same app inside
 *    an instrumented test exposes 54 nodes within ~3ms. Do not take a shell dump as evidence that
 *    content is missing — it is evidence about the dump.
 */
final class Journey {

    static final String PKG = "app.closelistening.player";

    /**
     * The single origin the app talks to, reachable from the device on 127.0.0.1.
     *
     * Not the emulator's 10.0.2.2 alias: the tier sets `adb reverse tcp:4174` so the Android build's
     * API base is BYTE-IDENTICAL to the iOS one and the two tiers cannot drift apart on
     * configuration (Makefile, `test-android`). Mirrors `IOS_ORIGIN_PORT`; if that moves, this moves.
     */
    static final int ORIGIN_PORT = 4174;

    private Journey() {}

    static UiDevice device() {
        return UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
    }

    // ---------------------------------------------------------------- evidence

    /**
     * A bounded, failure-PROOF inventory for diagnostic messages.
     *
     * Every accessor is guarded. The iOS twin of this helper twice blew up on its own snapshot
     * mid-walk and replaced the real failure with its own; the tree mutates while you iterate it
     * here too. It is allowed to return less than the whole truth. It is not allowed to throw.
     *
     * UNLABELLED nodes are REPORTED, not filtered. An interactive element that is in the tree with
     * no accessible name is invisible both here and to {@link #find}, so dropping it would let
     * "absent from the inventory" read as "absent from the tree" — two bugs with opposite fixes.
     */
    static String labelledInventory(int limit) {
        List<String> out = new ArrayList<>();
        try {
            for (UiObject2 o : device().findObjects(By.pkg(PKG))) {
                if (out.size() >= limit) break;
                String name = nameOf(o);
                Boolean click = attr(o, UiObject2::isClickable);
                String cls = String.valueOf(attr(o, UiObject2::getClassName));
                cls = cls.substring(cls.lastIndexOf('.') + 1);
                if (name.isEmpty()) {
                    if (Boolean.TRUE.equals(click)) out.add("<UNLABELLED>[" + cls + "]");
                } else {
                    out.add(name + "[" + cls + (Boolean.TRUE.equals(click) ? ",click" : "") + "]");
                }
            }
        } catch (Throwable ignored) {
            // A diagnostic that throws is worse than a thin one.
        }
        if (out.isEmpty()) {
            // EMPTY has two very different causes and they need opposite fixes: the app is not in
            // the foreground (a launch, a crash, a system dialog stealing focus), or it is there
            // with an empty tree. Naming the foreground window separates them in one line — without
            // it, "<nothing labelled>" sent a diagnosis down the wrong path for an hour.
            return "<nothing labelled; foreground window = " + foregroundWindow() + ">";
        }
        return String.join(" | ", out);
    }

    /** Which app currently owns the screen — the app under test, the launcher, or a system dialog. */
    static String foregroundWindow() {
        try {
            return String.valueOf(device().getCurrentPackageName());
        } catch (Throwable t) {
            return "<unknown>";
        }
    }

    /** Dump the raw hierarchy — for the case where names alone cannot separate two hypotheses. */
    static String hierarchy() {
        try {
            java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();
            device().dumpWindowHierarchy(buf);
            return buf.toString();
        } catch (Throwable t) {
            return "<hierarchy unavailable: " + t + ">";
        }
    }

    // ------------------------------------------------------------------ lookup

    /** The accessible name, wherever it landed. Empty when the node carries none. */
    static String nameOf(UiObject2 o) {
        String t = String.valueOf(attr(o, UiObject2::getText));
        if (!"null".equals(t) && !t.trim().isEmpty()) return t.trim();
        String d = String.valueOf(attr(o, UiObject2::getContentDescription));
        if (!"null".equals(d) && !d.trim().isEmpty()) return d.trim();
        return "";
    }

    /**
     * First element matching ANY of {@code names} — by ENUMERATING nodes, not by `BySelector`.
     *
     * ## `By.desc(...)` DOES NOT MATCH WEBVIEW CONTENT. Measured 2026-09-25.
     *
     * The Pause button was on screen, `clickable=true`, with `desc=[Pause]` read straight off the
     * node — and every one of these returned nothing:
     *
     *     By.pkg(PKG).desc("Pause")        -> null
     *     By.desc("Pause")                 -> null   (unscoped, so it is not the package filter)
     *     By.pkg(PKG).descContains("Pause")-> null
     *
     * while `findObjects(By.pkg(PKG))` returned that very node and `getContentDescription()` on it
     * gave "Pause". So selector matching on `desc` is evaluated against something that does not
     * carry web content's descriptions, even though reading the materialised node does.
     *
     * The consequence is worth stating plainly: every lookup in this harness that appeared to work
     * was working through `By.text`, i.e. through nodes where Chromium populates `text`. Every
     * icon-only control whose name lives in `contentDescription` was silently unfindable, and the
     * failures read as "the control is not there" — which is how an "Add to queue" that was plainly
     * on screen got reported as absent on Android.
     *
     * It also explains why adding `sr-only` spans fixed findability as well as the screen-reader
     * name: an `sr-only` span is a real TEXT node.
     *
     * Enumeration is slower than a selector. It is the only form that sees what is actually there.
     *
     * Prefers a CLICKABLE match. Web content exposes each control twice — an inert `TextView`
     * carrying the words and a `View` carrying the same name as its contentDescription and the
     * click handler — and UI Automator returns them in tree order, which puts the label first.
     * Tapping that does nothing at all, silently, and the failure surfaces several steps later as
     * "the page never changed".
     */
    static UiObject2 find(List<String> names, boolean contains, long timeoutMs) {
        if (names == null || names.isEmpty()) return null;
        long deadline = System.currentTimeMillis() + timeoutMs;
        do {
            UiObject2 fallback = null;
            List<UiObject2> all;
            try {
                all = device().findObjects(By.pkg(PKG));
            } catch (Throwable t) {
                all = java.util.Collections.emptyList();
            }
            for (UiObject2 o : all) {
                String name = nameOf(o);
                if (name.isEmpty() || !matches(name, names, contains)) continue;
                if (Boolean.TRUE.equals(attr(o, UiObject2::isClickable))) return o;
                UiObject2 clickable = clickableAncestorOf(o);
                if (clickable != null) return clickable;
                if (fallback == null) fallback = o;
            }
            if (fallback != null) return fallback;
            sleep(400);
        } while (System.currentTimeMillis() < deadline);
        return null;
    }

    private static boolean matches(String name, List<String> wanted, boolean contains) {
        for (String w : wanted) {
            if (contains ? name.toLowerCase().contains(w.toLowerCase()) : name.equalsIgnoreCase(w)) {
                return true;
            }
        }
        return false;
    }

    static UiObject2 find(String name, boolean contains, long timeoutMs) {
        return find(Arrays.asList(name), contains, timeoutMs);
    }

    /**
     * How many DISTINCT controls carry this exact name.
     *
     * For asserting UNIQUENESS before tapping. The iOS twin needed this after an unscoped lookup
     * matched a different episode that happened to be downloaded already and reported a success it
     * had not produced — `Library → Downloaded` renders the same controls bare and that view is
     * kept alive. A count of one cannot pick wrong; a count above one says so out loud instead of
     * guessing.
     *
     * Counts by BOUNDS, not by node, because web content exposes every control twice — once as the
     * text and once as the interactive wrapper — so a raw node count reports two for a page with
     * one control and the assertion fires on healthy markup.
     */
    static int countDistinct(String name) {
        java.util.Set<String> seen = new java.util.HashSet<>();
        try {
            for (BySelector sel : selectorsFor(name, false)) {
                for (UiObject2 o : device().findObjects(sel)) {
                    Rect b = attr(o, UiObject2::getVisibleBounds);
                    if (b != null) seen.add(b.centerX() + "," + b.centerY());
                }
            }
        } catch (Throwable ignored) {
            // A count that throws would replace the real assertion with its own failure.
        }
        return seen.size();
    }

    /**
     * The clickable ANCESTOR of a labelled node.
     *
     * The words and the click handler frequently sit on different nodes: `<button><span>Sign
     * in</span></button>` surfaces as a clickable `View` wrapping a non-clickable `TextView`. A
     * match on the text alone is therefore the wrong element to tap, and the right one is its
     * parent. Bounded to three levels — beyond that the "ancestor" is a whole section and tapping
     * it means tapping something the test never named.
     */
    private static UiObject2 clickableAncestorOf(UiObject2 node) {
        UiObject2 cur = node;
        for (int i = 0; i < 3; i++) {
            UiObject2 parent = attr(cur, UiObject2::getParent);
            if (parent == null) return null;
            if (Boolean.TRUE.equals(attr(parent, UiObject2::isClickable))) return parent;
            cur = parent;
        }
        return null;
    }

    private static List<BySelector> selectorsFor(String name, boolean contains) {
        // Scoped to the app package so a system dialog's button can never satisfy an app assertion.
        return contains
                ? Arrays.asList(By.pkg(PKG).textContains(name), By.pkg(PKG).descContains(name))
                : Arrays.asList(By.pkg(PKG).text(name), By.pkg(PKG).desc(name));
    }

    // -------------------------------------------------------------- interaction

    /**
     * Find and tap, lifting the element clear of the bottom nav first.
     *
     * The transport and the tab bar overlap: an unscrolled tap on a control near the bottom lands
     * on a nav tab instead, which navigates away and makes the next assertion fail somewhere else
     * entirely. The iOS tier documents the same trap.
     */
    static boolean tap(List<String> names, boolean contains, long timeoutMs) {
        UiObject2 el = find(names, contains, timeoutMs);
        if (el == null) return false;
        int floor = device().getDisplayHeight() - 220;
        Rect lastGood = attr(el, UiObject2::getVisibleBounds);
        for (int i = 0; i < 6; i++) {
            Rect b = attr(el, UiObject2::getVisibleBounds);
            if (b == null) break;
            lastGood = b;
            if (b.bottom <= floor) break;
            swipeUp();
            UiObject2 again = find(names, contains, 3_000);
            if (again == null) {
                // THE SWIPE LOST IT, and giving up here was wrong twice over (2026-09-24).
                //
                // Two different things produce this and both are recoverable: the swipe scrolled
                // the control out of view, or the swipe DISMISSED the surface carrying it — an
                // open overflow menu goes away when you scroll the page behind it. The old code
                // returned false, so it reported "the landing offered no route to the downloaded
                // episodes" about a page whose own inventory listed `Play what's downloaded`.
                //
                // Put the page back, look again, and if it is there take it WITHOUT scrolling
                // further. Clicking under the bottom nav is a worse outcome than not clicking, but
                // never clicking a control that is plainly present is worse than both.
                swipeDown();
                again = find(names, contains, 3_000);
                if (again == null) break;
                el = again;
                lastGood = attr(el, UiObject2::getVisibleBounds);
                break;
            }
            el = again;
        }
        try {
            el.click();
            return true;
        } catch (Throwable t) {
            // The node went stale between the last lookup and the click. Its position did not.
            if (lastGood != null) {
                try {
                    device().click(lastGood.centerX(), lastGood.centerY());
                    return true;
                } catch (Throwable ignored) {
                    return false;
                }
            }
            return false;
        }
    }

    static boolean tap(String name, boolean contains, long timeoutMs) {
        return tap(Arrays.asList(name), contains, timeoutMs);
    }

    /**
     * Tap the match named {@code name} that sits BELOW {@code anchor}.
     *
     * For when a name is genuinely ambiguous on the page and only position separates the two — the
     * case of record is `/login`, which carries the masthead "Sign in" above the form and the
     * form's own submit "Sign in" below it. Both names are correct; the markup is not the problem.
     * Taking the first match silently clicks the masthead, which navigates to the page you are
     * already on, so the form is never submitted and nothing reports an error.
     */
    static boolean tapBelow(String name, UiObject2 anchor, long timeoutMs) {
        Rect a = attr(anchor, UiObject2::getVisibleBounds);
        if (a == null) return false;
        long deadline = System.currentTimeMillis() + timeoutMs;
        do {
            for (BySelector sel : selectorsFor(name, false)) {
                List<UiObject2> hits;
                try {
                    hits = device().findObjects(sel);
                } catch (Throwable t) {
                    continue;
                }
                for (UiObject2 o : hits) {
                    Rect b = attr(o, UiObject2::getVisibleBounds);
                    if (b == null || b.top < a.top) continue;
                    UiObject2 target = Boolean.TRUE.equals(attr(o, UiObject2::isClickable))
                            ? o
                            : clickableAncestorOf(o);
                    if (target == null) continue;
                    try {
                        target.click();
                        return true;
                    } catch (Throwable ignored) {
                        // Gone between the match and the click; the retry below re-resolves it.
                    }
                }
            }
            sleep(400);
        } while (System.currentTimeMillis() < deadline);
        return false;
    }

    /**
     * `steps` is a DURATION, not a smoothness knob: UiAutomator spends ~5ms per step, so the 12
     * this started with was a 60ms flick that the WebView treated as a fling and often ignored
     * entirely. Settings then reported "no Offline mode row" while sitting on the Settings page
     * with the row three sections below the fold, because nothing had actually scrolled
     * (2026-09-24). 40 steps is ~200ms, which is what a deliberate drag looks like.
     */
    private static final int SWIPE_STEPS = 40;

    static void swipeUp() {
        UiDevice d = device();
        d.swipe(d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.75),
                d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.30), SWIPE_STEPS);
        sleep(900);
    }

    static void swipeDown() {
        UiDevice d = device();
        d.swipe(d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.30),
                d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.75), SWIPE_STEPS);
        sleep(700);
    }

    /**
     * Scroll down until a matching element appears, or the page stops moving.
     *
     * The terminating condition is the PAGE ENDING, not a swipe count. On iOS a fixed count of 8
     * broke two suites on 2026-09-18 when surfaces legitimately grew past it, and both failures
     * read as product bugs ("the UI download did not land", "sign-in did not complete") on an app
     * that was working perfectly with the content simply further down. A count encodes how long a
     * page happens to be today; stalling does not.
     *
     * NOT FOUND rewinds to the top. Scrolling is a side effect, and a failed search that leaves the
     * app at the bottom breaks the NEXT step rather than this one — the iOS twin cost a diagnosis
     * exactly that way.
     */
    /**
     * NOTE FOR CALLERS: this searches DOWNWARD ONLY, from wherever the page currently sits.
     *
     * It does not rewind first, because several callers deliberately search from a known position.
     * The consequence is that a section ABOVE the current scroll offset can never be found, and the
     * failure reads as "the section is missing" rather than "you were already past it" — which is
     * exactly how a suite reported the Downloaded section absent on an account that had just
     * downloaded two episodes (2026-09-24). If you are searching a page you did not just arrive at
     * the top of, swipe down a few times first.
     */
    static UiObject2 scrollTo(List<String> names, boolean contains, int maxSwipes) {
        String lastSignature = "";
        int stalled = 0;
        for (int i = 0; i <= maxSwipes; i++) {
            UiObject2 hit = find(names, contains, 1_500);
            if (hit != null) return hit;
            swipeUp();
            String signature = signature();
            if (signature.equals(lastSignature)) {
                if (++stalled >= 2) break;
            } else {
                stalled = 0;
            }
            lastSignature = signature;
        }
        UiObject2 last = find(names, contains, 1_500);
        if (last != null) return last;
        for (int i = 0; i < 12; i++) swipeDown();
        return null;
    }

    static UiObject2 scrollTo(String name, boolean contains) {
        return scrollTo(Arrays.asList(name), contains, 40);
    }

    /**
     * Dismiss an entity/topic/storyline card if one is open. Idempotent, and cheap when none is.
     *
     * These cards render OVER everything, including the bottom tab bar, so one left open silently
     * swallows every later tap — and the failure then names the victim, never the culprit. Measured
     * 2026-09-26: `test03` taps a topic row, which opens the topic card, then asserted
     * `openTab("Home")` and failed with the card plainly on screen ("TOPIC | Close | systems
     * thinking | Follow — systems thinking | …").
     *
     * The iOS twin omits the same dismissal and gets away with it only because it DISCARDS
     * `openTab`'s result there, so a failed tab tap goes unnoticed. The Android assertion is the
     * honest one, which is why the fix belongs here rather than in a weaker assertion.
     *
     * "Back" as well as "Close": an entity card labels its dismiss control Back, not Close, when
     * `dismissAtRoot` is false, and a Close-only search finds nothing.
     */
    static void dismissCards() {
        for (int i = 0; i < 3; i++) {
            if (find(Arrays.asList("Close", "Back"), false, 1_200) == null) return;
            if (!tap(Arrays.asList("Close", "Back"), false, 1_200)) return;
            sleep(1_200);
        }
    }

    /**
     * Name AND vertical position of the nodes BELOW the sticky chrome.
     *
     * Position as well as name, because names alone would assume the visible set changes as you
     * scroll — true of these surfaces today but written down nowhere, so it would rot silently.
     *
     * Below the chrome, because the masthead and the bottom nav are FIXED. Taking the first twelve
     * nodes in tree order takes the masthead every time, so the signature never changed, every page
     * read as stalled after two swipes, and `scrollTo` gave up long before reaching anything below
     * the fold. That is how Settings reported "no Offline mode row" while sitting on the Settings
     * page with the row three sections down (2026-09-24). The iOS twin's comment warns about a
     * sticky header doing exactly this; I ported the warning and then wrote the bug it describes.
     */
    private static String signature() {
        StringBuilder sb = new StringBuilder();
        try {
            int top = (int) (device().getDisplayHeight() * 0.12);
            int bottom = (int) (device().getDisplayHeight() * 0.88);
            int n = 0;
            for (UiObject2 o : device().findObjects(By.pkg(PKG))) {
                if (n >= 12) break;
                Rect b = attr(o, UiObject2::getVisibleBounds);
                if (b == null || b.centerY() < top || b.centerY() > bottom) continue;
                String name = nameOf(o);
                if (name.isEmpty()) continue;
                sb.append(name).append('@').append(b.top).append('|');
                n++;
            }
        } catch (Throwable ignored) {
            // A stale signature just means "changed", which costs one extra swipe.
        }
        return sb.toString();
    }

    // ------------------------------------------------------------- navigation

    /** Bottom tab bar. */
    static boolean openTab(String name) {
        return tap(name, false, 20_000);
    }

    /**
     * Masthead avatar → Profile.
     *
     * TOP first: the control is in the masthead, so it scrolls away with the page, and a node above
     * the viewport is still in the tree with bounds that no tap can reach. Whatever the previous
     * step left on screen, the header is reachable from the top.
     *
     * ## Why this takes the identity, and the iOS twin does not
     *
     * The link's accessible name is `auth.user?.name || t('profile.title')` (App.vue) — the ACCOUNT
     * NAME when it has resolved, "Your profile" only when it has not. The iOS helper hard-codes
     * `["Your profile", "simtest", "uitest"]`, which works there only because its suites use the
     * seeded accounts. With per-suite identities that list matches nothing: the run of 2026-09-24
     * signed in perfectly as `harnesssmoketests`, showed that exact name in the masthead, and then
     * reported "sign-in did not complete" because the verifier could not name what it was looking
     * at. The same latent flaw is in the iOS file, masked by its account choices.
     */
    static boolean openProfile(List<String> labels) {
        for (int i = 0; i < 12; i++) swipeDown();
        if (tap(labels, false, 15_000)) return true;
        // POSITIONAL fallback: when signed in as an account this caller cannot name — the
        // ensure-signed-in path, which has to reach Profile to sign the OTHER account out before it
        // can sign this one in. The avatar is the rightmost control in the masthead band whatever
        // it is called, and position does not depend on knowing the name.
        return tapRightmostInMasthead();
    }

    /** The seeded accounts plus the generic label — for callers with no identity of their own. */
    static boolean openProfile() {
        return openProfile(Arrays.asList("Your profile", "simtest", "uitest"));
    }

    private static boolean tapRightmostInMasthead() {
        UiObject2 best = null;
        int bestLeft = -1;
        try {
            int band = (int) (device().getDisplayHeight() * 0.12);
            for (UiObject2 o : device().findObjects(By.pkg(PKG).clickable(true))) {
                Rect b = attr(o, UiObject2::getVisibleBounds);
                if (b == null || b.top > band || b.width() > device().getDisplayWidth() / 3) continue;
                if (b.left > bestLeft) {
                    bestLeft = b.left;
                    best = o;
                }
            }
        } catch (Throwable ignored) {
            return false;
        }
        if (best == null) return false;
        try {
            best.click();
            sleep(1_500);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Profile → Settings, verified by ARRIVING rather than by the tap returning true.
     *
     * A tap that lands on a label with no clickable ancestor reports success and navigates
     * nowhere. `setOfflineMode` then hunted "Offline mode" on the Profile page and reported the row
     * missing — a failure that names the wrong screen, and the second time in this port that a
     * successful-looking tap did nothing (2026-09-24). So the postcondition is the presence of a
     * control that exists ONLY on Settings, and the navigation is retried before it is believed.
     */
    static boolean openSettings(List<String> profileLabels) {
        for (int attempt = 1; attempt <= 2; attempt++) {
            boolean profile = openProfile(profileLabels);
            sleep(2_000);
            boolean onProfile = find(Arrays.asList("Sign out"), false, 3_000) != null
                    || find(Arrays.asList("Settings"), true, 3_000) != null;
            boolean settingsTap = tap("Settings", true, 15_000);
            sleep(2_000);
            if (scrollTo(Arrays.asList(OFFLINE_ROW), false, 12) != null) return true;
            System.out.println("=====SETTINGS_NAV attempt " + attempt
                    + " profileTap=" + profile + " reachedProfile=" + onProfile
                    + " settingsTap=" + settingsTap
                    + " :: " + labelledInventory(16) + "=====");
        }
        return false;
    }

    /**
     * Drive Settings → "Offline mode" to an ABSOLUTE state; a no-op when it already matches.
     *
     * Absolute, not a toggle, because the switch is device-local: it survives a relaunch AND an
     * account change. Left on, it breaks every later suite's network assertions for a reason that
     * has nothing to do with them — the cross-suite leak of 2026-09-16 on iOS, where one suite
     * leaving forced-offline ON produced three failures that each named the victim, never the
     * culprit.
     *
     * The state persists to `localStorage`, which the host cannot reach, so driving the UI is the
     * only way to set it.
     *
     * ## Read the BANNER, never the checkbox
     *
     * Chromium reports the input as `checkable=false`, so `isChecked()` is pinned to false and
     * cannot describe the state. Believing it cost a long detour: three interactions each reported
     * "nothing changed" while every one of them HAD flipped the switch — and the run that finally
     * proved it did so by leaving forced-offline ON, which wedged the next run at sign-in with the
     * banner "Offline mode is on — showing saved" plainly on screen. The tier's own first
     * cross-suite leak, caused by the helper written to prevent them.
     */
    /**
     * The Settings row's label, matched EXACTLY.
     *
     * A substring match on "Offline mode" also matches the BANNER — "Offline mode is on — showing
     * saved" — which is present precisely when the switch is on. So the OFF direction walked up
     * from the banner, found the nearest checkbox, and drove the VOICE INPUT setting instead, three
     * times, reporting success each round (2026-09-24). The evidence was one line:
     * `rowName='Offline mode is on — showing saved'`.
     *
     * The lesson generalises past this row: a `contains` match is a guess that no other copy on the
     * page shares the words, and a status message about a setting almost always shares them.
     */
    private static final String OFFLINE_ROW = "Offline mode";

    static boolean setOfflineMode(boolean wanted, List<String> profileLabels) {
        for (int round = 1; round <= 3; round++) {
            boolean observed = isForcedOffline();
            System.out.println("=====OFFLINE_SET round " + round + " observed=" + observed
                    + " wanted=" + wanted + "=====");
            if (observed == wanted) return true;

            if (!openSettings(profileLabels)) {
                System.out.println("=====OFFLINE_SET settings unreachable :: "
                        + labelledInventory(12) + "=====");
                return false;
            }
            UiObject2 row = scrollTo(Arrays.asList(OFFLINE_ROW), false, 20);
            if (row == null) {
                System.out.println("=====OFFLINE_SET no 'Offline mode' row :: "
                        + labelledInventory(16) + "=====");
                return false;
            }
            UiObject2 box = nearestCheckable(row);
            if (box == null) {
                System.out.println("=====OFFLINE_SET nothing checkable for the row :: "
                        + labelledInventory(16) + "=====");
                return false;
            }

            // EXACTLY ONE interaction per round, then go and look.
            //
            // The first version fired three interactions back to back because it could not observe
            // the result of any of them from Settings. Each one that LANDED flipped the switch, so
            // an even number returned it to where it started: driving it ON worked and driving it
            // OFF silently did not — the worst kind of half-working. A control whose state you
            // cannot read must be driven one step at a time, verifying after each.
            //
            // The ACCESSIBILITY click, not a coordinate tap. Measured one interaction at a time
            // (2026-09-24): the a11y click flipped it every time; coordinate taps on the same
            // element flipped nothing. The reason is in the numbers — the checkbox reported bounds
            // of y=133, then y=1960, then y=249 on three consecutive visits, because where a web
            // page sits when you arrive is not stable. A coordinate is a guess about scroll
            // position; an accessibility action addresses the element itself.
            System.out.println("=====OFFLINE_SET clicking box bounds=" + attr(box, UiObject2::getVisibleBounds)
                    + " rowBounds=" + attr(row, UiObject2::getVisibleBounds)
                    + " rowName='" + nameOf(row) + "'"
                    + " rowCls=" + attr(row, UiObject2::getClassName) + "=====");
            try {
                box.click();
            } catch (Throwable t) {
                System.out.println("=====OFFLINE_SET click threw " + t + "=====");
                return false;
            }
            sleep(1_500);
            System.out.println("=====OFFLINE_SET round " + round + " clicked the box=====");
        }
        boolean now = isForcedOffline();
        if (now != wanted) {
            System.out.println("=====OFFLINE_SET after 3 rounds the app reports offline=" + now
                    + ", wanted " + wanted + " :: " + labelledInventory(16) + "=====");
        }
        return now == wanted;
    }

    /**
     * Is the app ACTUALLY in forced-offline mode?
     *
     * Asked on Home, because the banner renders on content surfaces and not on Settings — so
     * checking where the switch lives always answers "no". Returning to Home also leaves the app
     * somewhere every caller can continue from.
     */
    private static boolean isForcedOffline() {
        openTab("Home");
        sleep(2_500);
        return forcedOfflineBannerShowing();
    }

    /**
     * The checkable node for a labelled row.
     *
     * The LABEL and the CHECKBOX are separate nodes — `<label>Offline mode</label><input
     * type=checkbox>` — and only the input reports `checked`. Reading the label's state returns
     * false forever, so the toggle looks stuck off and the test flips it every time.
     */
    /**
     * Is the app in forced-offline mode — read from the BANNER, not from the checkbox.
     *
     * Chromium reports the `<input type=checkbox>` as `checkable=false`, so `isChecked()` is pinned
     * to false and can never describe the real state. Three interactions therefore all looked like
     * they "did nothing" while every one of them had in fact toggled the switch — and the run that
     * proved it did so by leaving forced-offline ON and wedging the NEXT run, which could no longer
     * sign in. (WebKit exposes the same input's state as value "0"/"1", which is why the iOS twin
     * can read the control directly. This is a real platform difference, not a bug in either app.)
     *
     * The banner is the better observable regardless: `i18n en.json :: offlineForced` renders only
     * when the app has actually entered forced-offline, so it proves the STATE rather than the
     * appearance of a tick.
     */
    static boolean forcedOfflineBannerShowing() {
        return find(Arrays.asList("Offline mode is on"), true, 4_000) != null;
    }

    /**
     * The checkbox belonging to {@code labelled}, by CONTAINMENT rather than by proximity.
     *
     * ## The bug this replaces, which is the one this file warned about
     *
     * Settings has TWO checkboxes — "Voice input for notes" and "Offline mode" — and the previous
     * version short-circuited to `checkables.get(0)` whenever only one was currently visible,
     * skipping its own distance guard. So when the Offline row was scrolled out of view the helper
     * silently drove the VOICE switch instead, three times, while reporting "clicked the box" each
     * time and the app stayed offline (2026-09-24). Driving the wrong control is the worst outcome
     * available here — it corrupts a setting nobody asked about AND reports success — and this
     * file's own comment said so two revisions before it happened.
     *
     * Containment has no threshold to get wrong: the markup is
     * `<label><span>Offline mode</span><input type=checkbox></label>`, so the input is inside an
     * ancestor of the label text and inside NO ancestor of any other row. Walking up from the text
     * and taking the first ancestor that contains a checkbox cannot pick a different row's input.
     */
    private static UiObject2 nearestCheckable(UiObject2 labelled) {
        UiObject2 cur = labelled;
        for (int up = 0; up < 4; up++) {
            UiObject2 parent = attr(cur, UiObject2::getParent);
            if (parent == null) break;
            for (BySelector sel : Arrays.asList(
                    By.clazz("android.widget.CheckBox"),
                    By.clazz("android.widget.Switch"),
                    By.checkable(true))) {
                List<UiObject2> hits = attr(parent, p -> p.findObjects(sel));
                if (hits != null && !hits.isEmpty()) return hits.get(0);
            }
            cur = parent;
        }
        System.out.println("=====CHECKBOX no checkable inside any of the 4 ancestors of '"
                + nameOf(labelled) + "'=====");
        return null;
    }

    // ----------------------------------------------------------------- plumbing

    /**
     * Every accessor re-resolves the node, and a node that has gone stale throws.
     *
     * Package-visible so suites that enumerate for themselves (`AppJourneyTests.tapTopmost`) share
     * this null-and-stale handling instead of re-deriving it — the alternative is a second, subtly
     * different accessor, which is how the two ports drifted in the first place.
     */
    static <T> T attr(UiObject2 o, Getter<T> getter) {
        if (o == null) return null;
        try {
            return getter.get(o);
        } catch (Throwable t) {
            return null;
        }
    }

    interface Getter<T> {
        T get(UiObject2 o) throws Throwable;
    }

    static void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Save a screenshot to {@code /sdcard/lp-shots/<name>.png}.
     *
     * The iOS suite has had {@code Journey.shot} from the start; Android never did, and the gap
     * cost real time on 2026-09-26. An assertion reading "no dictation control" reports THE ABSENCE
     * OF AN ACCESSIBILITY NODE, which cannot distinguish "the control is not rendered" from "the
     * control is drawn but unreadable" — and on Android, where the tree holds only on-screen nodes
     * and drops labels whose subtree has no text, that difference IS the diagnosis. A picture
     * separates the two in one glance; an inventory dump never can.
     *
     * Best-effort: failing to write a screenshot must never mask the assertion that asked for it.
     */
    static void shot(String name) {
        try {
            // The APP'S external files dir, not /sdcard directly: scoped storage refuses the
            // latter, and `mkdirs()` reports that by returning false rather than throwing — so the
            // first version failed with ENOENT inside UiDevice and produced no screenshot at all.
            // Pull with:
            //   adb pull /sdcard/Android/data/app.closelistening.player/files/lp-shots/<name>.png
            java.io.File base = InstrumentationRegistry.getInstrumentation()
                    .getTargetContext().getExternalFilesDir(null);
            java.io.File dir = new java.io.File(base, "lp-shots");
            if (!dir.exists() && !dir.mkdirs()) {
                System.out.println("=====SHOT could not create " + dir.getAbsolutePath() + "=====");
                return;
            }
            java.io.File out = new java.io.File(dir, name + ".png");
            boolean ok = device().takeScreenshot(out);
            System.out.println("=====SHOT " + (ok ? "saved " : "FAILED ") + out.getAbsolutePath()
                    + "=====");
        } catch (Throwable t) {
            System.out.println("=====SHOT threw " + t + "=====");
        }
    }
}
