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
     * First element matching ANY of {@code names}, across both name fields.
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
            for (String name : names) {
                for (BySelector sel : selectorsFor(name, contains)) {
                    List<UiObject2> hits;
                    try {
                        hits = device().findObjects(sel);
                    } catch (Throwable t) {
                        continue;
                    }
                    for (UiObject2 o : hits) {
                        if (Boolean.TRUE.equals(attr(o, UiObject2::isClickable))) return o;
                        UiObject2 clickable = clickableAncestorOf(o);
                        if (clickable != null) return clickable;
                        if (fallback == null) fallback = o;
                    }
                }
            }
            if (fallback != null) return fallback;
            sleep(400);
        } while (System.currentTimeMillis() < deadline);
        return null;
    }

    static UiObject2 find(String name, boolean contains, long timeoutMs) {
        return find(Arrays.asList(name), contains, timeoutMs);
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
        for (int i = 0; i < 6; i++) {
            Rect b = attr(el, UiObject2::getVisibleBounds);
            if (b == null || b.bottom <= floor) break;
            swipeUp();
            el = find(names, contains, 3_000);
            if (el == null) return false;
        }
        try {
            el.click();
            return true;
        } catch (Throwable t) {
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

    static void swipeUp() {
        UiDevice d = device();
        d.swipe(d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.72),
                d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.28), 12);
        sleep(700);
    }

    static void swipeDown() {
        UiDevice d = device();
        d.swipe(d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.28),
                d.getDisplayWidth() / 2, (int) (d.getDisplayHeight() * 0.72), 12);
        sleep(500);
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
     * Name AND vertical position of the first handful of nodes.
     *
     * Names alone would assume the visible set changes as you scroll. That is true of these
     * surfaces today but written down nowhere, so a sticky header would make every page look
     * stalled after two swipes. Positions move whenever the page does, which is the thing actually
     * being detected.
     */
    private static String signature() {
        StringBuilder sb = new StringBuilder();
        try {
            int n = 0;
            for (UiObject2 o : device().findObjects(By.pkg(PKG))) {
                if (n++ >= 12) break;
                Rect b = attr(o, UiObject2::getVisibleBounds);
                sb.append(nameOf(o)).append('@').append(b == null ? -1 : b.top).append('|');
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

    /** Profile → Settings. */
    static boolean openSettings(List<String> profileLabels) {
        if (!openProfile(profileLabels)) return false;
        sleep(2_000);
        if (!tap("Settings", true, 20_000)) return false;
        sleep(2_000);
        return true;
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
     */
    static boolean setOfflineMode(boolean wanted, List<String> profileLabels) {
        if (!openSettings(profileLabels)) {
            System.out.println("=====OFFLINE_SET settings unreachable :: " + labelledInventory(12) + "=====");
            return false;
        }
        UiObject2 control = scrollTo(Arrays.asList("Offline mode"), true, 20);
        if (control == null) {
            System.out.println("=====OFFLINE_SET no 'Offline mode' row :: " + labelledInventory(16) + "=====");
            return false;
        }
        UiObject2 target = Boolean.TRUE.equals(attr(control, UiObject2::isCheckable))
                ? control
                : nearestCheckable(control);
        if (target == null) {
            System.out.println("=====OFFLINE_SET row found but nothing checkable near it; row cls="
                    + attr(control, UiObject2::getClassName) + " :: " + labelledInventory(16) + "=====");
            return false;
        }
        Boolean checked = attr(target, UiObject2::isChecked);
        if (checked != null && checked == wanted) return true;
        try {
            target.click();
        } catch (Throwable t) {
            return false;
        }
        sleep(1_500);
        UiObject2 after = scrollTo(Arrays.asList("Offline mode"), true, 6);
        if (after == null) return false;
        UiObject2 verify = Boolean.TRUE.equals(attr(after, UiObject2::isCheckable))
                ? after
                : nearestCheckable(after);
        Boolean now = verify == null ? null : attr(verify, UiObject2::isChecked);
        if (now == null || now != wanted) {
            System.out.println("=====OFFLINE_SET tapped but state is " + now + ", wanted " + wanted
                    + " :: " + labelledInventory(16) + "=====");
            return false;
        }
        return true;
    }

    /**
     * The checkable node for a labelled row.
     *
     * The LABEL and the CHECKBOX are separate nodes — `<label>Offline mode</label><input
     * type=checkbox>` — and only the input reports `checked`. Reading the label's state returns
     * false forever, so the toggle looks stuck off and the test flips it every time.
     */
    private static UiObject2 nearestCheckable(UiObject2 labelled) {
        Rect anchor = attr(labelled, UiObject2::getVisibleBounds);
        if (anchor == null) return null;
        // By POSITION, not by ancestry. Walking up from the label and searching its subtree found
        // nothing: the checkbox is a SIBLING of the text, not a descendant of any of its first
        // three ancestors, so the search kept missing a control that the inventory printed on the
        // very next line (`<UNLABELLED>[CheckBox]`, measured 2026-09-24). Row layout is what ties a
        // label to its input on screen, and it is what ties them here.
        UiObject2 best = null;
        int bestDistance = Integer.MAX_VALUE;
        try {
            for (UiObject2 c : device().findObjects(By.pkg(PKG).checkable(true))) {
                Rect b = attr(c, UiObject2::getVisibleBounds);
                if (b == null) continue;
                int distance = Math.abs(b.centerY() - anchor.centerY());
                // A row is one line tall; anything further away belongs to a different setting, and
                // silently flipping the WRONG switch is the worst outcome available here.
                if (distance < bestDistance && distance <= anchor.height() * 3) {
                    bestDistance = distance;
                    best = c;
                }
            }
        } catch (Throwable ignored) {
            return null;
        }
        return best;
    }

    // ----------------------------------------------------------------- plumbing

    /** Every accessor re-resolves the node, and a node that has gone stale throws. */
    private static <T> T attr(UiObject2 o, Getter<T> getter) {
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
}
