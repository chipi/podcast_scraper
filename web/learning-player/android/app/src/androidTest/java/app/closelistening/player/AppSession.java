package app.closelistening.player;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;
import androidx.test.uiautomator.Until;

import java.util.Arrays;

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
     * Signed in AT ALL.
     *
     * "Sign out" lives on the PROFILE page and nowhere else, while the app cold-boots to Home. The
     * iOS twin asked a Home screen whether it had a Profile-only control and got "no" every time,
     * whatever the session was — reported as "the app fell back to signed-out" on a device whose
     * token was perfectly valid. Go to Profile first, then read the answer.
     */
    /**
     * Is there ANY session, without needing to know whose?
     *
     * Read from the masthead "Sign in" link's ABSENCE rather than by hunting the avatar. App.vue
     * states this as the design: "Signed-in state is still legible from the masthead: the Sign in
     * link is absent." That signal does not depend on the account's name, where every avatar-based
     * check does — and with per-suite identities the name is exactly what a generic caller cannot
     * know.
     */
    static boolean hasAnySession() {
        return Journey.find(Arrays.asList("Sign in"), false, 8_000) == null;
    }

    static boolean isSignedIn() {
        Journey.openProfile();
        // SCROLL to it: "Sign out" is deliberately the last control on Profile (#1962 — "quiet,
        // last, least weight"), so on any account with content it is below the fold. Asking whether
        // it is on SCREEN is not the question being asked.
        if (Journey.scrollTo("Sign out", false) == null) return false;
        // The painted session is not the answer — the revalidation that follows it is.
        Journey.sleep(6_000);
        return Journey.scrollTo("Sign out", false) != null;
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
        // By the identity label WHEN IT IS THERE, else the generic one. The masthead avatar is
        // labelled `auth.user?.name || 'Your profile'`, so an account whose name has not resolved
        // is labelled generically — and reaching for the identity alone reported SIGNED OUT about
        // an app that was demonstrably signed in (the iOS twin, 2026-09-24).
        if (!Journey.openProfile(Arrays.asList(identity, "Your profile"))) return false;
        // The RIGHT account, checked FIRST — before the scroll below moves it off screen. Profile
        // prints the name and the email, and a dev identity appears in at least one of them.
        if (Journey.find(Arrays.asList(identity), true, 10_000) == null) return false;
        if (Journey.scrollTo("Sign out", false) == null) return false;
        Journey.sleep(6_000);
        return Journey.scrollTo("Sign out", false) != null;
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
            entry.click();
        } catch (Throwable t) {
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
        if (!Journey.tapBelow("Sign in", input, 10_000)) return false;

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

    private static UiObject2 waitForField(long timeoutMs) {
        return Journey.device().wait(
                Until.findObject(By.pkg(Journey.PKG).clazz("android.widget.EditText")), timeoutMs);
    }

    /**
     * Sign out and stay out — the precondition for `/offline`, which only renders when signed out.
     */
    static boolean signOut() {
        if (!hasAnySession()) return true; // already out; the caller's precondition holds
        Journey.openProfile();
        UiObject2 out = Journey.scrollTo("Sign out", false);
        if (out == null) return false;
        try {
            out.click();
        } catch (Throwable t) {
            return false;
        }
        Journey.sleep(3_000);
        return !hasAnySession();
    }

    /** Leave the app signed in as {@code identity}, whatever it was signed in as before. */
    static boolean ensureSignedIn(String identity) {
        if (isSignedIn(identity)) return true;
        // Signed in as SOMEONE ELSE: the dev picker is unreachable while a session exists (there is
        // no "Sign in" link on a signed-in app), so without this the suite would silently keep the
        // previous suite's account — the exact collision per-suite identities exist to prevent.
        //
        // `hasAnySession`, not `isSignedIn()`: the latter hunts the avatar by name, and the name is
        // the other account's, which is precisely what this branch does not know.
        if (hasAnySession() && !signOut()) return false;
        return signIn(identity);
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

    /** Cold start — relaunch so what is on disk is re-read, not just re-activated. */
    static void relaunch() {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Journey.device().pressHome();
        Intent launch = ctx.getPackageManager().getLaunchIntentForPackage(Journey.PKG);
        if (launch == null) throw new AssertionError("the app under test is not installed");
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        ctx.startActivity(launch);
        Journey.device().wait(Until.hasObject(By.pkg(Journey.PKG).depth(0)), 30_000);
        // Boot paints the device snapshot and then revalidates; assert after that lands.
        Journey.sleep(7_000);
    }
}
