package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Base64;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

/**
 * Email magic-link sign-in, end to end on Android (#2272) — the port of the iOS
 * `MagicLinkJourneyTests.swift`: create an account, land on its profile, sign out, sign back in
 * from a CLOSED app, land home.
 *
 * THREE PHASES, run one at a time, because the link travels through a real mailbox in between. The
 * operator (or a script) runs the delivery worker after M1 and M2, takes the link from the outbox,
 * and passes it to the next phase BASE64-encoded (a raw link carries `&`, which the device shell
 * would split):
 *
 *   -e lp.magic.email test@closelistening.app   (M1, M2)
 *   -e lp.magic.link_b64 <base64 of the link>   (M2)
 *
 *   M1  signed-out app → "Email me a sign-in link" → address → "Check your email"
 *   M2  open link 1 → NEW account lands on Profile → sign out → request link 2
 *   M3  is NOT here — see the note at the end of the class and android/scripts/magic-link-cold-launch.sh:
 *       with the app STOPPED, link 2 launches it → RETURNING account signed in, on Home
 *
 * The link is opened the way Mail opens it: an ACTION_VIEW on the https/http URL, so the BROWSER
 * follows the verify redirect to `closelistening://auth#token=…` and Android hands that to the app
 * through the manifest's intent filter. That browser hop is the part iOS cannot prove for Android.
 *
 * Run through `make test-app-android-magic-link`; a phase without its arguments FAILS.
 */
@RunWith(AndroidJUnit4.class)
public class MagicLinkJourneyTests extends UITestCase {

    /** No other page shows all three together. */
    private static final List<String> PROFILE_TABS = Arrays.asList("Account", "Topics", "Stats");

    private static String arg(String name) {
        Bundle args = InstrumentationRegistry.getArguments();
        String v = args == null ? null : args.getString(name);
        return v == null || v.trim().isEmpty() ? null : v.trim();
    }

    private static String email() {
        return arg("lp.magic.email");
    }

    private static String link() {
        String b64 = arg("lp.magic.link_b64");
        if (b64 == null) return null;
        return new String(Base64.decode(b64, Base64.DEFAULT), StandardCharsets.UTF_8).trim();
    }

    private static boolean onProfile(long timeoutMs) {
        for (String tab : PROFILE_TABS) {
            if (Journey.find(Arrays.asList(tab), false, timeoutMs) == null) return false;
        }
        return true;
    }

    /** From a signed-out app: reach /login, open the email form, send a link to {@code address}. */
    private static void requestLink(String address, String tag) {
        assertNotNull("no signed-out 'Sign in' on screen :: " + Journey.labelledInventory(12),
                Journey.find(Arrays.asList("Sign in"), false, 20_000));
        assertTrue("could not tap 'Sign in'", Journey.tap("Sign in", false, 10_000));
        assertTrue("no magic-link button on /login :: " + Journey.labelledInventory(12),
                Journey.tap("Email me a sign-in link", false, 15_000));
        Journey.shot(tag + "-email-form");

        // /login on a dev build has TWO text fields: the dev picker's "or a custom name…" and this
        // one. Take the field whose hint/text is the email placeholder; failing that, the LAST field
        // on the page, since the email form is rendered below the dev picker.
        UiObject2 field = null;
        long deadline = System.currentTimeMillis() + 10_000;
        while (field == null && System.currentTimeMillis() < deadline) {
            List<UiObject2> fields = Journey.device().findObjects(
                    By.pkg(Journey.PKG).clazz("android.widget.EditText"));
            for (UiObject2 f : fields) {
                String t = String.valueOf(f.getText()) + " " + String.valueOf(f.getContentDescription());
                if (t.contains("you@example.com")) field = f;
            }
            if (field == null && fields.size() >= 2) field = fields.get(fields.size() - 1);
            if (field == null) Journey.sleep(400);
        }
        assertNotNull("the email field did not appear :: " + Journey.labelledInventory(16), field);
        field.click();
        field.setText(address);
        assertTrue("no 'Send link' button", Journey.tap("Send link", false, 10_000));
        boolean sent = Journey.find(Arrays.asList("Check your email"), false, 15_000) != null;
        Journey.shot(tag + "-check-your-email");
        assertTrue("the app never confirmed the link was sent :: " + Journey.labelledInventory(12), sent);
    }

    /**
     * Open {@code link} as Mail would and get the app to the foreground. Taps through a browser's
     * first-run screens and its "open in app?" prompt when they appear — whichever of these this
     * emulator's browser shows is logged, because the browser hop is what this phase exists to see.
     */
    private static void openLink(String link) {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Intent view = new Intent(Intent.ACTION_VIEW, Uri.parse(link));
        view.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        ctx.startActivity(view);
        Journey.mark("=====MAGIC opened link in the browser=====");

        List<String> taps = Arrays.asList(
                // Chrome first run, on a fresh emulator image.
                "Use without an account", "Accept & continue", "No thanks", "No, thanks",
                "Continue without an account",
                // Handing the custom scheme to the app.
                "Open", "Continue", "Open app", "Close Listening", "Always", "Just once");
        long deadline = System.currentTimeMillis() + 45_000;
        while (System.currentTimeMillis() < deadline) {
            if (Journey.device().hasObject(By.pkg(Journey.PKG).depth(0))) {
                Journey.mark("=====MAGIC app is in the foreground=====");
                return;
            }
            for (String label : taps) {
                UiObject2 b = Journey.device().findObject(By.text(label));
                if (b != null && !Journey.PKG.equals(b.getApplicationPackage())) {
                    Journey.mark("=====MAGIC tapping '" + label + "' in "
                            + b.getApplicationPackage() + "=====");
                    b.click();
                    break;
                }
            }
            Journey.sleep(500);
        }
        Journey.shot("magic-link-stuck");
        fail("the app never came to the foreground after opening the link; foreground = "
                + Journey.foregroundWindow() + " :: " + Journey.labelledInventory(16));
    }

    @Test
    public void testM1RequestALinkForANewAccount() {
        assertNotNull("missing -e lp.magic.email — run via make test-app-android-magic-link", email());
        AppSession.relaunch();
        assertTrue("could not reach a signed-out state", AppSession.signOut());
        requestLink(email(), "M1");
    }

    @Test
    public void testM2NewAccountLandsOnProfileThenSignsOut() {
        assertNotNull("missing -e lp.magic.email — run via make test-app-android-magic-link", email());
        assertNotNull("missing -e lp.magic.link_b64 — run via make test-app-android-magic-link", link());
        AppSession.relaunch();
        openLink(link());
        boolean landed = onProfile(20_000);
        Journey.shot("M2-landed");
        assertTrue("a NEW account must land on Profile (new=1), not Home :: "
                + Journey.labelledInventory(16), landed);
        assertTrue("on Profile but not signed in", AppSession.isSignedIn());
        assertTrue("could not sign out", AppSession.signOut());
        Journey.shot("M2-signed-out");
        requestLink(email(), "M2");
    }

    // NO M3 HERE, deliberately. M3 is "the link LAUNCHES the app", and instrumentation cannot reach
    // it: the test runs inside the app's own process, so force-stopping the app killed the test
    // itself (2026-10-03: `am instrument` printed nothing at all), and `am instrument` starts the
    // app anyway. It is driven from the shell instead: `android/scripts/magic-link-cold-launch.sh`.
}
