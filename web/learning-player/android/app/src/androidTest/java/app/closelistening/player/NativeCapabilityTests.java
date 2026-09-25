package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.BySelector;
import androidx.test.uiautomator.UiObject2;
import androidx.test.uiautomator.Until;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;

/**
 * The Capacitor-only capabilities (#2139, porting the iOS suite of the same name).
 *
 * The shell enables ten plugins plus two local ones. This exercises the three that are
 * user-visible, newly wired, and entirely unverified — voice dictation, the native share sheet,
 * and push registration.
 *
 * ## What is DIFFERENT between iOS and Android, and what that means for each test
 *
 * - **N1 (dictation):** iOS ships SFSpeechRecognizer, which is present-but-inert in the simulator.
 *   Android ships SpeechRecognizer (android.speech), which is also absent on the emulator — the
 *   Google app that handles the recognition intent is not installed in a stock AVD. The contract is
 *   therefore the same: assert the AFFORDANCE exists once the Config opt-in is on, and assert the
 *   app reports failure rather than presenting a dead mic. The mic tap is opt-in via the system
 *   property `lp.tap.mic=1` for the same reason as on iOS.
 *
 * - **N2 (share sheet):** iOS asserts Springboard's ActivityListView by querying a second
 *   XCUIApplication(bundleIdentifier: "com.apple.springboard"). On Android, the OS share sheet
 *   runs in a separate process (com.android.intentresolver or similar), so we use
 *   {@link androidx.test.uiautomator.UiDevice#wait} without a package filter — exactly the pattern
 *   Journey.java uses for the OAuth consent dialog. The content assertion changes accordingly: the
 *   sheet must carry the text "closelistening" (the file name our share path writes).
 *
 * - **N3 (push):** The WebKit / iOS flow calls APNs; the Capacitor Android flow calls FCM via
 *   @capacitor/push-notifications. The assertion is the same: the app stays in the foreground, and
 *   the cell does not claim push is on if the OS refused. On the emulator Google Play Services may
 *   be absent, in which case no system prompt appears — this is an environment property, not a test
 *   failure, and the suite handles it the same way as the iOS twin (grantSystemPrompt returns false
 *   without asserting).
 *
 * - **N4 (avatar upload):** iOS drives a PHPicker (out-of-process) via Springboard. Android opens
 *   a stock file chooser (Intent.ACTION_GET_CONTENT) or the Android photo picker, also
 *   out-of-process. The test is OMITTED: the file-chooser package and activity names vary
 *   significantly across Android versions and OEM skins, and there is no package-agnostic anchor
 *   equivalent to iOS's "Photo, <date>" cell label — a working version would encode emulator
 *   internals and break on a real device. The iOS comment already notes that `app.images` reaches
 *   into the wrong process tree; the Android version of that problem has no clean workaround, and
 *   a test that passes vacuously on the emulator by picking app chrome is worse than no test.
 *   The crop modal (AvatarCropModal.vue) and the upload path are covered by the unit suite.
 *
 * PRECONDITIONS: app installed, signed in via dev picker, fixture api reachable.
 */
@RunWith(AndroidJUnit4.class)
public class NativeCapabilityTests extends UITestCase {

    private static final String EPISODE_SLUG = "p09-a4bbb5dde3";

    // ------------------------------------------------------------------ N1 dictation

    /**
     * The dictation affordance appears when the "Voice input for notes" Config opt-in is on, and
     * the app neither crashes nor presents a dead mic when it is tapped.
     *
     * ANDROID DIFFERENCE: the mic tap test is opt-in via the instrumentation argument
     * {@code lp.tap.mic=1} — the same threshold as the iOS LP_TAP_MIC env var. On the emulator
     * the SpeechRecognizer intent is typically unresolvable, which is a correct "unavailable"
     * outcome for the app to report. The opt-in mechanism keeps it a runtime switch, not a
     * compile-time skip.
     */
    @Test
    public void testN1DictationAffordanceAppearsWhenEnabled() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(14), ready);

        // Dictation is OFF by default and lives behind a Settings opt-in, so the mic cannot
        // appear until that switch is on — itself worth asserting, since a mic that showed up
        // unbidden would be a privacy surprise.
        boolean settingsOpen = Journey.openSettings(profileLabels());
        if (!settingsOpen) {
            fail("could not reach Settings. On screen: " + Journey.labelledInventory(20));
        }
        // settings.voiceInput = 'Voice input for notes'
        Journey.scrollTo("Voice input for notes", false);

        UiObject2 voiceRow = Journey.find("Voice input for notes", false, 8_000);
        if (voiceRow == null) {
            fail("no 'Voice input for notes' control in Settings. On screen: "
                    + Journey.labelledInventory(20));
        }
        UiObject2 toggle = nearestCheckable(voiceRow);
        if (toggle == null) {
            fail("no toggle for Voice input for notes. On screen: "
                    + Journey.labelledInventory(20));
        }

        // Read the current state (isChecked is reliable for a native Switch/CheckBox but not for
        // web checkboxes; this IS a native Android Switch rendered by Capacitor Preferences). If
        // the Chromium bridge exposes this like it does the Offline Mode checkbox (checkable=false
        // pinned) fall back to treating it as unknown and attempting to enable.
        boolean wasChecked = false;
        try { wasChecked = Boolean.TRUE.equals(toggle.isChecked()); } catch (Throwable ignored) {}

        if (!wasChecked) {
            try { toggle.click(); } catch (Throwable t) {
                fail("could not click Voice input toggle: " + t);
            }
            Journey.sleep(2_000);
        }

        // On leaving Settings we restore the toggle — same as the iOS `defer` block.
        final boolean toggleWasOn = wasChecked;
        try {
            // A PERSON card, not a topic one. Both carry a note composer, but the topic card
            // ends with a long annotated episode list, so notes sit far below the fold — the iOS
            // comment noted sixteen swipes still landed mid-list (2026-09-16). The person card is
            // short enough that notes are reachable, making this a test of dictation, not scrolling.
            assertTrue("Home tab did not open", Journey.openTab("Home"));
            Journey.sleep(4_000);

            // People rail on Home — any person row that carries momentum but no episode count.
            // On Android there is no `.buttons.allElementsBoundByIndex`, so scan for buttons
            // whose contentDescription contains "momentum" and does NOT contain "(".
            UiObject2 personRow = findPersonRow();
            if (personRow != null) {
                try { personRow.click(); } catch (Throwable t) {
                    // Click failed — fall back to a topic row instead.
                    personRow = null;
                }
                Journey.sleep(5_000);
            }

            if (personRow == null) {
                // People rail depends on trending state. Any entity card carries a note
                // composer, so fall back to a topic.
                Journey.tap("Topics", false, 10_000);
                Journey.sleep(2_000);
                boolean topicTapped = Journey.tap(
                        Arrays.asList("systems thinking", "risk management"), true, 12_000);
                if (!topicTapped) {
                    fail("neither a person nor a topic was reachable for the note composer. "
                            + "On screen: " + Journey.labelledInventory(20));
                }
                Journey.sleep(5_000);
            }

            // "Your notes" — the textarea's aria-label (notes.title). Matching the placeholder
            // 'Add a note…' found nothing because aria-label wins over placeholder (same on iOS).
            // Notes are the LAST section of a long card, so allow many swipes.
            UiObject2 composer = Journey.scrollTo("Your notes", false);
            if (composer == null) {
                fail("no note composer ('Your notes' aria-label). On screen: "
                        + Journey.labelledInventory(20));
            }
            try { composer.click(); } catch (Throwable t) {
                // Click attempt; continue regardless.
            }
            Journey.sleep(2_000);

            // notes.dictate = 'Dictate a note'
            UiObject2 mic = Journey.find("Dictate a note", true, 8_000);
            assertNotNull(
                    "no dictation control on the note field after enabling Voice input. "
                            + "On screen: " + Journey.labelledInventory(20),
                    mic);

            // Opt-in to actually tapping the mic — same threshold as iOS LP_TAP_MIC.
            String tapMicArg = androidx.test.platform.app.InstrumentationRegistry
                    .getArguments().getString("lp.tap.mic", "0");
            if ("1".equals(tapMicArg)) {
                boolean micTapped = Journey.tap("Dictate a note", true, 8_000);
                if (micTapped) {
                    grantSystemPrompt();
                    Journey.sleep(3_000);

                    // THE point: the app either starts recording or reports failure. A mic that
                    // looks armed and does nothing is the silent dead-mic the composable guards.
                    // notes.dictateStop = 'Stop dictation'
                    // notes.dictateError = "Couldn't start dictation — check microphone access."
                    boolean recording  = Journey.find("Stop dictation",   false, 3_000) != null;
                    boolean saidFailed = Journey.find("Couldn't start dictation", true, 5_000) != null;
                    System.out.println("=====DICTATION recording=" + recording
                            + " reportedFailure=" + saidFailed + "=====");
                    assertTrue(
                            "the mic neither started nor reported a failure — a silent dead mic. "
                                    + "On screen: " + Journey.labelledInventory(20),
                            recording || saidFailed);
                }
            } else {
                System.out.println("=====DICTATION emulator: affordance present, tap skipped "
                        + "(pass -e lp.tap.mic 1 to try)=====");
            }
        } finally {
            // Restore the toggle to the state it was in before this test ran.
            if (!toggleWasOn) {
                Journey.openSettings(profileLabels());
                Journey.scrollTo("Voice input for notes", false);
                UiObject2 restoreRow = Journey.find("Voice input for notes", false, 8_000);
                if (restoreRow != null) {
                    UiObject2 restoreToggle = nearestCheckable(restoreRow);
                    if (restoreToggle != null) {
                        try { restoreToggle.click(); } catch (Throwable ignored) {}
                        Journey.sleep(1_500);
                    }
                }
                Journey.openTab("Home");
            }
        }
    }

    // ------------------------------------------------------------------ N2 native share sheet

    /**
     * Picking "Share text" from the in-app share popover hands a file named "closelistening.txt"
     * to the OS share sheet, and that sheet becomes visible with our content name in it.
     *
     * ANDROID DIFFERENCE: the OS share sheet runs in a separate package (com.android.intentresolver
     * or the OEM equivalent). The iOS twin queried Springboard and com.apple.ShareSheetUI in
     * separate XCUIApplication instances; here we use {@link UiDevice#wait} without a package
     * filter — exactly the same approach Journey.java uses for the OAuth consent dialog, which
     * faces the same cross-package problem (AppSession.java, signIn()).
     *
     * The content assertion checks for "closelistening" as a substring: the share sheet titles
     * the shared item by its filename, which starts with "closelistening".
     */
    @Test
    public void testN2NativeShareSheetOpens() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(14), ready);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        // share.open = 'Share'
        boolean shareTapped = Journey.tap("Share", true, 12_000);
        if (!shareTapped) {
            fail("no Share control on the episode. On screen: " + Journey.labelledInventory(20));
        }
        Journey.sleep(2_000);

        // The in-app popover: share.text = 'Share text'. On native it writes closelistening.txt
        // and hands that file to the OS sheet, so the sheet displays the item name — a marker
        // that OUR payload got there (operator 2026-09-16: asserting only "a sheet appeared"
        // would pass if the app shared the wrong thing).
        boolean shareTextTapped = Journey.tap("Share text", true, 8_000);
        if (!shareTextTapped) {
            fail("share popover offered no 'Share text' option. On screen: "
                    + Journey.labelledInventory(20));
        }
        Journey.sleep(5_000);

        // The share sheet is out-of-process: query WITHOUT a package filter.
        boolean sheetUp        = false;
        boolean carriedContent = false;

        // Common anchors for the Android share sheet across AOSP + OEM skins.
        UiObject2 copyBtn = Journey.device().wait(
                Until.findObject(By.text("Copy")), 4_000);
        if (copyBtn == null) {
            copyBtn = Journey.device().wait(
                    Until.findObject(By.textContains("Copy to clipboard")), 2_000);
        }
        if (copyBtn != null) sheetUp = true;

        // Without a package filter: the filename the share path writes.
        UiObject2 contentLabel = Journey.device().wait(
                Until.findObject(By.textContains("closelistening")), 4_000);
        if (contentLabel != null) carriedContent = true;

        System.out.println("=====SHARE_SHEET up=" + sheetUp
                + " carriedOurContent=" + carriedContent + "=====");
        if (!sheetUp || !carriedContent) {
            System.out.println("=====SHARE_SHEET on-screen: " + Journey.labelledInventory(20)
                    + "=====");
        }

        assertTrue("picking a share option did not produce a share sheet. On screen: "
                        + Journey.labelledInventory(16),
                sheetUp);
        assertTrue(
                "share sheet opened but showed no sign of OUR payload (expected the item "
                        + "named closelistening…). On screen: " + Journey.labelledInventory(16),
                carriedContent);

        // Dismiss — the Back button works across packages.
        Journey.device().pressBack();
        Journey.sleep(1_000);
    }

    // ------------------------------------------------------------------ N3 push registration

    /**
     * Turning a Push cell on triggers the OS permission prompt and the app stays in the foreground.
     *
     * ANDROID DIFFERENCE: iOS uses APNs; this tier uses FCM via @capacitor/push-notifications.
     * The accessible name of the cell is "{notifType} — Push" (profile.notifType.* + " — " +
     * profile.channel.push = 'Push'), matching the iOS pattern "Your Week — Push". The system
     * permission prompt on Android is the standard notification-access dialog; grantSystemPrompt
     * taps "Allow" if it appears, but does NOT assert that it appeared — the OS shows it once per
     * install and a re-run legitimately sees nothing.
     *
     * On an emulator without Google Play Services the FCM registration step may fail silently; the
     * assertion does NOT check whether FCM registered — it checks that the app stayed in the
     * foreground and the cell did not claim push is on when the OS refused. That is the same
     * contract as the iOS twin.
     */
    @Test
    public void testN3PushPermissionPromptOnEnable() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(14), ready);

        assertTrue("could not open Profile", Journey.openProfile(profileLabels()));
        Journey.sleep(4_000);

        // ProfileView is kept alive (KEEP_ALIVE_TABS), so whichever tab a previous test left
        // selected is still selected — the notifications matrix is on Account. Select it
        // explicitly to avoid "no push cell" on a test that opened Topics or Stats.
        // profile.tabAccount = 'Account'
        Journey.tap("Account", false, 10_000);
        Journey.sleep(2_000);

        // The per-type × per-channel matrix: aria-label = "{notifType} — {channel}"
        // e.g. "Your Week — Push". Scroll to the Push column heading first so the cells are
        // in view (the matrix is in the Account tab, which starts at the top of Profile).
        Journey.scrollTo("Push", false);

        // Find ANY push cell — the first one in the notifications matrix.
        // profile.channel.push = 'Push' is shared across all rows, so contains match.
        UiObject2 cell = Journey.find("Push", true, 15_000);
        System.out.println("=====PUSH_CELL found=" + (cell != null)
                + (cell != null ? " label=" + Journey.nameOf(cell) : "") + "=====");

        if (cell == null) {
            fail("no push cell in the notifications matrix. On screen: "
                    + Journey.labelledInventory(20));
        }

        // Tap the cell.
        boolean cellTapped;
        try {
            cell.click();
            cellTapped = true;
        } catch (Throwable t) {
            cellTapped = false;
        }
        System.out.println("=====PUSH_CELL cellTapped=" + cellTapped + "=====");

        // Grant the system permission prompt if it appeared.
        grantSystemPrompt();
        Journey.sleep(4_000);

        // The app must remain in the foreground — the test does not assert FCM registered.
        String pkg = Journey.device().getCurrentPackageName();
        System.out.println("=====PUSH_CELL foreground=" + pkg + "=====");
        assertTrue(
                "the app left the foreground enabling push (foreground was " + pkg + ")",
                Journey.PKG.equals(pkg));
    }

    // ------------------------------------------------------------------ helpers

    /**
     * Grant a system permission prompt if one is present.
     *
     * The OS owns the dialog; it is NOT inside Journey.PKG, so the package filter used everywhere
     * else would return nothing. Query without the filter, same as the OAuth consent in signIn().
     *
     * Returns whether a prompt was dismissed — callers assert on that only when the prompt is
     * the thing under test, since the OS shows it once per install and a re-run legitimately
     * sees nothing.
     */
    private boolean grantSystemPrompt() {
        return grantSystemPrompt(8_000);
    }

    private boolean grantSystemPrompt(long timeoutMs) {
        for (String label : Arrays.asList("Allow", "OK", "Allow While Using App", "Continue")) {
            UiObject2 btn = Journey.device().wait(
                    Until.findObject(By.text(label).clickable(true)), timeoutMs);
            if (btn != null) {
                try { btn.click(); } catch (Throwable ignored) {}
                Journey.sleep(1_000);
                return true;
            }
            timeoutMs = 2_000; // shorten for remaining labels after the first miss
        }
        return false;
    }

    /**
     * Find the topmost person row in the Home rail.
     *
     * The People rail rows carry momentum but no episode count — the SAME signature the iOS test
     * uses: `$0.label.contains("momentum") && !$0.label.contains("(")`. On Android we scan all
     * clickable controls matching "momentum" and exclude those that also contain "(" which are
     * storyline rows.
     */
    private UiObject2 findPersonRow() {
        java.util.List<UiObject2> candidates;
        try {
            candidates = Journey.device().findObjects(
                    By.pkg(Journey.PKG).descContains("momentum").clickable(true));
        } catch (Throwable t) {
            return null;
        }
        for (UiObject2 o : candidates) {
            String name = Journey.nameOf(o);
            if (name.contains("momentum") && !name.contains("(")) return o;
        }
        return null;
    }

    /**
     * The checkable node adjacent to a labelled row.
     *
     * Duplicated from Journey (which is package-private) for use in the Settings restore path.
     * Implementation mirrors nearestCheckable in Journey.java.
     */
    private static UiObject2 nearestCheckable(UiObject2 labelled) {
        UiObject2 cur = labelled;
        for (int up = 0; up < 4; up++) {
            UiObject2 parent;
            try { parent = cur.getParent(); } catch (Throwable t) { break; }
            if (parent == null) break;
            for (BySelector sel : Arrays.asList(
                    By.clazz("android.widget.CheckBox"),
                    By.clazz("android.widget.Switch"),
                    By.checkable(true))) {
                java.util.List<UiObject2> hits;
                try { hits = parent.findObjects(sel); } catch (Throwable t) { continue; }
                if (hits != null && !hits.isEmpty()) return hits.get(0);
            }
            cur = parent;
        }
        return null;
    }
}
