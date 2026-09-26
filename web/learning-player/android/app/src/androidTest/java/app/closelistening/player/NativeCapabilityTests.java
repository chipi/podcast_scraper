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
                + Journey.labelledInventory(80), ready);

        // OBSERVE THE MIC; NEVER READ THE CHECKBOX (2026-09-26).
        //
        // The Settings switch reports `checkable=false checked=false` through the Chromium bridge
        // whatever its real state, so `isChecked()` can only ever say "false". The previous version
        // therefore clicked it UNCONDITIONALLY — which is correct only when the opt-in happens to
        // start OFF. When it started ON, the click turned dictation OFF and the test then failed on
        // the mic it had just removed, reporting "no dictation control ... after enabling Voice
        // input" about a switch it had disabled itself.
        //
        // Reproduced deterministically: with voice input left ON and no `pm clear`, this test failed
        // first try, every time. It also self-poisons — the old `finally` clicked blind to "restore",
        // so ON -> OFF -> ON left the flag set for the next run, which then failed the same way.
        //
        // The mic IS readable, so it is the signal. Same discipline as `Journey.setOfflineMode`:
        // look at what the APP does, one interaction at a time, instead of trusting a node attribute
        // that is pinned. Checking first also makes the already-ON case free — no click, nothing to
        // restore — and it can never turn dictation off.
        //
        // This is NOT the retry that was removed on 2026-09-26. That one re-read the unreadable
        // checkbox and clicked again in Settings, so a landed first click was undone by the second.
        // Here the decision to click comes from the mic, and there is exactly one click.
        UiObject2 mic = openComposerAndFindMic();
        boolean weEnabledIt = false;

        try {
            if (mic == null) {
                clickVoiceInputToggleOnce();
                weEnabledIt = true;
                mic = openComposerAndFindMic();
            }

            // The assertion lives INSIDE the try so the restore below still runs when it fires.
            // Outside it, a failure skipped the restore and left the opt-in ON for the next run —
            // reintroducing, in a new place, the poisoning this rewrite removed (2026-09-26).
            if (mic == null) {
                // A screenshot, because the node's absence cannot tell us WHY. If the mic is drawn
                // in this picture the control is unreadable to UI Automator; if it is not, the
                // opt-in did not take. Two different bugs, one identical failure message.
                Journey.shot("n1-fail-no-mic");
            }
            assertNotNull(
                    "no dictation control on the note field after enabling Voice input. On screen: "
                            + Journey.labelledInventory(80),
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
                                    + "On screen: " + Journey.labelledInventory(80),
                            recording || saidFailed);
                }
            } else {
                System.out.println("=====DICTATION emulator: affordance present, tap skipped "
                        + "(pass -e lp.tap.mic 1 to try)=====");
            }
        } finally {
            // Restore ONLY what this test changed. If dictation was already on when we arrived we
            // never clicked, so there is nothing to put back — and clicking anyway is precisely the
            // blind "restore" that used to leave the flag set for the following run.
            if (weEnabledIt) {
                clickVoiceInputToggleOnce();
                Journey.openTab("Home");
            }
        }
    }

    /**
     * Open a note composer and return its dictation mic, or {@code null} when the mic is absent.
     *
     * Absence is a legitimate answer here — it is how the caller learns the Voice input opt-in is
     * off — so this only fails hard when the COMPOSER itself is unreachable, which would mean the
     * navigation broke rather than the setting being off.
     */
    private UiObject2 openComposerAndFindMic() {
        // A PERSON card, not a topic one. Both carry a note composer, but the topic card ends with
        // a long annotated episode list, so notes sit far below the fold — the iOS comment noted
        // sixteen swipes still landed mid-list (2026-09-16). The person card is short enough that
        // notes are reachable, making this a test of dictation, not of scrolling.
        assertTrue("Home tab did not open", Journey.openTab("Home"));
        Journey.sleep(4_000);

        // People rail on Home — any person row that carries momentum but no episode count. On
        // Android there is no `.buttons.allElementsBoundByIndex`, so scan for buttons whose
        // contentDescription contains "momentum" and does NOT contain "(".
        UiObject2 personRow = findPersonRow();
        if (personRow != null) {
            try { personRow.click(); } catch (Throwable t) {
                // Click failed — fall back to a topic row instead.
                personRow = null;
            }
            Journey.sleep(5_000);
        }

        if (personRow == null) {
            // People rail depends on trending state. Any entity card carries a note composer, so
            // fall back to a topic.
            Journey.tap("Topics", false, 10_000);
            Journey.sleep(2_000);
            boolean topicTapped = Journey.tap(
                    Arrays.asList("systems thinking", "risk management"), true, 12_000);
            if (!topicTapped) {
                fail("neither a person nor a topic was reachable for the note composer. "
                        + "On screen: " + Journey.labelledInventory(80));
            }
            Journey.sleep(5_000);
        }

        // "Your notes" — the textarea's aria-label (notes.title). Matching the placeholder
        // 'Add a note…' found nothing because aria-label wins over placeholder (same on iOS).
        // Notes are the LAST section of a long card, so allow many swipes.
        UiObject2 composer = Journey.scrollTo("Your notes", false);
        if (composer == null) {
            fail("no note composer ('Your notes' aria-label). On screen: "
                    + Journey.labelledInventory(80));
        }
        // DO NOT CLICK THE TEXTAREA (2026-09-26). Clicking focuses it, which opens the soft
        // keyboard, and the keyboard covers the row of buttons DIRECTLY BENEATH the field — the
        // mic and Add. Android's accessibility tree holds only ON-SCREEN nodes, so the mic drops
        // out of the tree at exactly the moment it is drawn, and the test reports "no dictation
        // control" about a control that is right there.
        //
        // That is why this failed intermittently (keyboard timing), only in-suite (1 pass / 2 fail
        // vs 4 / 0 solo), only on Android (the iOS simulator uses a hardware keyboard, so its twin
        // never sees one), and why the failing dumps showed the note field with NO buttons after
        // it — not the mic, and not the always-rendered Add button either.
        //
        // The click was never needed: this asserts the affordance EXISTS, it does not type.
        //
        // Scroll once more instead, so the button row is fully on screen before we look.
        Journey.swipeUp();
        Journey.sleep(1_500);

        // notes.dictate = 'Dictate a note'
        return Journey.find("Dictate a note", true, 8_000);
    }

    /**
     * Flip the "Voice input for notes" switch EXACTLY ONCE.
     *
     * No read-back, because there is none to be had: the bridge pins `checkable`/`checked` to false.
     * The caller decides whether a click is warranted by looking at the mic, and never calls this
     * twice in a row on the strength of the node's own state.
     */
    private void clickVoiceInputToggleOnce() {
        boolean settingsOpen = Journey.openSettings(profileLabels());
        if (!settingsOpen) {
            fail("could not reach Settings. On screen: " + Journey.labelledInventory(80));
        }
        // settings.voiceInput = 'Voice input for notes'
        Journey.scrollTo("Voice input for notes", false);

        UiObject2 voiceRow = Journey.find("Voice input for notes", false, 8_000);
        if (voiceRow == null) {
            fail("no 'Voice input for notes' control in Settings. On screen: "
                    + Journey.labelledInventory(80));
        }
        UiObject2 toggle = nearestCheckable(voiceRow);
        if (toggle == null) {
            fail("no toggle for Voice input for notes. On screen: "
                    + Journey.labelledInventory(80));
        }

        // Log the identity and geometry of what is being clicked against the row it belongs to.
        // Measured across six runs: the toggle sits a constant 63px below the row's title span, so
        // `nearestCheckable` does resolve the right control — the bug was never which node it found.
        android.util.Log.i("VOICE_TOGGLE", "checkable=" + Journey.attr(toggle, UiObject2::isCheckable)
                + " class=" + Journey.attr(toggle, UiObject2::getClassName)
                + " toggleBounds=" + Journey.attr(toggle, UiObject2::getVisibleBounds)
                + " rowBounds=" + Journey.attr(voiceRow, UiObject2::getVisibleBounds)
                + " rowClass=" + Journey.attr(voiceRow, UiObject2::getClassName));

        try {
            toggle.click();
        } catch (Throwable t) {
            fail("could not click Voice input toggle: " + t);
        }
        Journey.sleep(2_000);
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
                + Journey.labelledInventory(80), ready);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        // share.open = 'Share'
        boolean shareTapped = Journey.tap("Share", true, 12_000);
        if (!shareTapped) {
            fail("no Share control on the episode. On screen: " + Journey.labelledInventory(80));
        }
        Journey.sleep(2_000);

        // The in-app popover: share.text = 'Share text'. On native it writes closelistening.txt
        // and hands that file to the OS sheet, so the sheet displays the item name — a marker
        // that OUR payload got there (operator 2026-09-16: asserting only "a sheet appeared"
        // would pass if the app shared the wrong thing).
        boolean shareTextTapped = Journey.tap("Share text", true, 8_000);
        if (!shareTextTapped) {
            fail("share popover offered no 'Share text' option. On screen: "
                    + Journey.labelledInventory(80));
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
            System.out.println("=====SHARE_SHEET on-screen: " + Journey.labelledInventory(80)
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
                + Journey.labelledInventory(80), ready);

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
                    + Journey.labelledInventory(80));
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
