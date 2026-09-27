package app.closelistening.player;

import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Drives the Settings → "Offline mode" switch on device (#2139, porting the iOS suite).
 *
 * WHY a test rather than a tap: the switch persists to `localStorage` (`lp.forceOffline`), and the
 * host has no way to reach Chromium's storage for the WebView — it cannot be pre-seeded the way a
 * native token can. The device tier is the only thing that can press it.
 *
 * Deliberately narrow: it flips the switch and asserts it flipped, nothing else. The observation of
 * what forced-offline does to the app under broader conditions lives in `OfflineCacheTests` and
 * `OfflineAutoAdvanceTests`.
 *
 * ANDROID DIFFERENCE — the iOS version read the switch's value directly via WebKit's value="0"/"1"
 * exposure. Chromium reports `checkable=false` on `<input type=checkbox>` so `isChecked()` is
 * pinned to false and cannot be used. State is instead read from the offline banner's presence on
 * Home — `Journey.forcedOfflineBannerShowing()` — which proves the ACTUAL state (did the app enter
 * forced-offline mode?) rather than the appearance of a tick. Both assertions remain: that the
 * switch was driven, and that the switch was restored.
 */
@RunWith(AndroidJUnit4.class)
public class ConfigOfflineToggleTests extends UITestCase {

    @Test
    public void togglesForcedOfflineOn() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        // Read the initial state from the banner (not the checkbox — see class-level note).
        boolean initiallyOn = Journey.forcedOfflineBannerShowing();

        // Drive it to ON, then read back.
        boolean setOn = Journey.setOfflineMode(true, profileLabels());
        assertTrue("could not turn Offline mode ON. On screen: " + Journey.labelledInventory(16),
                setOn);

        // Verify the banner confirms the app entered forced-offline.
        Journey.openTab("Home");
        Journey.sleep(2_500);
        boolean bannerOn = Journey.forcedOfflineBannerShowing();
        assertTrue(
                "Offline mode did not take effect after being turned ON — the banner did not appear. "
                        + "On screen: " + Journey.labelledInventory(80),
                bannerOn);

        // RESTORE. The switch is PERSISTED (localStorage), so leaving it flipped hands every later
        // test an app in forced-offline: reads fast-fail, the surfaces empty out, and the failures
        // read as unrelated product bugs — three tests blamed for a fourth test's side effect
        // (the cross-suite leak of 2026-09-16 on iOS, reproduced here without this restore).
        //
        // Always restore to the state it was in BEFORE this test ran, not unconditionally to OFF —
        // so if this suite is run against an app that was already in forced-offline, it leaves it
        // that way.
        boolean restored = Journey.setOfflineMode(initiallyOn, profileLabels());
        assertTrue(
                "could not restore Offline mode to its pre-test state (" + initiallyOn + "). "
                        + "On screen: " + Journey.labelledInventory(16),
                restored);

        Journey.openTab("Home");
        Journey.sleep(2_500);
        boolean bannerAfterRestore = Journey.forcedOfflineBannerShowing();
        assertTrue(
                "left the offline switch in the wrong state after the test (expected="
                        + initiallyOn + " actual=" + bannerAfterRestore + "). "
                        + "On screen: " + Journey.labelledInventory(16),
                bannerAfterRestore == initiallyOn);
    }
}
