package app.closelistening.player;

import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;
import androidx.test.uiautomator.Until;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.InputStream;

/**
 * The SYSTEM controls for playback (operator 2026-10-05): Android had none. The WebView's
 * `navigator.mediaSession` never reached the OS, so the shade showed a bare "Playing in the
 * background" and the lock screen showed nothing. The app now runs a native media session
 * (NowPlaying.java) with a media notification bound to it.
 *
 * Asserted from OUTSIDE the app, the way a listener meets it: the notification shade must carry the
 * episode and a Pause control, and that control must actually pause the app. `dumpsys media_session`
 * confirms the OS sees an active session for this package — the thing the lock screen and the
 * system output switcher read.
 */
@RunWith(AndroidJUnit4.class)
public class MediaNotificationTests extends UITestCase {

    private static final String SLUG = "p06-7217050bc6";
    private static final String TITLE = "Signal, Noise, and the Space Between";
    /**
     * Every shade lookup is scoped to the SYSTEM UI. Unscoped, the first version "found" the title
     * and the Pause button on the app's own player page — the shade was not even open in its
     * screenshot — and passed for a notification it never saw.
     */
    private static final String SYSTEM_UI = "com.android.systemui";

    @Test
    public void theShadeShowsTheEpisodeAndItsPauseControlPausesTheApp() throws Exception {
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), startClean());

        AppSession.openEpisode(SLUG);
        assertTrue("the deep link did not land. On screen: " + Journey.labelledInventory(80),
                Journey.find(TITLE, true, 20_000) != null);
        assertTrue("no Play control. On screen: " + Journey.labelledInventory(24),
                Journey.tap("Play", false, 20_000));
        assertTrue("playback never started. On screen: " + Journey.labelledInventory(24),
                Journey.find("Pause", false, 20_000) != null);

        String sessions = shell("dumpsys media_session");
        assertTrue("the OS has no media session for the app:\n" + sessions,
                sessions.contains("app.closelistening.player") && sessions.contains("CloseListening"));

        Journey.device().openNotification();
        UiObject2 title = Journey.device().wait(Until.findObject(By.pkg(SYSTEM_UI).textContains(TITLE)), 10_000);
        assertTrue("the shade does not show the episode — still the bare keep-alive?", title != null);
        Journey.shot("media-notification-playing");
        UiObject2 pause = Journey.device().wait(Until.findObject(By.pkg(SYSTEM_UI).desc("Pause")), 5_000);
        assertTrue("the media notification has no Pause control", pause != null);
        pause.click();
        Journey.sleep(1_500);
        Journey.device().pressBack(); // close the shade

        assertTrue("Pause in the shade did not pause the app. On screen: "
                + Journey.labelledInventory(24), Journey.find("Play", false, 10_000) != null);
        // Paused, the controls STAY — so the lock screen can resume.
        Journey.device().openNotification();
        UiObject2 play = Journey.device().wait(Until.findObject(By.pkg(SYSTEM_UI).desc("Play")), 5_000);
        Journey.shot("media-notification-paused");
        Journey.device().pressBack();
        assertTrue("paused, the media notification went away (it must stay, with Play)", play != null);
    }

    private static String shell(String command) throws Exception {
        android.os.ParcelFileDescriptor pfd = InstrumentationRegistry.getInstrumentation()
                .getUiAutomation().executeShellCommand(command);
        try (InputStream in = new FileInputStream(pfd.getFileDescriptor());
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            return out.toString("UTF-8");
        }
    }
}
