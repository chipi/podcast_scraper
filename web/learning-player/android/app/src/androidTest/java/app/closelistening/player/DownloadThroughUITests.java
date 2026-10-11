package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import android.graphics.Rect;

import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.List;

/**
 * Download an episode the way a person does — by tapping the control (#2139, porting #1925/4).
 *
 * This is the suite the offline pair depends on: it leaves two REAL episodes on the device, so
 * nothing downstream has to seed a registry. Seeding covered playback from disk and nothing else —
 * the whole path from "tap Download" to "there is a playable file on this device" (audio-source
 * resolution, the size preflight, the filesystem write, the artwork and transcript that ride
 * along, the registry entry) went unexercised, and two of the three production bugs that arc found
 * lived exactly there.
 *
 * PRECONDITIONS — `make test-android`, which stands up ONE origin serving both the api and the
 * episode audio, reached from the emulator through `adb reverse`. The fixture corpus stores
 * `content.media_url` as a relative `/audio/<id>.mp3` and the app absolutises it against the api
 * base, so with the api alone every download 404s. On Android it additionally needs the debug
 * cleartext config and `allowMixedContent` — media is fetched by the WebView, which will not load
 * an http audio URL into an https page.
 */
@RunWith(AndroidJUnit4.class)
public class DownloadThroughUITests extends UITestCase {

    /**
     * SHARED account, deliberately: this suite CREATES the seed the two offline suites consume.
     * Per-suite isolation would give it an empty account and the seed would be invisible.
     */
    @Override
    protected String accountIdentity() {
        return SHARED_SEEDED_IDENTITY;
    }

    /**
     * Real episodes of "The Drift" (p06) — slugs and titles read from the fixture corpus, not
     * invented. The seed this replaces wrote titles ("Signal Offline One") that no episode has, so
     * it could never have noticed the app disagreeing with the server.
     */
    private static final String FIRST_SLUG = "p06-7217050bc6";
    private static final String FIRST_TITLE = "Signal, Noise, and the Space Between";
    private static final String SECOND_SLUG = "p06-5416bc0968";
    private static final String SECOND_TITLE = "The Conversation About Conversations";

    private static final String DOWNLOAD = "Download for offline";
    private static final String DOWNLOADED = "Downloaded — tap to remove";
    private static final String OVERFLOW = "More actions";
    private static final String QUEUE_ADD = "Add to queue";
    private static final String QUEUE_REMOVE = "Remove from queue";

    @Test
    public void downloadsTwoEpisodesThroughTheUIAndQueuesThem() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        // In THIS order, so the queue ends up [first, second]: auto-advance downstream needs a
        // known one, and the queue control appends.
        download(FIRST_SLUG, FIRST_TITLE);
        download(SECOND_SLUG, SECOND_TITLE);

        // QUEUE THE EPISODES THIS TEST DOWNLOADED, and prove the queue holds them (2026-10-11).
        //
        // The queue step used to run on the PLAYER page and tap the first "Add to queue" it found.
        // The player has no queue control for its own episode, so that was always a "More like
        // this" recommendation's toggle: the test queued an episode it never named, and its flip
        // check ("Remove from queue" anywhere on screen) could not tell. Now each title is queued
        // from its OWN row on the show page, and the queue page must list both, in order.
        AppSession.openLink("podcast/p06");
        Journey.sleep(4_000);
        for (String title : Arrays.asList(FIRST_TITLE, SECOND_TITLE)) queueFromItsRow(title);
        assertQueueHolds(Arrays.asList(FIRST_TITLE, SECOND_TITLE));

        // The end state, asserted by TITLE. A title is unique per episode, where the download-state
        // labels are not — those can be satisfied by some other episode that happened to be
        // downloaded already.
        assertTrue("Library did not open", Journey.openTab("Library"));
        for (String title : Arrays.asList(FIRST_TITLE, SECOND_TITLE)) {
            // Downloaded sits at the END of Saved, below every capture the account holds — it was
            // moved there on 2026-09-18 because a native build had been opening Saved on a
            // device-storage list instead of the user's own captures. `scrollTo` terminates on the
            // page ending rather than a count, so it stays correct wherever the section lands next.
            // Rewind to the top first: the tab keeps its previous scroll position.
            for (int i = 0; i < 6; i++) Journey.swipeDown();
            if (Journey.scrollTo(title, false) == null) {
                fail(title + " is not in the Downloaded list after scrolling to the end of Saved — "
                        + "the UI download did not land. On screen: " + Journey.labelledInventory(80));
            }
        }
    }

    /**
     * The queue toggle of THIS title's row. An EpisodeCard puts its actions under the artwork, LEFT
     * of the title and below the title's top edge, so the row's toggle is the nearest one below the
     * title on that side. The previous card's toggle sits above the title and the next card's
     * further down, so "nearest below" cannot pick a neighbour.
     */
    private static UiObject2 queueToggle(String title) {
        Rect t = null;
        UiObject2 best = null;
        int bestY = Integer.MAX_VALUE;
        try {
            List<UiObject2> all = Journey.device().findObjects(By.pkg(Journey.PKG));
            for (UiObject2 o : all) {
                if (title.equals(Journey.nameOf(o))) {
                    t = Journey.attr(o, UiObject2::getVisibleBounds);
                    if (t != null) break;
                }
            }
            if (t == null) return null;
            for (UiObject2 o : all) {
                String n = Journey.nameOf(o);
                if (!QUEUE_ADD.equals(n) && !QUEUE_REMOVE.equals(n)) continue;
                if (!Boolean.TRUE.equals(Journey.attr(o, UiObject2::isClickable))) continue;
                Rect b = Journey.attr(o, UiObject2::getVisibleBounds);
                if (b == null) continue;
                if (b.centerY() > t.top && b.centerX() < t.left && b.centerY() < bestY) {
                    best = o;
                    bestY = b.centerY();
                }
            }
        } catch (Throwable ignored) {
            // A tree that changed mid-read; the caller looks again.
        }
        return best;
    }

    private void queueFromItsRow(String title) {
        if (Journey.scrollTo(title, false) == null) {
            fail(title + " is not on the show page. On screen: " + Journey.labelledInventory(80));
        }
        UiObject2 toggle = queueToggle(title);
        // Clear of the tab bar and a docked mini-player: the toggle sits a card's artwork below the
        // title, so a title just in view can leave its toggle under them.
        Rect b = toggle == null ? null : Journey.attr(toggle, UiObject2::getVisibleBounds);
        if (b != null && b.bottom > Journey.device().getDisplayHeight() - 260) {
            Journey.swipeUp();
            Journey.sleep(1_000);
            toggle = queueToggle(title);
        }
        if (toggle == null) {
            fail("no queue toggle in the row of " + title + ". On screen: "
                    + Journey.labelledInventory(80));
        }
        if (QUEUE_ADD.equals(Journey.nameOf(toggle))) toggle.click();
        // THAT row's toggle, re-read: the store drives its label, so the flip is the write confirmed.
        boolean flipped = false;
        for (int i = 0; i < 20 && !flipped; i++) {
            UiObject2 now = queueToggle(title);
            flipped = now != null && QUEUE_REMOVE.equals(Journey.nameOf(now));
            if (!flipped) Journey.sleep(500);
        }
        assertTrue("queued " + title + " from its row and its toggle never read \"" + QUEUE_REMOVE
                + "\", so the queue was not written. On screen: " + Journey.labelledInventory(80),
                flipped);
    }

    /** Up next lists exactly what was queued, in the order it was added. */
    private void assertQueueHolds(List<String> titles) {
        // To the top first: the masthead scrolls with the page, and off screen its node is absent.
        for (int i = 0; i < 12; i++) Journey.swipeDown();
        Journey.sleep(1_000);
        assertTrue("the masthead queue control was not reachable. On screen: "
                + Journey.labelledInventory(80), Journey.tap("Queue (", true, 15_000));
        int lastTop = Integer.MIN_VALUE;
        for (String title : titles) {
            UiObject2 row = Journey.scrollTo(title, false);
            assertNotNull(title + " was queued from its row but is not in Up next. On screen: "
                    + Journey.labelledInventory(80), row);
            Rect r = Journey.attr(row, UiObject2::getVisibleBounds);
            int top = r == null ? lastTop : r.top;
            assertTrue("Up next holds " + titles + " out of the order they were added",
                    top >= lastTop);
            lastTop = top;
        }
    }

    /** Open one episode, tap Download, and wait for the app to report it stored. */
    private void download(String slug, String title) {
        AppSession.openEpisode(slug);
        assertTrue("the deep link did not land on " + title + ". On screen: "
                + Journey.labelledInventory(80), Journey.find(title, true, 20_000) != null);

        // The Download control lives in the player's OVERFLOW menu, not the action row, so the menu
        // must be opened before the control exists at all — the panel is `v-if="open"`.
        if (!Journey.tap(OVERFLOW, false, 20_000)) {
            fail("no \"" + OVERFLOW + "\" control on " + title + " after 20s. On screen: "
                    + Journey.labelledInventory(80));
        }

        // Already downloaded from an earlier run? Remove it, so this suite proves the DOWNLOAD path
        // rather than inheriting someone else's success.
        if (Journey.menuItem(DOWNLOADED, 6_000) != null) {
            Journey.tapMenuItem(DOWNLOADED, 6_000);
            // Activating a menu item CLOSES the panel (`@activated="close"`), so the control we
            // would wait on next is not slow — it is off screen. Re-open before looking.
            Journey.sleep(2_000);
            Journey.tap(OVERFLOW, false, 15_000);
            if (Journey.menuItem(DOWNLOAD, 25_000) == null) {
                fail("removing the existing download never restored the Download control for "
                        + title + ". On screen: " + Journey.labelledInventory(80));
            }
        }

        // UNIQUENESS instead of a container scope. The menu teleports to `<body>` to escape clipped
        // ancestors, so it is not inside any content container even when open, and scoping by
        // testid is not available — web content exposes no resource-id, so every selector that
        // works here is an accessible NAME. Asserting there is exactly one is strictly stronger
        // than the scope it replaces: a count of one cannot pick wrong, and a count above one says
        // so instead of guessing.
        //
        // COUNTED AS MENU ITEMS (2026-10-11). The player page now carries episode cards whose rows
        // show "Download for offline" as a BUTTON (measured: 1 MenuItem + 3 Buttons), so a count by
        // name alone fired on healthy markup. The menu's item is the one with the MenuItem role.
        int candidates = Journey.countMenuItems(DOWNLOAD);
        if (candidates > 1) {
            fail(candidates + " Download menu items are on screen for " + title + " — a second menu "
                    + "is open, so tapping would be a guess. On screen: " + Journey.labelledInventory(80));
        }
        if (!Journey.tapMenuItem(DOWNLOAD, 20_000)) {
            // Deliberately NOT conflated with "signed out" or "never loaded": reaching here means
            // the page rendered AND the menu opened. The control is native-only and self-hides on
            // web, so if it is absent the app does not believe it is running natively.
            // A wide inventory on purpose: the panel TELEPORTS to `<body>`, so its items land at
            // the end of the tree and a short cap cuts them off — reporting "no Download control"
            // about a menu whose contents were simply past the limit.
            fail("the overflow menu opened on " + title + " but carries no \"" + DOWNLOAD
                    + "\" control. On screen: " + Journey.labelledInventory(60));
        }

        waitForDownloaded(title);
    }

    /**
     * Poll BY RE-OPENING the menu.
     *
     * The transfer is real — a few MB of fixture audio over the loopback proxy, plus artwork and
     * the transcript — and "Downloaded" is the app's OWN report that the bytes are on disk and the
     * registry agrees, which is the assertion a seeded registry could never make.
     *
     * Every download STATE label lives on the control INSIDE the panel, and activating a menu item
     * closes the panel. Waiting on a closed menu watches an element that cannot appear: on iOS it
     * reported "<none of the known states>", which was true and completely misleading — the control
     * was not in a bad state, it was not on screen.
     */
    private void waitForDownloaded(String title) {
        List<String> states = Arrays.asList(
                DOWNLOADED, "Downloading", "Waiting for Wi-Fi", "Waiting for a connection",
                "Download failed — tap to retry", DOWNLOAD);
        String lastSeen = "<never observed>";
        for (int i = 0; i < 9; i++) {
            // The MENU's state, never a card's: once the first episode is stored, its card on the
            // second episode's page reads "Downloaded" too, and would pass this for the wrong one.
            if (Journey.menuItem(DOWNLOADED, 2_000) != null) return;
            if (Journey.find(OVERFLOW, false, 5_000) != null) Journey.tap(OVERFLOW, false, 5_000);
            StringBuilder seen = new StringBuilder();
            for (String s : states) {
                UiObject2 el = Journey.menuItem(s, 500);
                if (el != null) seen.append(seen.length() == 0 ? "" : ", ").append(s);
            }
            if (seen.length() > 0) lastSeen = seen.toString();
            if (Journey.menuItem(DOWNLOADED, 1_000) != null) return;
            Journey.sleep(10_000);
        }
        // WHICH state it reached, not just "not downloaded". The control's accessible name encodes
        // the whole state machine, so naming the one on screen separates "queued behind the Wi-Fi
        // policy" — the emulator has no radio, so a network-status probe can return unknown, which
        // the scheduler deliberately refuses to start on — from a real transfer failure or a tap
        // that never registered.
        fail(title + " never reached the downloaded state after ~90s. Last state observed on the "
                + "control: " + lastSeen + ". On screen: " + Journey.labelledInventory(80));
    }
}
