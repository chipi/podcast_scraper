package app.closelistening.player;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;

import java.util.Arrays;

/**
 * The two surfaces that EXIST ONLY ON DEVICE (#2139, porting the iOS suite of the same name).
 *
 * Both are gated on `isNative()`, so the browser tier cannot see either — a Playwright spec would
 * assert an empty row and an empty page and pass for the wrong reason. That is exactly the shape of
 * gap this tier exists to close.
 *
 * 1. **Up next's inline download control.** `showDownload` promotes `DownloadButton` out of the ⋯
 *    into the visible row, because "is this on the device?" is the question Up next answers —
 *    usually right before losing signal. Behind a menu, the answer was invisible.
 *
 * 2. **`/offline` — "On this device".** The only surface that works offline AND signed out. Every
 *    other route is behind the login-first guard, and signing in needs a network, so a lapsed
 *    session on a plane made already-downloaded episodes unreachable from the device holding them.
 *
 * ## What is asserted, and what deliberately is not
 *
 * UI Automator reads the ACCESSIBILITY TREE, not the DOM. It can see a control's NAME ("Download
 * for offline" vs "Downloaded — tap to remove") and whether it is clickable, which is all the
 * behaviour here needs. It CANNOT see colour, so the accent-when-downloaded styling is not
 * asserted — the state is carried by the accessible name too, which is the part that reaches a
 * screen-reader user and the part a test can honestly check. Colour stays a screenshot-review
 * concern.
 *
 * PRECONDITIONS — `make test-android`, which runs `DownloadThroughUITests` first. This suite READS
 * those downloads, so it takes the SHARED account.
 */
/*
 * ORDERED, and the names carry the order. JUnit4's default is hash-based and unspecified, which is
 * fine until one test leaves the device in a state the others cannot start from — and `z1` does
 * exactly that, unavoidably. It signs out WHILE forced-offline, which is a genuine dead end: the
 * offline switch lives in Settings, Settings is behind the masthead avatar, the avatar needs a
 * session, and getting a session needs a network. Nothing inside the test can undo it, so it goes
 * LAST and `make test-android` resets the device afterwards with `pm clear`.
 *
 * The prefixes are ugly. They are also the only thing that makes the ordering visible at the call
 * site rather than buried in a comment nobody reads before adding a test above it.
 */
@RunWith(AndroidJUnit4.class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class NativeOnlySurfacesTests extends UITestCase {

    /**
     * SHARED, because this suite reads what the download suite left behind. A per-suite identity
     * would give it an empty account and every assertion below would pass vacuously on "nothing is
     * downloaded" — the failure mode this file exists to catch.
     */
    @Override
    protected String accountIdentity() {
        return SHARED_SEEDED_IDENTITY;
    }

    private static final String SEEDED_SLUG = "p06-7217050bc6";
    private static final String SEEDED_TITLE = "Signal, Noise, and the Space Between";
    private static final String DOWNLOADED = "Downloaded — tap to remove";
    private static final String DOWNLOAD = "Download for offline";

    // ------------------------------------ 1. Up next carries the download control in the row

    @Test
    public void a1UpNextShowsTheDownloadControlWithoutOpeningTheOverflow() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity(), ready);

        // Queue the seeded episode, so Up next has a row whose download state we already know.
        //
        // The queue control is a MENU ITEM, not part of the visible action row — which carries
        // favourite, add-to-collection, share and ⋯ and nothing else. The download suite reached it
        // without opening anything only because its own flow leaves the ⋯ open; relying on that
        // here reported "neither queue control was reachable" about a player page that was
        // perfectly healthy (2026-09-24).
        AppSession.openEpisode(SEEDED_SLUG);
        assertTrue("no ⋯ on the player, so the queue control cannot be reached. On screen: "
                + Journey.labelledInventory(20), Journey.tap("More actions", false, 20_000));
        Journey.sleep(1_500);
        // Say what the menu actually contains. "neither queue control was reachable" is true of a
        // menu that never opened AND of one that opened without the item, and those need opposite
        // fixes — the first time this failed, the inventory showed the action row with no menu
        // items at all and that was the whole diagnosis (2026-09-24).
        // Ask the DIRECT question. A capped inventory cannot answer it: the panel teleports to
        // `<body>`, so its items sit at the END of the tree and a limit of 30 cut them off — which
        // made an open menu look like a closed one.
        System.out.println("=====UPNEXT add=" + (Journey.find("Add to queue", false, 3_000) != null)
                + " remove=" + (Journey.find("Remove from queue", false, 3_000) != null)
                + " markPlayed=" + (Journey.find("Mark as played", false, 2_000) != null)
                + " download=" + (Journey.find("Download for offline", false, 2_000) != null)
                + " downloaded=" + (Journey.find("Downloaded — tap to remove", false, 2_000) != null)
                + "=====");
        if (!Journey.tap("Add to queue", false, 10_000)) {
            // Already queued by an earlier suite on the shared account — that is the state we want,
            // but say so by asserting the OTHER control rather than assuming it.
            assertNotNull("neither queue control was reachable on the player. On screen: "
                            + Journey.labelledInventory(20),
                    Journey.find("Remove from queue", false, 10_000));
        }

        // The masthead queue control, asserted on the way in rather than taken for granted: before
        // it existed, `/queue` was reachable only from the player or Home's resume hero, so
        // finishing everything made the queue unreachable without starting an episode you did not
        // want to play. If it regresses, the surface under test becomes unreachable on device.
        assertTrue("the masthead queue control was not reachable. On screen: "
                + Journey.labelledInventory(20), Journey.tap("Queue", false, 15_000));

        // The claim: the download control is in the ROW, reachable without opening the ⋯.
        assertNotNull(
                "no download control on the queue row — it is still behind the ⋯, which is the bug "
                        + "this change fixed. On screen: " + Journey.labelledInventory(24),
                Journey.scrollTo(Arrays.asList(DOWNLOADED, DOWNLOAD), false, 20));

        // ...and the ⋯ is still there: promoting download must not have REPLACED the overflow.
        assertNotNull("the overflow disappeared — download was meant to join the row, not take the "
                        + "⋯'s place. On screen: " + Journey.labelledInventory(24),
                Journey.find("More actions", false, 8_000));
    }

    @Test
    public void a2TheSeededEpisodeReadsAsDownloadedRatherThanOfferingToDownloadItAgain() {
        // The STATE half. A control that says "Download for offline" about a file already on disk
        // is the same failure as a Played filter matching nothing: the app holding the answer and
        // showing the opposite.
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity(), ready);

        assertTrue("Library was not reachable", Journey.openTab("Library"));
        assertNotNull("the Downloaded section was not reachable — the download suite may not have "
                        + "run. On screen: " + Journey.labelledInventory(24),
                Journey.scrollTo("Downloaded", false));
        assertNotNull("the seeded episode is missing from Downloaded. On screen: "
                + Journey.labelledInventory(24), Journey.scrollTo(SEEDED_TITLE, false));
        assertNotNull("a downloaded episode still offers \"" + DOWNLOAD + "\" — the state is not "
                        + "reaching the control. On screen: " + Journey.labelledInventory(24),
                Journey.find(DOWNLOADED, false, 8_000));
    }

    // ------------------------------------------- 2. "On this device" — offline AND signed out

    @Test
    public void z1OfflineAndSignedOutStillReachesTheDownloadedEpisodes() {
        // The flight case: not logged in, and unable to log in BECAUSE offline. Signed out and
        // offline, the landing's two CTAs are both dead ends, so the link to what is already on the
        // device has to live there.
        //
        // ORDER MATTERS. Offline is set FIRST, while still signed in, because the switch lives in
        // Settings and Settings is behind the masthead avatar: sign out first and there is no way
        // back in to flip it. On Android this is not merely awkward, it WEDGES the device — a run
        // that leaves forced-offline on cannot sign in again on the next run either.
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity(), ready);

        assertTrue("could not force offline mode", Journey.setOfflineMode(true, profileLabels()));
        try {
            assertTrue("could not sign out", AppSession.signOut());

            assertTrue("the landing offered no route to the downloaded episodes while offline and "
                            + "signed out. On screen: " + Journey.labelledInventory(24),
                    Journey.tap("Play what's downloaded", false, 20_000));

            assertNotNull("the offline downloads page did not render. On screen: "
                    + Journey.labelledInventory(24), Journey.find("On this device", false, 15_000));
            // The POINT of the page: the episodes are actually LISTED, not an empty shell that
            // merely loads.
            assertNotNull("signed out and offline, the downloaded episode is still unreachable — "
                            + "the gap is not closed. On screen: " + Journey.labelledInventory(24),
                    Journey.scrollTo(SEEDED_TITLE, false));
        } finally {
            // NO RECOVERY IS POSSIBLE HERE, and pretending otherwise made this worse.
            //
            // The first version tried `ensureSignedIn` then `setOfflineMode(false)`. It cannot
            // work: signing in needs a network, the network is off, and the switch that would turn
            // it back on is in Settings behind the masthead avatar, which only exists when signed
            // in. So the cleanup threw "no dev identity input" and buried a passing test body under
            // a failure in its own teardown.
            //
            // The honest arrangement is the one now in place: this test runs LAST (see the class
            // comment) and `make test-android` resets the device with `pm clear` afterwards. Said
            // out loud here so a run read in isolation is not a mystery.
            System.out.println("=====OFFLINE_SIGNED_OUT this test deliberately leaves the device "
                    + "offline AND signed out; it cannot undo either from here. `make test-android` "
                    + "resets with `pm clear`. Run it standalone and the NEXT run starts wedged."
                    + "=====");
        }
    }

    @Test
    public void a3TheOfflinePageRefusesToRenderWhileOnline() {
        // The privacy gate, not a nicety.
        //
        // Downloads are namespaced per account so a shared phone cannot show one person's listening
        // history to the next, and this route deliberately reads the LAST account's registry.
        // Offline that is the only door; online it would be a back door. So online it must redirect
        // to the landing, where you can actually sign in. This is the boundary, not a smoke test.
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity(), ready);
        Journey.setOfflineMode(false, profileLabels());
        assertTrue("could not sign out", AppSession.signOut());

        assertFalse("the offline downloads link is offered while online, where signing in is the "
                        + "better answer. On screen: " + Journey.labelledInventory(24),
                Journey.find("Play what's downloaded", false, 5_000) != null);
        assertFalse("the offline downloads page rendered while online",
                Journey.find("On this device", false, 3_000) != null);
    }
}
