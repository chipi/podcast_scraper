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

        // Queue the seeded episode from a LIST ROW, then open the queue panel from under the
        // artwork.
        //
        // The first version tapped ⋯ on the player and looked for a queue control there, copying
        // the iOS suite. It is not there and never has been: `EpisodeActions` renders `QueueButton`
        // in the visible ROW when `!hideQueue` and inside the ⋯ only when `hideQueue`, and the
        // player uses neither — its action row is favourite, add-to-collection, share, ⋯. A direct
        // probe of the open menu settled it (2026-09-24):
        //
        //     =====UPNEXT add=false remove=false markPlayed=true download=false downloaded=true=====
        //
        // The menu was open the whole time. The assertion was inherited rather than checked, which
        // is the same mistake as inventing a fixture title — asserting against a remembered app
        // instead of this one. Operator: "queue actions exist only when you open queue in the upper
        // section under the artwork."
        // QUEUE FROM A LIST. Not from the player, which is what the iOS suite does.
        //
        // Add-to-queue does not exist on the episode player, and that is correct product
        // behaviour rather than a gap: you are already playing the episode, so queueing it again
        // means nothing (operator 2026-09-25). The control lives on LIST rows — browse, library,
        // Home's what's-new — where you are scanning episodes you have not committed to.
        //
        // The iOS suite opens an episode and looks for "Add to queue" there. It passes only
        // because its lookup cannot tell the player's buttons apart and matches a different one;
        // the enumeration this harness now uses is precise, so the same step fails honestly.
        // Porting that step faithfully would mean copying a test that verifies nothing.
        assertTrue("Home was not reachable", Journey.openTab("Home"));
        for (int i = 0; i < 6; i++) Journey.swipeDown();
        boolean queued = Journey.scrollTo(java.util.Arrays.asList("Add to queue"), false, 25) != null
                && Journey.tap("Add to queue", false, 10_000);
        if (!queued) {
            // Already queued on the shared account is a legitimate state — assert the OTHER
            // control rather than assuming it.
            queued = Journey.find("Remove from queue", false, 5_000) != null;
        }
        assertTrue("no queue control on any Home list row, so nothing can reach Up next. On screen: "
                + Journey.labelledInventory(30), queued);

        // Then the queue itself, from the masthead.
        //
        // TO THE TOP FIRST. Queueing above scrolled Home down, and the masthead scrolls with the
        // page — on Android an off-screen node is not merely unreachable, it is ABSENT from the
        // accessibility tree, so `tap` has nothing to find and the failure reads as "the control is
        // not there". Measured 2026-09-26: the dump here showed Your Week, Trends and the
        // time-range chips — the middle of Home, with the masthead well above it.
        for (int i = 0; i < 12; i++) Journey.swipeDown();
        Journey.sleep(1_000);
        // CONTAINS, because the badge joins the name as soon as anything is queued ("Queue (1)") —
        // which is the state this test has just created. Exact "Queue" only matches an EMPTY queue,
        // so it could never match here. The iOS twin hit precisely this and was fixed the same way.
        assertTrue("the masthead queue control was not reachable. On screen: "
                + Journey.labelledInventory(80), Journey.tap("Queue", true, 15_000));

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
