package app.closelistening.player;

import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A full visual sweep of the app's surfaces on Android — the sibling of `ScreenshotTourTests.swift`
 * (operator request 2026-10-03). The shots are stitched into one contact sheet by
 * `make android-contact-sheet`.
 *
 * Deliberately NOT assertion-heavy: this is a CAMERA, not a gate. The journey suites
 * (`AppJourneyTests`, `PersonalisationTests`, `OfflineCacheTests`, `ServerDegradedTests`) own the
 * assertions. If this failed on a missing element it would abandon the tour half-way and produce a
 * partial sheet, which is the opposite of what it is for — a surface that failed to load is exactly
 * the thing the sheet should show.
 *
 * ## The one assertion it DOES make
 *
 * It declares, at the end, every frame it never reached. The iOS tour learned this the hard way: its
 * frames were best-effort with no tally, so a stuck sheet silently dropped the last three screens
 * and the tour still reported success — the gap only surfaced when a human noticed a screenshot they
 * expected was not on the sheet. Shooting what it can and then declaring what it could not is the
 * honest version, and it is the difference between a short sheet and a short sheet nobody notices.
 *
 * ## Why the frame names match iOS exactly
 *
 * The two sheets are meant to be read SIDE BY SIDE. A surface that renders differently on Android is
 * the entire point of having both, and that comparison only works if `t13-episode` means the same
 * screen on each. The numeric prefixes drive tile order in the sheet.
 *
 * PRECONDITIONS: app installed, fixture api reachable. Run after the journey + personalisation
 * suites if you want Stats/Topics/Saved populated rather than empty — a freshly-signed-in account
 * photographs as a wall of empty states, which is the half a visual review cannot judge.
 */
@RunWith(AndroidJUnit4.class)
public class ScreenshotTourTests extends UITestCase {

    /**
     * SHARED account, deliberately. This suite photographs a POPULATED app, and the make target runs
     * the seeding suites first on purpose. Per-suite isolation would hand it an empty account and
     * the seed would be invisible — exactly the regression the iOS tour hit when per-suite
     * identities landed.
     */
    @Override
    protected String accountIdentity() {
        // An explicit `-e identity <name>` still wins, so the make target can put the seeding
        // suites and this one on the SAME account in one place rather than relying on two
        // defaults happening to agree.
        String forced = forcedIdentity();
        return (forced != null && !forced.isEmpty()) ? forced : SHARED_SEEDED_IDENTITY;
    }

    /** An episode with insights in the committed v3 corpus, so the panels photograph full. */
    private static final String EPISODE_SLUG = "p09-a4bbb5dde3";

    /** Every frame this tour is supposed to produce. The run FAILS if any is missing. */
    private static final List<String> EXPECTED_FRAMES = Arrays.asList(
            "t01-home", "t02-discover", "t03-search",
            "t04-library-following", "t05-library-saved", "t06-library-boards", "t07-library-revisit",
            "t08-profile-account", "t09-profile-interests", "t10-profile-stats",
            "t11-settings", "t12-settings-config",
            "t13-episode", "t14-episode-insights", "t15-episode-keypoints", "t16-episode-entities",
            "t17-share-popover", "t18-add-to-collection",
            "t19-topic", "t20-storyline", "t21-person",
            "t25-home-offline", "t26-home-final");

    private final Set<String> shotFrames = new LinkedHashSet<>();

    /** Shoot the current screen and record that we got there. */
    private void frame(String name) {
        Journey.shot(name);
        shotFrames.add(name);
    }

    /** Give the WebView time to paint before the shutter. Software GLES on CI is not fast. */
    private void settle(long ms) {
        Journey.sleep(ms);
    }

    @Test
    public void testTourEverySurface() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        // --- primary tabs ---------------------------------------------------------------------
        settle(4_000);
        frame("t01-home");

        if (Journey.openTab("Discover")) {
            settle(5_000);
            frame("t02-discover");
            // Search from Discover's own box: phones have no Search tab and no header magnifier,
            // so this is the route a person actually takes. With a query — Search on an empty box
            // does nothing, and t03 came back identical to t02.
            if (Journey.searchFromDiscover("risk")) {
                settle(4_000);
                frame("t03-search");
            }
        }

        // --- library, every tab ---------------------------------------------------------------
        if (Journey.openTab("Library")) {
            settle(4_000);
            // Library opens on Saved, so the Following frame needs its own tap.
            Journey.tap("Following", false, 8_000);
            settle(3_000);
            frame("t04-library-following");
            String[][] tabs = {
                { "Saved", "t05-library-saved" },
                { "Boards", "t06-library-boards" },
                { "Revisit", "t07-library-revisit" },
            };
            for (String[] t : tabs) {
                if (Journey.tap(t[0], false, 10_000)) {
                    settle(3_000);
                    frame(t[1]);
                }
            }
        }

        // --- profile, every tab, then settings -------------------------------------------------
        if (Journey.openProfile(profileLabels())) {
            settle(4_000);
            frame("t08-profile-account");
            if (Journey.tap("Interests", false, 10_000)) { settle(3_000); frame("t09-profile-interests"); }
            if (Journey.tap("Stats", false, 10_000)) { settle(3_000); frame("t10-profile-stats"); }
        }
        if (Journey.openSettings(profileLabels())) {
            settle(3_000);
            frame("t11-settings");
            // Config lives at the bottom — the offline switch and space reclaim.
            Journey.scrollTo("Offline mode", true);
            settle(1_500);
            frame("t12-settings-config");
        }

        // --- episode + knowledge panel ---------------------------------------------------------
        AppSession.openEpisode(EPISODE_SLUG);
        settle(6_000);
        frame("t13-episode");
        if (Journey.tap("Episode notes", true, 12_000)) {
            settle(4_000);
            frame("t14-episode-insights");
            // SCROLL to these sections, never tap their headers: they are <details> that are OPEN by
            // default, so a tap CLOSES them and the frames came out collapsed (2026-10-08, iOS tour).
            if (Journey.scrollTo(java.util.Arrays.asList("Key points"), true, 8) != null) {
                settle(2_000); frame("t15-episode-keypoints");
            }
            if (Journey.scrollTo(java.util.Arrays.asList("Topics & People"), true, 8) != null) {
                settle(2_000); frame("t16-episode-entities");
            }
            // Close the notes sheet by name ("Close panel", kp.close): `dismissCards` taps only an
            // exact "Close"/"Back", so the sheet stayed over the page and hid the Share and Add to
            // board controls the next two frames need (2026-10-08).
            Journey.tap(Arrays.asList("Close panel"), false, 4_000);
            settle(1_500);
        }

        // --- the overlays, which only a picture can confirm render correctly -------------------
        Journey.dismissCards();
        AppSession.openEpisode(EPISODE_SLUG);
        settle(5_000);
        // EXACTLY "Share": a contains-match also hits the notes export button ("Open these episode
        // notes to download or share them"), and t17 photographed the export viewer (2026-10-08).
        if (Journey.tap("Share", false, 10_000)) {
            settle(3_000);
            frame("t17-share-popover");
            Journey.tap(Arrays.asList("Close", "Cancel"), true, 4_000);
        }
        // Reopen the episode first: the share menu has no Close, so it stayed open over the page and
        // swallowed this tap — t18 was never shot (2026-10-08). And exactly "Add to board":
        // `contains "Add"` matches other controls.
        AppSession.openEpisode(EPISODE_SLUG);
        settle(4_000);
        if (Journey.tap(Arrays.asList("Add to board"), false, 10_000)) {
            settle(3_000);
            frame("t18-add-to-collection");
            Journey.tap(Arrays.asList("Close", "Cancel"), true, 4_000);
        }

        // --- entity surfaces --------------------------------------------------------------------
        // By DEEP LINK (2026-10-08): walking Home's chips landed on the wrong page — Search, the
        // queue, the notes export — and one stray tap stranded the rest of the tour. Same ids as the
        // fixture corpus serves.
        Journey.dismissCards();
        AppSession.openLink("topic/topic:systems-thinking");
        settle(5_000);
        frame("t19-topic");
        AppSession.openLink("storyline/thc:managing-risk");
        settle(5_000);
        frame("t20-storyline");
        AppSession.openLink("person/person:dr-elena-fischer");
        settle(5_000);
        frame("t21-person");

        // --- offline, then back ------------------------------------------------------------------
        Journey.dismissCards();
        if (Journey.setOfflineMode(true, profileLabels())) {
            Journey.openTab("Home");
            settle(4_000);
            frame("t25-home-offline");
            Journey.setOfflineMode(false, profileLabels());
        }
        Journey.openTab("Home");
        settle(4_000);
        frame("t26-home-final");

        // --- declare what we never reached --------------------------------------------------------
        StringBuilder missing = new StringBuilder();
        int n = 0;
        for (String want : EXPECTED_FRAMES) {
            if (!shotFrames.contains(want)) {
                if (n++ > 0) missing.append(", ");
                missing.append(want);
            }
        }
        assertTrue("the tour never reached " + n + " screen(s): " + missing
                + " — a sheet left open swallows every later tap, so check the SHOT markers in"
                + " logcat above", n == 0);
    }
}
