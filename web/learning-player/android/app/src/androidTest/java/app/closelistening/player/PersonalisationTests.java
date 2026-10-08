package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Personalisation journeys (#2139, porting the iOS suite of the same name).
 *
 * Both tests are about state the app only has AFTER the user does something, which is why the first
 * journey pass produced two empty panels and proved nothing:
 *
 *   - Stats said "Start listening to build your stats." because nothing had ever been played.
 *   - Topics said "No interests chosen yet." because no interests had ever been picked.
 *
 * An empty panel is only meaningful once the thing that fills it has happened. So these tests
 * PERFORM the action first (play episodes / choose interests) and then assert the panel changed.
 *
 * INTERESTS ARE EDITED IN PLACE (2026-10-04). Profile › Interests has a section per kind, each
 * offering "Follow X" suggestions behind its "+ Add"; a tap follows at once and the item moves to the section's
 * followed row, whose control is "Stop following X". This replaced a toggle picker whose chips
 * reported `checked=false` whatever their state, which is why the earlier version of test10 had to
 * infer selection from the card's empty state. Suggestions never include what is already followed,
 * so every "Follow …" control on the tab is safe to tap and there is no state to infer.
 */
@RunWith(AndroidJUnit4.class)
public class PersonalisationTests extends UITestCase {

    // Two real episodes that exist in the fixture corpus.
    private static final List<String> EPISODES = Arrays.asList("p09-a4bbb5dde3", "p07-2aceab172c");

    // ---- 09 play → stats ---------------------------------------------------

    @Test
    public void test09PlayFillsStats() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        for (String slug : EPISODES) {
            AppSession.openEpisode(slug);
            Journey.sleep(6_000);

            // The transport sits under the tab bar until scrolled — the trap the playback suite
            // documents. Scroll to it rather than hoping it is on screen.
            Journey.scrollTo(Arrays.asList("Play", "Pause", "Replay"), false, 5);

            // The episode may ALREADY be playing, or already finished: playback position is
            // persisted server-side, so a slug another test has opened comes back resumed — the
            // page then shows a "NEXT · IN 0:06" auto-advance countdown and no Play control at
            // all, which is how this test can fail (2026-09-16 on iOS). "Already playing"
            // satisfies the requirement — playback time accruing — just as well as "just started".
            boolean tapResult = Journey.tap("Play", false, 10_000);
            if (!tapResult) {
                // Not a failure if something is already playing.
                boolean alreadyPlaying = Journey.find("Pause", false, 5_000) != null;
                assertTrue(
                        "no Play control and nothing playing on " + slug + ". On screen: "
                                + Journey.labelledInventory(80),
                        alreadyPlaying);
            }

            // Let real playback time accrue; stats are driven by reported position, not just
            // opening a page.
            Journey.sleep(12_000);

            // Pausing flushes the position to the server in the same way backgrounding would.
            Journey.tap("Pause", false, 10_000);
            Journey.sleep(3_000);
        }

        // Navigate to Profile → Stats.
        assertTrue("Profile did not open", Journey.openProfile(profileLabels()));
        Journey.sleep(3_000);
        assertTrue("no Stats tab on Profile. On screen: " + Journey.labelledInventory(80),
                Journey.tap("Stats", false, 12_000));
        Journey.sleep(4_000);

        // Stats must no longer show its never-listened empty state.
        // i18n: stats.empty = "Start listening to build your stats."
        boolean emptyStillShowing = Journey.find("Start listening to build your stats", true, 5_000) != null;
        assertNull(
                // 20 items stops in the masthead on this screen, which says nothing about Stats.
                // Every dump raised tonight has changed the diagnosis it supported (2026-09-26).
                "Stats still shows its never-listened empty state after playing two episodes. "
                        + "On screen: " + Journey.labelledInventory(80),
                emptyStillShowing ? Journey.find("Start listening", true, 1_000) : null);
    }

    // ---- 10 interests → follows + Home ---------------------------------------

    @Test
    public void test10InterestsRenderAndFeedHome() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        assertTrue("Profile did not open", Journey.openProfile(profileLabels()));
        Journey.sleep(3_000);
        // i18n: profile.tabInterests = "Interests"
        assertTrue("no Interests tab on Profile. On screen: " + Journey.labelledInventory(80),
                Journey.tap("Interests", false, 12_000));

        // Suggestions live behind the Topics section's "+ Add" (one open at a time, 2026-10-04).
        // i18n: interestSections.add_topic = "Add a topic".
        assertTrue("no Add control on the Topics section. On screen: " + Journey.labelledInventory(80),
                Journey.tap("Add a topic", false, 12_000));
        // WAIT for the suggestions — they are fetched, and a snapshot taken while they load finds
        // nothing to tap. i18n: interestSections.suggested = "Suggested".
        assertTrue("the Interests tab never finished loading its suggestions. On screen: "
                + Journey.labelledInventory(80), Journey.find("Suggested", false, 20_000) != null);
        Journey.sleep(2_000);

        // Follow up to three suggestions. i18n: interestSections.follow = "Follow {name}".
        List<String> followed = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            String control = firstFollowControl();
            if (control == null) break;
            if (!Journey.tap(control, false, 5_000)) break;
            followed.add(control.substring(FOLLOW.length()));
            Journey.sleep(1_500);
        }
        Journey.mark("=====INTERESTS_FOLLOWED " + followed + "=====");
        assertTrue("no Follow suggestion could be tapped on the Interests tab. On screen: "
                + Journey.labelledInventory(80), !followed.isEmpty());

        // Each tap moved its item into the followed row. i18n: interestSections.remove.
        for (String label : followed) {
            assertNotNull("'" + label + "' was tapped but is not shown as followed. On screen: "
                            + Journey.labelledInventory(80),
                    Journey.scrollTo(Arrays.asList(UNFOLLOW + label), false, 10));
        }

        // Interests feed Home's recommendations, so Home must REFLECT them. Home's guided start
        // (2026-10-07) asks for three: at three it moves on from the interests step, so
        // interests.cardCta = "Choose interests" is gone; below three it stays and says how many
        // are left — guided.interestsToGo = "One more to go" / "{count} more to go".
        Journey.openTab("Home");
        Journey.sleep(6_000);
        if (followed.size() >= 3) {
            boolean stillPrompting = Journey.find("Choose interests", true, 5_000) != null;
            assertNull(
                    "Home is still on the interests step after three interests were chosen. On screen: "
                            + Journey.labelledInventory(80),
                    stillPrompting ? Journey.find("Choose interests", true, 1_000) : null);
        } else {
            assertNotNull(
                    "Home does not show how many interests are still needed after " + followed.size()
                            + " were chosen. On screen: " + Journey.labelledInventory(80),
                    Journey.find("more to go", true, 5_000));
        }

        // And the follows were WRITTEN, not just flipped on screen: back on Profile they are there.
        assertTrue("Profile did not open", Journey.openProfile(profileLabels()));
        Journey.sleep(3_000);
        Journey.tap("Interests", false, 10_000);
        Journey.sleep(3_000);
        List<String> names = new ArrayList<>();
        for (String label : followed) names.add(UNFOLLOW + label);
        assertNotNull(
                "none of the interests just followed (" + followed + ") are on the Profile "
                        + "Interests tab after leaving it. On screen: " + Journey.labelledInventory(80),
                Journey.scrollTo(names, false, 20));
    }

    private static final String FOLLOW = "Follow ";
    private static final String UNFOLLOW = "Stop following ";

    /** The accessible name of the first "Follow …" suggestion on screen, or null when none. */
    private static String firstFollowControl() {
        for (UiObject2 o : Journey.device().findObjects(By.pkg(Journey.PKG))) {
            String name = Journey.nameOf(o);
            // "Follow show" is a show page's control, never on this tab, but cheap to rule out.
            if (name.startsWith(FOLLOW) && !name.startsWith("Follow show")) return name;
        }
        return null;
    }
}
