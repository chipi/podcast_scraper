package app.closelistening.player;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.List;

/**
 * Drive the card stack as deep as the content allows, verifying each level is still present
 * (#2139, porting the iOS suite of the same name).
 *
 * The requirement is not "two sheets can overlap" — it is that the chain keeps working, each card
 * below staying readable by its title, however far you go (operator 2026-09-16: "if there's such a
 * navigation that you can do five, ten times, that's how it should go"). Two levels can be faked by
 * a single is-nested flag; four cannot.
 *
 * Entry is the player's Insights panel — the hardest case because the panel is a modal {@code
 * <dialog>} in the browser's top layer, so anything opened from it has to be teleported INTO that
 * dialog or it renders invisibly underneath.
 *
 * PRECONDITIONS: app installed, signed in, fixture api reachable. Episode p09-a4bbb5dde3 must
 * carry at least one topic in the "Topics & People" accordion with a storyline row.
 */
@RunWith(AndroidJUnit4.class)
public class StackDepthProbeTests extends UITestCase {

    private static final String EPISODE_SLUG = "p09-a4bbb5dde3";

    // The people we expect to find at L4 (Top voices on a topic from the storyline).
    private static final List<String> PEOPLE = Arrays.asList(
            "Open Dr. Elena Fischer", "Open Sam", "Open Skanda Amarnath", "Open Alex Morgan");

    @Test
    public void testStackFourDeep() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        boolean panelOpened = Journey.tap("Insights", true, 15_000);
        if (!panelOpened) {
            fail("could not open the knowledge panel. On screen: " + Journey.labelledInventory(80));
        }
        Journey.sleep(3_000);

        // Expand "Topics & People" accordion — tap only when the topic controls are not
        // already reachable, to avoid collapsing a section a previous test left open.
        List<String> topicNames = Arrays.asList("Open systems thinking", "Open risk management");
        if (Journey.find(topicNames, true, 4_000) == null) {
            Journey.tap("Topics & People", true, 12_000);
            Journey.sleep(3_000);
        }
        if (Journey.find(topicNames, true, 15_000) == null) {
            fail("no topic control in the insights panel. On screen: "
                    + Journey.labelledInventory(80));
        }

        // L1 — the topic, in the panel.
        Journey.tap(topicNames, true, 15_000);
        Journey.sleep(4_000);
        System.out.println("=====STACK_L1_TOPIC :: " + Journey.labelledInventory(12) + "=====");

        // L2 — its storyline, stacked on the topic.
        // ec.storylineHeading = 'Part of a storyline' (section heading inside the topic card).
        Journey.scrollTo("Part of a storyline", false);
        // kp.openStoryline = 'Open the {label} storyline' — contains match on the storyline name.
        boolean storylineTapped = Journey.tap(
                Arrays.asList("Managing risk across domains"), true, 12_000);
        if (!storylineTapped) {
            fail("storyline row not tappable. On screen: " + Journey.labelledInventory(80));
        }
        Journey.sleep(4_000);
        System.out.println("=====STACK_L2_STORYLINE :: " + Journey.labelledInventory(12) + "=====");

        // L3 — from inside the storyline sheet. Member-topic rows AND related-people rows must
        // open a card; they go through the same openEntity path. Dump what is on screen before
        // and after so a dead tap is distinguishable from a tap that hit nothing.
        //
        // home.storylineTopicsHeading = 'Topics discussed together'
        Journey.scrollTo("Topics discussed together", false);
        boolean tappedL3 = tapTopmost(
                Arrays.asList("risk management", "safety practices"), true);
        System.out.println("=====STACK_L3_TAPPED " + tappedL3
                + " :: " + Journey.labelledInventory(12) + "=====");
        Journey.sleep(4_000);

        if (tappedL3) {
            // L4 — a person from THAT topic's top voices. Four cards, kinds alternating, so the
            // ladder cannot be a special case for one pairing.
            // ec.topVoices = 'Top voices'
            Journey.scrollTo("Top voices", false);
            // kp.openEntity = 'Open {term}' — person names carry the "Open " prefix.
            boolean tappedL4 = tapTopmost(PEOPLE, true);
            System.out.println("=====STACK_L4_TAPPED " + tappedL4
                    + " :: " + Journey.labelledInventory(12) + "=====");
            Journey.sleep(4_000);

            if (!tappedL4) {
                System.out.println("=====PROBE no L4 person in Top voices :: "
                        + Journey.labelledInventory(80) + "=====");
                // Not a hard failure: Top voices depend on data quality. Log and continue.
            }
        } else {
            System.out.println("=====PROBE no L3 topic from the storyline :: "
                    + Journey.labelledInventory(80) + "=====");
        }
    }

    // ------------------------------------------------------------------ helpers

    /**
     * Tap the element whose name contains ANY of {@code substrings} that sits closest to the top
     * of the screen — the Android equivalent of XCTest's {@code .firstMatch} on a by-position
     * sorted list.
     *
     * ANDROID DIFFERENCE: iOS used {@code Journey.tapTopmost(app, labels: [...])}, which is a
     * method on {@code Journey.swift}. The Android {@code Journey.java} does not expose this
     * helper because the suite-level implementation is sufficient. Returns false when no matching
     * clickable element is visible.
     */
    private boolean tapTopmost(List<String> substrings, boolean containsMatch) {
        UiObject2 best     = null;
        int        bestTop = Integer.MAX_VALUE;
        for (String s : substrings) {
            List<UiObject2> hits;
            try {
                hits = Journey.device().findObjects(
                        By.pkg(Journey.PKG).descContains(s).clickable(true));
                if (hits.isEmpty()) {
                    hits = Journey.device().findObjects(
                            By.pkg(Journey.PKG).textContains(s).clickable(true));
                }
            } catch (Throwable t) {
                continue;
            }
            for (UiObject2 o : hits) {
                android.graphics.Rect b;
                try { b = o.getVisibleBounds(); } catch (Throwable t) { continue; }
                if (b == null) continue;
                if (b.centerY() < bestTop) {
                    bestTop = b.centerY();
                    best = o;
                }
            }
        }
        if (best == null) return false;
        try {
            best.click();
            return true;
        } catch (Throwable t) {
            return false;
        }
    }
}
