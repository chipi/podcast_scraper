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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
 * CHIP DISCOVERY — iOS scanned `app.switches + app.checkBoxes + app.buttons` by element type, then
 * filtered on label. Android distinguishes chips by CLASS, and this file used to say it could not.
 *
 * Measured 2026-10-02, on the picker, every node it offers:
 *
 *     macroeconomics … reliability (12 topics)   class=android.widget.ToggleButton
 *     Managing risk across domains (storyline)   class=android.widget.ToggleButton
 *     masthead, Queue, personalisationtests,     class=android.view.View
 *       Settings, Account, Stats
 *     Notifications, Change photo, Edit          class=android.widget.Button
 *
 * `aria-pressed` on the picker's `<button>` chips (InterestsPicker.vue:145-180) is what Chromium
 * maps to ToggleButton, and nothing else on the screen carries it. So class is the discriminator,
 * and the accessible-NAME approach this comment used to describe could not work: the Profile screen
 * stays in the node tree behind the sheet, its controls are clickable, and no hand-kept chrome list
 * anticipates them — the masthead, Queue and Notifications were returned as chips, ahead of the
 * real ones, and silently "chosen" without a tap.
 *
 * What the measurement CONFIRMED rather than overturned: state is unreadable. All 13 chips report
 * `checkable=false checked=false selected=false` whatever their real selection, so no `isChecked()`
 * read can tell a chosen chip from an unchosen one. That is why selection is decided from the
 * card's empty state instead, and why tapping is asserted separately from picking.
 */
@RunWith(AndroidJUnit4.class)
public class PersonalisationTests extends UITestCase {

    // Two real episodes that exist in the fixture corpus.
    private static final List<String> EPISODES = Arrays.asList("p09-a4bbb5dde3", "p07-2aceab172c");

    // Labels the interest picker renders as sheet chrome — not chips.
    private static final Set<String> CHROME = new HashSet<>(Arrays.asList(
            "Close", "Cancel", "Save", "Done", "Skip", "Topics", "Storylines",
            "Save interests", "Saving…", "Loading topics…"));

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

    // ---- 10 interests → chips + Home ----------------------------------------

    @Test
    public void test10InterestsRenderAndFeedHome() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        assertTrue("Profile did not open", Journey.openProfile(profileLabels()));
        Journey.sleep(3_000);

        assertTrue("no Topics tab on Profile. On screen: " + Journey.labelledInventory(80),
                Journey.tap("Topics", false, 12_000));
        Journey.sleep(3_000);

        // IS ANYTHING ALREADY CHOSEN? Read the CARD'S OWN EMPTY STATE, not a snapshot of the screen.
        //
        // Why this replaced a whole-screen capture (2026-10-02, measured). The chips cannot report
        // their own state: all 13 in the picker come back `checkable=false checked=false
        // selected=false`, confirming the 2026-09-26 note. So the previous version read "what is
        // already chosen" by collecting every named node under `By.pkg` — the WHOLE SCREEN — and
        // then treating a chip as already-selected when any captured name contained its label.
        // Both halves were read off the same screen, so the match always succeeded, and the
        // selection loop skipped every tap:
        //     =====INTERESTS_PRECHOSEN 19 [Account, Close Listening, Topics, Change photo, …]
        //     =====INTERESTS_PICKED 3 [Close Listening LISTEN. UNDERSTAND. REMEMBER. …, Queue,
        //                              Notifications]
        // Three "interests chosen", none of them a chip, none of them tapped. The empty set was
        // saved — `interests.json` on the account read exactly `[]` — and the test then failed its
        // own empty-state assertion and reported it as the app not persisting interests.
        //
        // The ambiguity only exists when the card is NON-empty. When it shows its empty state
        // nothing is selected, so every chip is safe to tap and no guess is needed. That is the
        // condition this reads, and `i18n: profile.noInterests = "No interests chosen yet."`.
        boolean startedEmpty = Journey.find("No interests chosen yet", true, 5_000) != null;

        // Only consulted when the card is NON-empty, where it is the card — not the chrome — that
        // renders the chosen labels. Names arrive with the kind prefix flattened in
        // ("THEMEShow Themes"), hence `contains` rather than equality at the use site.
        Set<String> alreadyChosen = new HashSet<>();
        if (!startedEmpty) {
            for (UiObject2 node : Journey.device().findObjects(By.pkg(Journey.PKG))) {
                String n = Journey.nameOf(node);
                if (!n.isEmpty()) alreadyChosen.add(n);
            }
        }
        Journey.mark("=====INTERESTS_PRECHOSEN startedEmpty=" + startedEmpty
                + " " + alreadyChosen.size() + " " + alreadyChosen + "=====");

        // i18n: profile.editInterests = "Edit"
        assertTrue(
                "no Edit control on the interests card. On screen: " + Journey.labelledInventory(80),
                Journey.tap("Edit", true, 12_000));
        Journey.sleep(4_000);

        // WAIT for the picker's content. It fetches clusters, so the first snapshot of the button
        // array can catch the sheet still on "Loading topics…" and return empty — the same
        // snapshot-before-layout trap that cost a diagnosis on the notifications matrix.
        //
        // i18n: interests.title = "Choose your interests"
        boolean pickerReady = Journey.find("Choose your interests", true, 20_000) != null;
        assertTrue("the interests picker never finished loading its topics. On screen: "
                + Journey.labelledInventory(80), pickerReady);
        Journey.sleep(2_000);

        // DISCOVER chips from the live tree.
        //
        // The iOS twin scanned by element type (switches + checkBoxes + buttons). On Android the
        // tree does not distinguish chip toggles by class reliably — they all arrive as View or
        // TextView nodes. Discovered by accessible name instead: any named, clickable node that is
        // not in the chrome set is a chip candidate. Prefering clickable nodes matches what
        // Journey.find does internally.
        List<String> chipLabels = discoverChips();
        Journey.mark("=====INTERESTS_CHIPS " + chipLabels.subList(0, Math.min(8, chipLabels.size())) + "=====");
        assertTrue(
                "the interests picker offered nothing tappable. On screen: "
                        + Journey.labelledInventory(80),
                !chipLabels.isEmpty());

        // Tap only chips that are NOT already selected. These are toggles, so tapping a selected
        // chip DESELECTS it — running after an earlier pass left three already chosen, turning them
        // all off left the account with zero interests, and Home rightly went on prompting. The
        // test then blamed the app for its own side effect (2026-09-16 on iOS).
        //
        // SELECTED detection on Android: there ISN'T any, and this is the measurement that matters
        // (2026-10-02). Every chip in the picker reports `checkable=false checked=false
        // selected=false` regardless of its real state, so `isChecked()` is not a read that can
        // work here — the comment that used to stand here claimed the opposite, that chips carry
        // `role=checkbox` and so report `isChecked()` truthfully. They do not.
        //
        // With no readable state, the only safe source is whether the CARD was empty before the
        // picker opened: empty card ⇒ nothing selected ⇒ every tap adds. When it was non-empty we
        // fall back to the card's rendered labels, which is sound now that `chipLabels` holds only
        // real chips — a chrome label can no longer collide with a chip label.
        List<String> chosen = new ArrayList<>();
        int picked = 0;
        int tapped = 0;
        for (String label : chipLabels) {
            if (picked >= 3) break;
            UiObject2 chip = Journey.find(label, false, 3_000);
            if (chip == null) continue;
            // A chip that is already chosen is still counted as "chosen" — we want it in the result
            // set for the final assertion — but we must NOT tap it, because tapping toggles it OFF.
            boolean alreadySelected = false;
            if (!startedEmpty) {
                for (String rendered : alreadyChosen) {
                    if (rendered.contains(label)) { alreadySelected = true; break; }
                }
            }
            chosen.add(label);
            if (!alreadySelected) {
                if (Journey.tap(label, false, 5_000)) tapped++;
                Journey.sleep(1_000);
            }
            picked++;
        }

        Journey.mark("=====INTERESTS_PICKED picked=" + picked + " tapped=" + tapped
                + " startedEmpty=" + startedEmpty + " " + chosen + "=====");

        // A SELECTION THAT WAS NEVER MADE MUST FAIL HERE, NOT AT THE CARD.
        //
        // This is the assertion whose absence let the defect above masquerade as an app bug for a
        // fortnight. `picked` counts candidates considered, including ones deliberately not tapped,
        // so `picked > 0` was true while `tapped` was 0 — the test saved an empty set and then
        // blamed the card for showing its empty state. When the card started empty, nothing is
        // selected yet, so at least one tap is REQUIRED for the rest of this test to mean anything.
        if (startedEmpty) {
            assertTrue(
                    "the card was empty and no chip was actually TAPPED, so an empty set is about "
                            + "to be saved and every assertion after this would be about the test's "
                            + "own side effect. picked=" + picked + " tapped=0 candidates="
                            + chipLabels.size() + " " + chipLabels,
                    tapped > 0);
        }
        assertTrue("no chip was picked (discoverChips found " + chipLabels.size() + " candidates "
                + "but none could be selected). On screen: " + Journey.labelledInventory(80),
                picked > 0);

        // Persist. i18n: interests.save = "Save" (also accept "Done" for picker variant).
        Journey.tap(Arrays.asList("Save", "Done", "Save interests"), false, 10_000);
        Journey.sleep(4_000);

        // The card must no longer show its empty state.
        // i18n: profile.noInterests = "No interests chosen yet."
        boolean noInterestsShowing = Journey.find("No interests chosen yet", true, 5_000) != null;
        assertNull(
                "interests card still shows its empty state after choosing interests. On screen: "
                        + Journey.labelledInventory(80),
                noInterestsShowing ? Journey.find("No interests chosen yet", true, 1_000) : null);

        // Interests feed Home's recommendations, so Home must REFLECT them — the first cut of this
        // test only screenshotted Home afterwards, which proves nothing.
        Journey.openTab("Home");
        Journey.sleep(6_000);

        // i18n: interests.cardCta = "Choose interests" — Home stops showing this once the user has
        // chosen. This is the honest Home-side consequence; matching a cluster label on Home would
        // test a relationship the product does not claim (cluster → rail is an internal mapping).
        boolean stillPrompting = Journey.find("Choose interests", true, 5_000) != null;
        assertNull(
                "Home is still prompting to choose interests after interests were chosen. On screen: "
                        + Journey.labelledInventory(80),
                stillPrompting ? Journey.find("Choose interests", true, 1_000) : null);

        // Where the chosen labels DO render verbatim is the Profile Topics tab.
        assertTrue("Profile did not open", Journey.openProfile(profileLabels()));
        Journey.sleep(3_000);
        Journey.tap("Topics", false, 10_000);
        Journey.sleep(3_000);

        // At least one of the chosen chip labels must appear on the Topics tab.
        //
        // CONTAINS, not exact (2026-09-26). The chip renders a KIND prefix beside the label, and
        // Android flattens the two into one node with NO separator, while iOS keeps them as
        // separate elements — so an exact match sees nothing while the interests are plainly
        // rendered. Measured:
        //     THEMEShow Themes[TextView]
        //     THEMELifelong Learning[TextView]
        //     Open Managing risk across domains[Button,click]
        // Note the third: a storyline chip is a BUTTON whose name is prefixed "Open …", so exact
        // matching could never have found that one either.
        UiObject2 rendered = Journey.scrollTo(chosen, true, 20);
        assertNotNull(
                "none of the interests just chosen (" + chosen + ") render on the Profile Topics "
                        + "tab. On screen: " + Journey.labelledInventory(80),
                rendered);
    }

    /**
     * Collect accessible names of interest chip candidates from the live tree.
     *
     * A chip is any named, clickable node whose label is not in {@link #CHROME} and is not empty.
     * The tree walk is bounded by a limit to keep it fast; 40 is enough to cover every picker that
     * shipped so far and none of them needed more than 20 chips.
     */
    private List<String> discoverChips() {
        List<String> result = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        try {
            // BY CLASS, because the picker's chips are the only ToggleButtons on the screen
            // (2026-10-02, measured — see CHIP_DISCOVERY on the class doc). The predicate used to
            // be `clickable(true)` minus a hand-kept CHROME set, which cannot work: the Profile
            // screen sits behind the sheet with its nodes still in the tree, and the masthead,
            // Queue, Notifications, Change photo, Settings, Account and Stats are all clickable and
            // none of them are in any chrome list anyone would think to write. They were returned
            // as chip candidates, ahead of the real chips, in tree order.
            for (UiObject2 o : Journey.device().findObjects(
                    By.pkg(Journey.PKG).clazz("android.widget.ToggleButton"))) {
                if (result.size() >= 40) break;
                String name = Journey.nameOf(o);
                if (name.isEmpty()) continue;
                // CHROME is still consulted. Scoping by class already excludes every control the
                // set names, so this is belt-and-braces against a future chrome control that
                // happens to carry `aria-pressed` — a segmented filter, say.
                if (CHROME.contains(name)) continue;
                if (seen.contains(name)) continue;
                seen.add(name);
                result.add(name);
                // MEASUREMENT, not a filter (2026-10-02). This class's two statements about chip
                // identity contradict each other — the header says class cannot be relied on, the
                // note at the selection loop records `class=ToggleButton` measured on three chips.
                // The fix for this test depends on which is true, so record the fields rather than
                // choose. Through `Journey.mark`, because `System.out` from instrumentation does
                // not reach `am instrument -w` and the two markers below were invisible for it.
                recordCandidate(o, name);
            }
        } catch (Throwable ignored) {
            // A partial list is better than a throw; the isEmpty() check handles the empty case.
        }
        return result;
    }

    /** One line per chip candidate: the fields that could tell a chip from the screen behind it. */
    private void recordCandidate(UiObject2 o, String name) {
        try {
            Journey.mark("=====CHIP_PROBE '" + name + "'"
                    + " class=" + o.getClassName()
                    + " checkable=" + o.isCheckable()
                    + " checked=" + o.isChecked()
                    + " selected=" + o.isSelected()
                    + " bounds=" + o.getVisibleBounds().toShortString()
                    + "=====");
        } catch (Throwable ignored) {
            // A probe that throws must never decide the test.
        }
    }
}
