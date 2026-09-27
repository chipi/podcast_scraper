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
 * filtered on label. On Android, `aria-pressed` chips surface as checkable nodes in the tree and
 * do not distinguish by class in a way we can rely on. Instead, this port discovers chips by their
 * accessible NAME from the full node tree, filtering out the known chrome controls (Close, Cancel,
 * Save, etc.). This is the same intent — "tap whatever the picker actually offers" — expressed
 * through the only handle the Android tree gives us: accessible name.
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

        // RECORD WHAT IS ALREADY CHOSEN, from the card, BEFORE opening the picker.
        //
        // The picker's own chips cannot tell us. Chromium maps their `aria-pressed` to a
        // ToggleButton CLASS but exposes no state with it — measured 2026-09-26 on three chips that
        // were all currently chosen:
        //     'Show Themes' checked=false selected=false checkable=false class=ToggleButton
        // So the `isChecked()` guard that used to stand here could never fire, and every run that
        // began with interests already set toggled them OFF, saved an empty set, and then failed
        // its own empty-state assertion. That is why this test alternated pass/fail across runs:
        // `pm clear` resets the DEVICE, but interests live server-side on the account.
        //
        // The Profile card DOES render them, so read them there. Names arrive with the kind prefix
        // flattened in ("THEMEShow Themes"), hence `contains` rather than equality below.
        Set<String> alreadyChosen = new HashSet<>();
        for (UiObject2 node : Journey.device().findObjects(By.pkg(Journey.PKG))) {
            String n = Journey.nameOf(node);
            if (!n.isEmpty()) alreadyChosen.add(n);
        }

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
        System.out.println("=====INTERESTS_CHIPS " + chipLabels.subList(0, Math.min(8, chipLabels.size())) + "=====");
        assertTrue(
                "the interests picker offered nothing tappable. On screen: "
                        + Journey.labelledInventory(80),
                !chipLabels.isEmpty());

        // Tap only chips that are NOT already selected. These are toggles, so tapping a selected
        // chip DESELECTS it — running after an earlier pass left three already chosen, turning them
        // all off left the account with zero interests, and Home rightly went on prompting. The
        // test then blamed the app for its own side effect (2026-09-16 on iOS).
        //
        // SELECTED detection on Android: Chromium surfaces `aria-pressed=true` as the node being
        // checked (`isChecked()=true`) for elements that the DOM marks as checkable. This is
        // opposite to the Offline-mode checkbox (which is `checkable=false` and always reports
        // false) — the difference is that chips have `role=checkbox` in the DOM and the offline
        // input does not. So for chips only, `isChecked()` is the right read; for the settings
        // switch it is not. Use `isChecked()` here and `forcedOfflineBannerShowing()` in
        // setOfflineMode — they address different element types.
        List<String> chosen = new ArrayList<>();
        int picked = 0;
        for (String label : chipLabels) {
            if (picked >= 3) break;
            UiObject2 chip = Journey.find(label, false, 3_000);
            if (chip == null) continue;
            // A chip that is already chosen is still counted as "chosen" — we want it in the result
            // set for the final assertion — but we must NOT tap it, because tapping toggles it OFF.
            //
            // Decided from the PROFILE CARD captured before the picker opened, not from the node.
            // See the note at that capture: `aria-pressed` reaches Android as a ToggleButton class
            // with NO state attached (checked/selected/checkable all false on a chosen chip), so
            // the `isChecked()` read that used to live here was always false and this loop switched
            // off exactly the interests it was supposed to keep.
            boolean alreadySelected = false;
            for (String rendered : alreadyChosen) {
                if (rendered.contains(label)) { alreadySelected = true; break; }
            }
            chosen.add(label);
            if (!alreadySelected) {
                Journey.tap(label, false, 5_000);
                Journey.sleep(1_000);
            }
            picked++;
        }

        System.out.println("=====INTERESTS_PICKED " + picked + " " + chosen + "=====");
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
            for (UiObject2 o : Journey.device().findObjects(By.pkg(Journey.PKG).clickable(true))) {
                if (result.size() >= 40) break;
                String name = Journey.nameOf(o);
                if (name.isEmpty()) continue;
                if (CHROME.contains(name)) continue;
                if (seen.contains(name)) continue;
                // Filter out controls that are obviously navigation, not content chips —
                // anything whose name matches the bottom tab bar labels.
                if (Arrays.asList("Home", "Discover", "Library", "Profile").contains(name)) continue;
                seen.add(name);
                result.add(name);
            }
        } catch (Throwable ignored) {
            // A partial list is better than a throw; the isEmpty() check handles the empty case.
        }
        return result;
    }
}
