package app.closelistening.player;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
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
 * Native journey suite (#2139, porting the iOS suite of the same name).
 *
 * Scope, in the order the operator asked for it:
 *   01  profile tabs (Account / Topics / Stats)
 *   02  episode → insights
 *   03  topic page → storyline page
 *   04  person page, reached from an episode's entities
 *   05  collections: create, then "Add to board" lists them
 *   06  share popover renders its options
 *   07  saved-item colour picker popover renders and a colour can be chosen
 *   11  storyline from the Insights panel stacks over the topic (canLayer=false path)
 *
 * Tests are numbered so they describe the sequence. Methods in this class are NOT expected to be
 * run in alphabetical order — JUnit 4 does NOT guarantee ordering, so each test navigates
 * from scratch via {@link #startClean()}.
 *
 * PRECONDITIONS: app installed, signed in via the dev picker, fixture api reachable.
 * Fixture slugs and titles are from the committed v3 corpus.
 */
@RunWith(AndroidJUnit4.class)
public class AppJourneyTests extends UITestCase {

    // An episode with 5 insights — picked from the fixture corpus so the insights assertions
    // test rendering, not an empty state.
    private static final String EPISODE_SLUG  = "p09-a4bbb5dde3";
    private static final String EPISODE_TITLE = "Risk Is a Systems Property";

    // One fixture episode per colour token. The Saved colour filter offers only colours actually in
    // use, so seeding one item per token is what makes the filter reviewable at all (operator
    // 2026-09-16).
    private static final String[][] COLOUR_SEEDS = {
        { "p09-a4bbb5dde3", "Amber"   },
        { "p09-6ac0bf4914", "Rose"    },
        { "p08-72169222b1", "Sky"     },
        { "p08-bd7cc798ff", "Emerald" },
        { "p07-2aceab172c", "Violet"  },
    };

    // ------------------------------------------------------------------ 01 profile

    @Test
    public void test01ProfileTabs() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        assertTrue("could not open Profile from the masthead avatar",
                Journey.openProfile(profileLabels()));
        Journey.sleep(3_000);

        for (String tab : Arrays.asList("Interests", "Stats")) {
            boolean tapped = Journey.tap(tab, false, 12_000);
            if (tapped) {
                Journey.sleep(3_000);
            } else {
                fail("profile tab '" + tab + "' not tappable. On screen: "
                        + Journey.labelledInventory(80));
            }
        }
    }

    // ------------------------------------------------------------------ 02 episode + insights

    @Test
    public void test02EpisodeAndInsights() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        assertNotNull("episode page did not render its title. On screen: "
                + Journey.labelledInventory(16),
                Journey.find(EPISODE_TITLE, true, 20_000));

        // The Episode notes panel — entry point is "✦ Episode notes" (kp.title). The ✦ is a
        // separate text node on iOS but on Android the whole label lands as one string, so
        // contains is the safe match.
        boolean tapped = Journey.tap("Episode notes", true, 15_000);
        if (tapped) {
            Journey.sleep(4_000);
        } else {
            Journey.scrollTo("Episode notes", true);
            Journey.sleep(2_000);
        }
        // The panel's own title (kp.title = 'Episode notes') must now be on screen. The opener pill
        // is v-if'd away while the panel is open, so this match can only be the panel.
        assertNotNull("the Episode notes panel title did not appear. On screen: "
                + Journey.labelledInventory(80),
                Journey.find("Episode notes", true, 10_000));
    }

    // ------------------------------------------------------------------ 03 topic + storyline

    @Test
    public void test03TopicAndStoryline() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        // Topic ids carry a `topic:` prefix that the deep-link validator rejects by design, so
        // topics are reached the way a user reaches them: the Home entity rail's Topics tab.
        Journey.tap("Topics", false, 15_000);
        Journey.sleep(3_000);

        // Any topic row from the rail — "systems thinking" and "risk management" are in the
        // p09 fixture but the exact ranking changes with the corpus.
        boolean topicTapped = Journey.tap(
                Arrays.asList("systems thinking", "risk management"), true, 12_000);
        if (topicTapped) {
            Journey.sleep(5_000);
        } else {
            fail("no topic row was tappable from the Home rail. On screen: "
                    + Journey.labelledInventory(80));
        }

        // Storylines: back to Home, switch the rail to Storylines, open the first openable one.
        // The topic tap above opens the topic CARD, which renders over the bottom tab bar. Dismiss
        // it before asking for a tab, or the tap lands on the card and the failure blames the tab
        // (measured 2026-09-26: "TOPIC | Close | systems thinking | Follow — systems thinking | …"
        // on screen while this asserted "Home tab did not open").
        Journey.dismissCards();
        // SAY WHAT IS ON SCREEN. This assertion had no inventory, so its failure named the tab it
        // could not find and nothing about why.
        assertTrue("Home tab did not open. On screen: " + Journey.labelledInventory(40),
                Journey.openTab("Home"));
        Journey.sleep(4_000);

        // BRING THE TRENDS RAIL INTO VIEW BEFORE ASKING FOR ITS TAB (2026-09-27).
        //
        // Home's LENGTH is a function of account state. With history the page grows a "Continue
        // listening" hero, a "Jump back in" row and a filled "Your Week" — measured on this very
        // failure: "CONTINUE LISTENING | … | 4 IN PROGRESS | Jump back in | … | 4 NEW | Your Week"
        // — and that pushes the Trends rail off screen. Android's tree holds only ON-SCREEN nodes,
        // so the tab is then ABSENT, not merely far down, and the tap fails naming the tab.
        //
        // The tier's `pm clear` resets the DEVICE, not the server-side ACCOUNT, so this test sees a
        // short Home on a fresh account and a long one after anything has played — which is why it
        // passed all day and then failed deterministically once the account had been used.
        Journey.scrollTo("Trends", false);
        boolean storylinesTab = Journey.tap("Storylines", false, 15_000);
        if (!storylinesTab) {
            fail("Storylines tab not found on the Home rail. On screen: "
                    + Journey.labelledInventory(80));
        }
        Journey.sleep(4_000);

        // Storyline rows: aria-label = "{label} ({count}) — {N}× momentum" (DiscoveryList
        // rowLabel). The iOS suite filtered by `contains("momentum") && contains("(")` — the
        // SAME strings appear here. Pick the topmost openable one.
        //
        // ANDROID DIFFERENCE: XCTest `.buttons.allElementsBoundByIndex.filter` is not available.
        // Instead, find all controls whose name contains "momentum" and tap the one closest to
        // the top of the screen (tapTopmost). The visible set is the same because DiscoveryList
        // renders these as `<button>` with the same accessible name pattern on both platforms.
        // SCROLL THE RAIL INTO VIEW FIRST. Android's accessibility tree contains only ON-SCREEN
        // nodes, so a rail below the fold is not merely hard to reach — it is absent, and
        // `tapTopmost` has nothing to enumerate. Measured 2026-09-26: the dump at this point ended
        // at "Trends", well above the storyline rows, on a Home that was rendering them.
        // WAIT FOR THE ROWS; DO NOT SCROLL FOR THEM (2026-09-27).
        //
        // `scrollTo` is the wrong instrument for a list that arrives over the network. It gives up
        // EARLY BY DESIGN: its stall detector breaks out once two consecutive swipes leave the page
        // signature unchanged, which on a Home this short is about four seconds — well before
        // `/api/app/trending` has answered. Its final probe is a further 1.5s, so the whole thing
        // concedes in ~5s and the rows land just after.
        //
        // That is why the failure was deterministic (2/2 probes) AND why the dump attached to it
        // contained the very row being looked for:
        //     Managing risk across domains (4) — 0.4× momentum[Button,click]
        // Present at dump time, absent while scrollTo was asking.
        //
        // `find` polls the whole tree until its deadline, so it covers both the fetch and the
        // render. Scroll only as a fallback, for a Home long enough to push the rail off screen.
        // Same lazy-render trap StackDepthProbeTests records for the insights accordion, where the
        // fix was likewise "a LONG final wait" rather than more swiping.
        // SCROLL *AND* WAIT, alternately. Neither alone works here.
        //
        // `scrollTo` probes 1.5s per swipe and gives up once two swipes leave the page signature
        // unchanged — too brief for rows that arrive from `/api/app/trending`. A plain `find`, even
        // a 40s one, fails differently: Android's tree holds only ON-SCREEN nodes, so waiting does
        // nothing while the list is below the fold. And `scrollTo("Trends")` returns the moment the
        // HEADING enters the tree, which can leave the rows beneath it still off screen — that is
        // what defeated the previous attempt here.
        //
        // So: probe generously, swipe, repeat. Covers a slow fetch and a long Home at once.
        UiObject2 storylineRow = null;
        for (int i = 0; i < 12 && storylineRow == null; i++) {
            storylineRow = Journey.find("momentum", true, 4_000);
            if (storylineRow == null) {
                Journey.swipeUp();
                Journey.sleep(800);
            }
        }
        if (storylineRow == null) {
            fail("no storyline row on Home after 12 scroll-and-wait rounds. On screen: "
                    + Journey.labelledInventory(80));
        }
        boolean storylineTapped = tapTopmost(Arrays.asList("momentum"), true);
        if (!storylineTapped) {
            fail("storyline rows are on screen but none was tappable. On screen: "
                    + Journey.labelledInventory(80));
        }
        Journey.sleep(5_000);

        // A storyline opens as a sheet; "Follow storyline" / "Following storyline" (ec.followStoryline
        // / ec.followingStoryline) exists ONLY on that sheet. The "Topics discussed together" heading
        // (home.storylineTopicsHeading) and "Open in page" also exist only there.
        assertNotNull(
                "the storyline sheet did not open. On screen: " + Journey.labelledInventory(80),
                Journey.find(Arrays.asList(
                        "Follow storyline", "Following storyline",
                        "Topics discussed together", "Open in page"),
                        true, 10_000));
    }

    // ------------------------------------------------------------------ 04 person from episode

    @Test
    public void test04PersonFromEpisode() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        // Three steps required before the person button is reachable:
        //   1. Open the knowledge panel (closed on a fresh launch) via the Episode notes entry.
        //   2. Expand the "Topics & People" accordion (kp.tags) — collapsed by default, only
        //      section headers in the tree until opened.
        //   3. Scroll the person control into view — offscreen web content is not in the tree.
        Journey.tap("Episode notes", true, 15_000);
        Journey.sleep(3_000);

        // The accordion REMEMBERS its state between tests. Tap only when the person controls
        // are not already reachable, and re-tap once if the first tap collapsed a section a
        // previous test left open (same trap as test04 on iOS, 2026-09-16).
        List<String> personNames = Arrays.asList("Open Dr. Elena Fischer", "Open Sam");
        if (Journey.find(personNames, true, 4_000) == null) {
            Journey.tap("Topics & People", true, 15_000);
            Journey.sleep(3_000);
            if (Journey.find(personNames, true, 6_000) == null) {
                Journey.tap("Topics & People", true, 10_000);
                Journey.sleep(3_000);
            }
        }

        // POLL: expanding the accordion lays the section out asynchronously.
        if (Journey.find(personNames, true, 20_000) == null) {
            Journey.scrollTo(personNames, true, 10);
        }

        boolean opened = Journey.tap(personNames, true, 15_000);
        if (!opened) {
            fail("no 'Open <person>' control on the episode. On screen: "
                    + Journey.labelledInventory(80));
        }
        Journey.sleep(5_000);

        // Guard against the false pass: a person page has no episode transport controls.
        // (i18n: player.skipForward or similar transport labels would only be present if we were
        // still on the player.)
        assertNull(
                "still on the player — tapping the person did not navigate to a person page. "
                        + "On screen: " + Journey.labelledInventory(16),
                Journey.find("Skip forward 30 seconds", false, 3_000));
    }

    // ------------------------------------------------------------------ 05 collections

    @Test
    public void test05CollectionsCreateAndAdd() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        assertTrue("Library tab did not open", Journey.openTab("Library"));
        Journey.sleep(4_000);

        // Collections live on the Library tab called "Boards" (library.collections = 'Boards').
        // NOT "Collections" — that is the API name and matches nothing on screen.
        boolean boardsTab = Journey.tap("Boards", false, 12_000);
        if (!boardsTab) {
            fail("no Boards tab in Library. On screen: " + Journey.labelledInventory(80));
        }
        Journey.sleep(3_000);

        for (String name : Arrays.asList("Test Board A", "Test Board B")) {
            // The name field: collections.namePlaceholder = 'New collection name'. On Android the
            // placeholder lands as the contentDescription on the EditText.
            UiObject2 field = Journey.find("New collection name", true, 12_000);
            if (field == null) {
                // Fallback: the text field without a name — only present when no named field exists.
                field = Journey.device().findObject(
                        By.pkg(Journey.PKG).clazz("android.widget.EditText"));
            }
            if (field == null) {
                fail("no collection-name field. On screen: " + Journey.labelledInventory(80));
                break;
            }
            try {
                field.click();
                field.setText(name);
            } catch (Throwable t) {
                fail("could not type into the collection-name field: " + t
                        + ". On screen: " + Journey.labelledInventory(16));
                break;
            }
            // collections.create = 'Create'
            boolean created = Journey.tap("Create", false, 8_000);
            if (!created) {
                fail("Create button not tappable for " + name + ". On screen: "
                        + Journey.labelledInventory(80));
                break;
            }
            Journey.sleep(3_000);
        }

        // Now the "Add to board" sheet on an episode must LIST those collections.
        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        // collections.addTo = 'Add to board'
        boolean addTapped = Journey.tap("Add to board", true, 15_000);
        if (addTapped) {
            Journey.sleep(3_000);
            assertNotNull(
                    "the Add-to-collection sheet did not list the collections that exist. "
                            + "On screen: " + Journey.labelledInventory(80),
                    Journey.find("Test Board A", true, 10_000));
        } else {
            fail("'Add to board' not reachable from the episode. On screen: "
                    + Journey.labelledInventory(80));
        }
    }

    // ------------------------------------------------------------------ 06 share

    @Test
    public void test06SharePopover() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        // share.open = 'Share'
        boolean shareTapped = Journey.tap("Share", true, 15_000);
        if (shareTapped) {
            Journey.sleep(3_000);
            // The popover offers card / link / text (share.card / share.link / share.text);
            // at least one must render, or nothing opened.
            assertNotNull(
                    "share popover did not render its options. On screen: "
                            + Journey.labelledInventory(80),
                    Journey.find(
                            Arrays.asList("Share card", "Share link", "Share text"),
                            true, 10_000));
        } else {
            fail("Share control not reachable from the episode. On screen: "
                    + Journey.labelledInventory(80));
        }
    }

    // ------------------------------------------------------------------ 07 saved colour picker

    @Test
    public void test07SavedColourPicker() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        // Seed every colour except the first (which is applied to the main episode below).
        // Saved is newest-first, so the episode favourited last is row one — each seed's colour
        // can be applied to the FIRST colour control without addressing a specific row.
        for (int i = 1; i < COLOUR_SEEDS.length; i++) {
            String slug   = COLOUR_SEEDS[i][0];
            String colour = COLOUR_SEEDS[i][1];

            AppSession.openEpisode(slug);
            Journey.sleep(5_000);
            // fav.add = 'Save' — EXACT: a substring match also hits the bookmark ("Save line")
            if (Journey.find("Save", false, 8_000) != null) {
                Journey.tap("Save", false, 8_000);
                Journey.sleep(3_000);
            }
            assertTrue("Library tab did not open (seeding " + colour + ")",
                    Journey.openTab("Library"));
            Journey.sleep(3_000);
            // library.saved = 'Saved'
            Journey.tap("Saved", true, 10_000);
            Journey.sleep(2_000);
            // highlights.colorPick = 'Colour' — EXACT match so we do not also hit
            // library.savedFilterColor = 'Filter by colour' (same trap as iOS 2026-09-16).
            // BOTH RESULTS CHECKED (2026-09-26). These were discarded, so a popover that never
            // opened — or a swatch tap that missed — failed silently here and surfaced four
            // episodes later as "the Saved colour filter offers 0 colour(s)", which blames the
            // seed rather than naming the step. Same blindness as a discarded `openTab`.
            boolean opened = Journey.tap("Colour", false, 10_000) || Journey.tap("Color", false, 2_000);
            assertTrue("the colour control did not open while seeding " + colour
                    + ". On screen: " + Journey.labelledInventory(60), opened);
            Journey.sleep(2_000);
            // highlights.setColor = 'Set colour: {color}'
            boolean picked = Journey.tap("Set colour: " + colour, true, 8_000);
            assertTrue("no swatch named 'Set colour: " + colour + "' in the open popover. "
                    + "On screen: " + Journey.labelledInventory(60), picked);
            Journey.sleep(2_000);
        }

        // Favourite the main episode so Saved has something to colour-code.
        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);
        // Favourite only if not ALREADY favourited — a blind tap toggles and would leave Saved
        // empty (iOS 2026-09-16).
        if (Journey.find("Save", false, 8_000) != null) {
            Journey.tap("Save", false, 8_000);
            Journey.sleep(3_000);
        }
        // fav.remove = 'Remove from Saved'
        assertNotNull(
                "the episode is not favourited, so Saved would be empty. On screen: "
                        + Journey.labelledInventory(16),
                Journey.find("Remove from Saved", true, 10_000));

        assertTrue("Library tab did not open", Journey.openTab("Library"));
        Journey.sleep(4_000);
        Journey.tap("Saved", true, 12_000);
        Journey.sleep(3_000);

        // The colour control opens a popover of named colours. Exact match on 'Colour' / 'Color'
        // — see the seeding loop above.
        boolean colourTapped = Journey.tap("Colour", false, 12_000)
                || Journey.tap("Color", false, 2_000);
        if (colourTapped) {
            Journey.sleep(3_000);
            UiObject2 swatch = Journey.find(
                    Arrays.asList("Amber", "Rose", "Sky", "Emerald", "Violet"),
                    true, 10_000);
            if (swatch == null) {
                // 20 items got as far as the masthead and the Library tabs — nowhere near a popover
                // (2026-09-26). Same lesson as the insights dump: raise the ceiling or the
                // diagnostic describes the wrong part of the screen.
                fail("colour popover rendered no colour choices. On screen: "
                        + Journey.labelledInventory(80));
            }
            try {
                swatch.click();
            } catch (Throwable t) {
                fail("could not tap the colour swatch: " + t);
            }
            Journey.sleep(3_000);
        } else {
            fail("colour control not reachable from Saved. On screen: "
                    + Journey.labelledInventory(80));
        }

        // The FILTER (library.savedFilterColorOnly = 'Only {color}') offers only colours in use.
        // Asserting count > 1 stops this silently collapsing back to a single swatch (2026-09-16).
        assertTrue("Library tab did not open", Journey.openTab("Library"));
        Journey.sleep(3_000);
        Journey.tap("Saved", true, 12_000);
        Journey.sleep(3_000);

        int offeredCount = 0;
        for (String[] seed : COLOUR_SEEDS) {
            // library.savedFilterColorOnly = 'Only {color}' — e.g. "Only Amber"
            if (Journey.find("Only " + seed[1], true, 2_000) != null) offeredCount++;
        }
        assertTrue(
                "the Saved colour filter offers " + offeredCount + " colour(s) — the seed "
                        + "coloured too few items. On screen: " + Journey.labelledInventory(80),
                offeredCount > 1);
    }

    // ------------------------------------------------------------------ 11 storyline from Insights

    /**
     * Inside the Knowledge Panel the entity card is INLINE, so the layering policy says a sheet
     * must not stack over it — the panel is already the layer. Topics replace in place via the
     * shell's back stack; a storyline has no back-stack equivalent, so it routes to its standalone
     * page instead.
     *
     * The assertion is that the storyline sheet IS visible AND the topic is still identifiable
     * by its title below it (stacking keeps it; routing away destroys it). ANDROID DIFFERENCE:
     * same contract, same labels — the layering logic is in the Vue components, not in native code,
     * so the observable is the same on both platforms.
     */
    @Test
    public void test11StorylineFromInsightsStacksOverTheTopic() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity() + ". On screen: "
                + Journey.labelledInventory(80), ready);

        AppSession.openEpisode(EPISODE_SLUG);
        Journey.sleep(6_000);

        boolean panelOpened = Journey.tap("Episode notes", true, 15_000);
        if (!panelOpened) {
            fail("could not open the knowledge panel. On screen: " + Journey.labelledInventory(80));
        }
        Journey.sleep(3_000);

        // Accordion state is preserved between tests — tap only when the topic controls are not
        // already reachable, and re-tap once if the first tap collapsed a section.
        List<String> topicNames = Arrays.asList("Open systems thinking", "Open risk management");
        if (Journey.find(topicNames, true, 4_000) == null) {
            // SCROLL to the section first — this is a real iOS/Android difference, not a port slip.
            // Android's accessibility tree contains only ON-SCREEN nodes, where iOS keeps off-screen
            // ones with negative coordinates. So `find` cannot see "Topics & People" while the panel
            // is scrolled to the top, and `tap` never had anything to aim at. Measured 2026-09-26:
            // an 80-item inventory of the open panel returned 19 nodes and ended inside the
            // key-points list — the section simply was not in the tree.
            if (Journey.scrollTo("Topics & People", true) == null) {
                fail("the Topics & People section is not in the insights panel at all. On screen: "
                        + Journey.labelledInventory(80));
            }
            Journey.tap("Topics & People", true, 12_000);
            Journey.sleep(3_000);
            if (Journey.find(topicNames, true, 6_000) == null) {
                Journey.tap("Topics & People", true, 10_000);
                Journey.sleep(3_000);
            }
        }
        if (Journey.find(topicNames, true, 20_000) == null) {
            // 20 was not enough to reach the Topics & People section at all — the dump ended in the
            // key-points list and said nothing about the controls under test (2026-09-26). A
            // diagnostic that stops before the thing it is diagnosing is worse than none: it looks
            // like evidence.
            fail("no topic control in the insights panel. On screen: "
                    + Journey.labelledInventory(80));
        }

        // L1 — drill into a topic IN THE PANEL.
        Journey.tap(topicNames, true, 15_000);
        Journey.sleep(4_000);

        // The topic card inside the panel offers its storyline: a link card with no heading since
        // 2026-10-05 ("Part of a storyline" is gone) — scroll to the storyline's own name, then open it.
        Journey.scrollTo("Managing risk across domains", true);
        boolean storylineTapped = Journey.tap(
                Arrays.asList("Managing risk across domains"), true, 12_000);
        if (!storylineTapped) {
            fail("storyline row not tappable in the panel. On screen: "
                    + Journey.labelledInventory(80));
        }
        Journey.sleep(5_000);

        // Assert on something ONLY a storyline renders. The first version checked for
        // "Managing risk across domains" and passed while nothing had opened — that string is the
        // label of the storyline link card inside the topic card itself.
        // ec.followStoryline = 'Follow storyline' / ec.followingStoryline = 'Following storyline'
        UiObject2 followBtn = Journey.find(
                Arrays.asList("Follow storyline", "Following storyline"), true, 12_000);
        assertNotNull(
                "no storyline sheet on screen — the storyline never opened, or it opened behind "
                        + "the panel's top layer. On screen: " + Journey.labelledInventory(80),
                followBtn);

        // And the topic underneath must STILL be identifiable by its title — stacking keeps it,
        // routing away destroys it.
        assertNotNull(
                "the topic is gone — the storyline replaced it instead of stacking over it. "
                        + "On screen: " + Journey.labelledInventory(80),
                Journey.find(
                        Arrays.asList("systems thinking", "risk management"), true, 8_000));
    }

    // ------------------------------------------------------------------ helpers

    /**
     * Tap the element whose name contains ANY of {@code substrings} that sits CLOSEST TO THE TOP
     * of the screen (i.e. has the smallest visible y-centre).
     *
     * This replaces XCTest's `.buttons.allElementsBoundByIndex.filter { }.first`, which sorts by
     * tree order. On Android UI Automator the tree order does not guarantee top-to-bottom, so the
     * explicit position sort is required.
     *
     * Returns false when no matching, clickable element is found.
     */
    /**
     * ENUMERATES. Do not put a `BySelector` back here (2026-09-26).
     *
     * This used `By.pkg(PKG).descContains(s)` with a `textContains` fallback, and BOTH are blind to
     * what it is looking for. `By.desc` DOES NOT MATCH WEBVIEW CONTENT — the measurement that
     * opened this whole arc, recorded at length on {@link Journey#find} — and the fallback cannot
     * help because these rows are named by `aria-label`, which arrives as a contentDescription and
     * never as text.
     *
     * So this could never find a storyline row, and said "no storyline row was tappable" about a
     * rail that was rendering them. `Journey.find` and `AppSession.waitForField` were converted to
     * enumeration when that was discovered; this one was missed, and nothing ran it until tonight.
     * `StackDepthProbeTests` calls it too.
     */
    private boolean tapTopmost(List<String> substrings, boolean requiresContains) {
        UiObject2 best     = null;
        int        bestTop = Integer.MAX_VALUE;
        java.util.List<UiObject2> all;
        try {
            all = Journey.device().findObjects(By.pkg(Journey.PKG));
        } catch (Throwable t) {
            return false;
        }
        for (UiObject2 o : all) {
            Boolean clickable = Journey.attr(o, UiObject2::isClickable);
            if (!Boolean.TRUE.equals(clickable)) continue;
            String name = Journey.nameOf(o);
            if (name.isEmpty()) continue;
            boolean matches = false;
            for (String s : substrings) {
                if (requiresContains
                        ? name.toLowerCase().contains(s.toLowerCase())
                        : name.equalsIgnoreCase(s)) {
                    matches = true;
                    break;
                }
            }
            if (!matches) continue;
            android.graphics.Rect b;
            try { b = o.getVisibleBounds(); } catch (Throwable t) { continue; }
            if (b == null) continue;
            if (b.centerY() < bestTop) {
                bestTop = b.centerY();
                best = o;
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
