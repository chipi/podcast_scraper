package app.closelistening.player;

import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiObject2;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Every control a listener can tap must be REACHABLE BY NAME to assistive technology (#2139).
 *
 * ## Why this exists rather than another one-off fix
 *
 * Three separate defects this week turned out to be one class — the accessible name is not what
 * `aria-label` says — and each was found by accident, by some other test happening to hunt that one
 * control:
 *
 *  1. **Pruned entirely (WebKit).** The masthead profile link carried an `aria-label`, but its only
 *     child was an `aria-hidden` avatar, so WebKit dropped the whole LINK from the tree. Visible on
 *     screen, present in the DOM, findable by Playwright, unreachable to VoiceOver.
 *  2. **Present but unnamed (Chromium).** `aria-haspopup` plus a fully hidden subtree leaves the
 *     node in the tree with an EMPTY name — the overflow ⋯, Notifications, Share and Add to
 *     collection all announced as just "Button".
 *  3. **Named by its GLYPH (Chromium).** `FavoriteButton` renders `♡` as text beside an
 *     `aria-label` of "Add to favourites", and the tree carries `♡`. A screen reader announces a
 *     symbol. So does every test that looks the control up by name — which is exactly why an
 *     "Add to queue" control that is plainly on screen could not be found.
 *
 * None of these is visible to a screenshot (the control is drawn), to a DOM query (the element
 * exists), or to Playwright (which reads the DOM). Only the device tier can see them, and only if
 * something looks. This is the something.
 *
 * ## What it asserts
 *
 * A FLOOR, not an inventory: every CLICKABLE node must have a name that a person could act on.
 * Two ways to fail — no name at all, and a name made only of symbols or punctuation. Asserting an
 * exact set would make every UI change a test edit and the suite would be re-baselined rather than
 * read.
 *
 * Deliberately NOT asserted: element TYPE. `aria-haspopup` maps to one thing and `aria-pressed` to
 * another, both correct; pinning types would fail on correct markup.
 */
@RunWith(AndroidJUnit4.class)
public class AccessibleNameAuditTests extends UITestCase {

    /** Reads the account the download suite seeds, so the surfaces have real content on them. */
    @Override
    protected String accountIdentity() {
        return SHARED_SEEDED_IDENTITY;
    }

    /**
     * A name is USABLE when it contains at least one letter or digit.
     *
     * `♡` and `⋯` are names in the technical sense and useless in every sense that matters: a
     * screen reader announces a symbol, and a test cannot address the control by intent.
     */
    private static boolean usable(String name) {
        for (char c : name.toCharArray()) {
            if (Character.isLetterOrDigit(c)) return true;
        }
        return false;
    }

    /** Every clickable on the current screen whose name a person could not act on. */
    private List<String> unusableOn(String surface) {
        List<String> bad = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        try {
            for (UiObject2 o : Journey.device().findObjects(By.pkg(Journey.PKG).clickable(true))) {
                String name = Journey.nameOf(o);
                if (usable(name)) continue;
                String cls = String.valueOf(o.getClassName());
                cls = cls.substring(cls.lastIndexOf('.') + 1);
                // Bounds included: two unnamed controls are otherwise indistinguishable in the
                // report, and "which one" is the first thing anyone fixing this needs.
                String entry = (name.isEmpty() ? "<NO NAME>" : "<SYMBOL-ONLY '" + name + "'>")
                        + " " + cls + " " + o.getVisibleBounds();
                if (seen.add(entry)) bad.add(entry);
            }
        } catch (Throwable t) {
            // A diagnostic that throws replaces the finding with its own failure.
            bad.add("<audit of " + surface + " threw " + t + ">");
        }
        return bad;
    }

    @Test
    public void everyTappableControlHasANameAPersonCanActOn() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity(), ready);

        List<String> findings = new ArrayList<>();

        for (String tab : Arrays.asList("Home", "Discover", "Library")) {
            if (!Journey.openTab(tab)) {
                findings.add("[" + tab + "] the tab itself was unreachable");
                continue;
            }
            Journey.sleep(3_000);
            for (String bad : unusableOn(tab)) findings.add("[" + tab + "] " + bad);
        }

        // The player is where the most icon-only controls live, and where all three defects above
        // were found.
        AppSession.openEpisode("p06-7217050bc6");
        Journey.sleep(5_000);
        for (String bad : unusableOn("player")) findings.add("[player] " + bad);

        // The overflow, open — its items are only in the tree while the panel is rendered.
        if (Journey.tap("More actions", false, 15_000)) {
            Journey.sleep(2_000);
            for (String bad : unusableOn("player ⋯")) findings.add("[player ⋯] " + bad);
        }

        assertTrue(
                "These controls are tappable but carry no name a person could act on, so a screen "
                        + "reader announces nothing useful and no test can address them by intent:\n  "
                        + String.join("\n  ", findings)
                        + "\n\nFix by giving the control a real accessible name — an `sr-only` span "
                        + "inside it, the pattern already used for the masthead profile link. An "
                        + "`aria-label` ALONE is not enough: it is dropped when the subtree is fully "
                        + "`aria-hidden` and overridden when the element has visible glyph text.",
                findings.isEmpty());
    }
}
