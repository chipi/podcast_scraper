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
        // THE PLATFORM'S OWN VIEW, read in the SAME pass (2026-09-27, #2156).
        //
        // `UiObject2` offers only `getText()` and `getContentDescription()`, and Android keeps
        // names in three more places: `hintText`, `stateDescription`, and `labeledBy` — a POINTER
        // to the node holding the words, which is how `<label><span>Offline mode</span><input></label>`
        // names its input without copying the string onto it. A control named that way looks
        // nameless to the loop below and is not.
        //
        // Same pass, never a dump taken afterwards: the key is screen bounds and a surface does not
        // have to be at the same scroll position a moment later.
        java.util.Map<String, A11yProbe.Named> probe = A11yProbe.actionableByBounds();
        // ONE STALE NODE MUST NOT BLIND THE WHOLE SURFACE (2026-09-27).
        //
        // Every accessor re-resolves the node, so a node the page re-rendered under us throws
        // `StaleObjectException` mid-loop. That used to escape to the outer catch, which replaced
        // every finding on the surface with a single "<audit threw …>" line — so a surface that
        // churns while being walked reported one noisy non-finding instead of its real ones.
        // Measured on `Profile ▸ Topics`, which hydrates its chips after first paint: 2 runs in 6.
        //
        // A node that went stale is one this pass did not judge, so it is COUNTED rather than
        // ignored: if enough of them go, the surface was not meaningfully audited and saying so is
        // the honest result. Silently skipping would make a churning surface look clean, which is
        // the failure this whole file exists to catch.
        int stale = 0;
        int examined = 0;
        try {
            for (UiObject2 o : Journey.device().findObjects(By.pkg(Journey.PKG).clickable(true))) {
                examined += 1;
                try {
                String name = Journey.nameOf(o);
                if (usable(name)) continue;

                // Before believing the finding, ask the platform. A node the probe can see AND
                // holds a usable name for is correctly named — the harness simply cannot reach it.
                A11yProbe.Named platform = A11yProbe.at(probe, o.getVisibleBounds());
                if (platform != null && usable(platform.best())) continue;
                String kind = String.valueOf(o.getClassName());
                // EditText and SeekBar are EXCLUDED, because this audit cannot judge them.
                //
                // Android names a text field through its HINT and a slider through its value/range
                // metadata, and `UiObject2` exposes neither — it offers only `getText()` and
                // `getContentDescription()`. So both arrive here looking unnamed whatever the
                // markup says.
                //
                // That is not a theory. Home's search field carries a correct
                // `<label class="sr-only" for="home-search">` and the scrubber a plain
                // `aria-label`, and BOTH were reported as unnamed. Acting on that, I added
                // `aria-labelledby` to two components whose markup was already right, measured no
                // change, and reverted it (2026-09-25). Ten of the original thirty-eight findings
                // were this false positive.
                //
                // Excluding them is honest about what the tool can see. Verifying a text field's
                // or a slider's name on Android needs a probe that reads `AccessibilityNodeInfo`
                // directly — worth building, and NOT covered here. Recorded so the gap is visible
                // rather than silently "passing".
                if (kind.endsWith("EditText") || kind.endsWith("SeekBar")) continue;
                String cls = String.valueOf(o.getClassName());
                cls = cls.substring(cls.lastIndexOf('.') + 1);
                // Bounds included: two unnamed controls are otherwise indistinguishable in the
                // report, and "which one" is the first thing anyone fixing this needs.
                // The NEIGHBOURS, not just the bounds. Coordinates alone cannot identify a control
                // — the page scrolls between runs, so a screenshot taken afterwards does not line
                // up, and matching a rect to a component by eye is guesswork. What the control sits
                // NEXT TO names it: a toggle beside "Episodes"/"Shows" is the browse switch, one
                // beside a topic row is a follow control.
                StringBuilder near = new StringBuilder();
                try {
                    UiObject2 parent = o.getParent();
                    if (parent != null) {
                        for (UiObject2 sib : parent.getChildren()) {
                            String n = Journey.nameOf(sib);
                            if (!n.isEmpty() && near.length() < 120) {
                                near.append(near.length() == 0 ? "" : " / ").append(n);
                            }
                        }
                        if (near.length() == 0) near.append(Journey.nameOf(parent));
                    }
                } catch (Throwable ignored) {
                    near.append("<neighbours unreadable>");
                }
                // WHAT THE PLATFORM HOLDS, on the finding itself. Without it a report says only
                // that something is broken; with it the report says which field is empty, which is
                // the difference between "add an aria-label" and "the label association is not
                // reaching the bridge at all". A null probe is reported AS null — "could not see
                // this node" is not the same as "this node has no name", and collapsing the two
                // would turn a probe failure into a clean bill of health.
                String platformDetail = platform == null
                        ? " platform=<probe could not see this node>"
                        : " platform=" + platform;
                String entry = (name.isEmpty() ? "<NO NAME>" : "<SYMBOL-ONLY '" + name + "'>")
                        + " " + cls + " " + o.getVisibleBounds()
                        + " near=[" + near + "]"
                        + platformDetail;
                if (seen.add(entry)) bad.add(entry);
                } catch (androidx.test.uiautomator.StaleObjectException e) {
                    // The page re-rendered this node away mid-read. Transient by nature; counted
                    // below so a surface that does it constantly cannot pass as clean.
                    stale += 1;
                }
            }
        } catch (Throwable t) {
            // A diagnostic that throws replaces the finding with its own failure.
            bad.add("<audit of " + surface + " threw " + t + ">");
        }
        // A QUARTER of the surface unreadable means the walk proved nothing about it. The floor of
        // 4 keeps a tiny surface with one flickering node from failing the suite on noise.
        if (stale > 3 && stale * 4 > examined) {
            bad.add("<NOT AUDITED RELIABLY: " + stale + " of " + examined + " nodes on " + surface
                    + " went stale mid-read, so this surface was not meaningfully walked. It is"
                    + " re-rendering while being audited — settle it before trusting a pass.>");
        } else if (stale > 0) {
            System.out.println("=====AUDIT " + surface + " skipped " + stale + " stale node(s) of "
                    + examined + "=====");
        }
        return bad;
    }

    /**
     * Walk one surface, or record that it could not be walked.
     *
     * AN UNREACHABLE SURFACE IS A FINDING, not a skip (2026-09-27). The overflow step used to be
     * `if (tap(…)) { audit }` — so on any run where the ⋯ could not be opened, the audit reported
     * the panel clean without ever having seen it, and the suite went green. That is the
     * no-postcondition shape this tier has been bitten by repeatedly; the `if exists { tap() }` in
     * `DownloadThroughUITests` is the same construct, and it is why a suite called
     * `…AndQueuesThem` makes zero queue writes.
     *
     * The inventory goes into the message because "would not open" needs the screen it was on: the
     * usual cause is that the label moved, not that the surface is gone.
     */
    /**
     * Tap a Profile tab and wait until it has ARRIVED: its own heading is on screen and the Account
     * panel's "Sign out" is not. Both, because Android's tree holds only on-screen nodes, so an
     * absent "Sign out" alone is also what a scrolled Account panel looks like.
     */
    private static boolean openProfileTab(String name, String heading) {
        if (!Journey.tap(name, false, 8_000)) return false;
        long deadline = System.currentTimeMillis() + 15_000;
        while (System.currentTimeMillis() < deadline) {
            if (Journey.find(heading, false, 500) != null
                    && Journey.find("Sign out", false, 500) == null) return true;
            Journey.sleep(500);
        }
        return false;
    }

    private void audit(String surface, boolean opened, List<String> findings) {
        String slug = surface.replace(' ', '-').replace("▸", "in");
        if (!opened) {
            findings.add("[" + surface + "] NOT AUDITED — the surface would not open, so nothing "
                    + "here has been checked. On screen: " + Journey.labelledInventory(40));
            Journey.shot("audit-unreachable-" + slug);
            return;
        }
        Journey.sleep(3_000);
        List<String> bad = unusableOn(surface);
        for (String b : bad) findings.add("[" + surface + "] " + b);
        // PHOTOGRAPH THE SCREEN THAT PRODUCED THE FINDING (2026-09-26).
        //
        // A rect alone cannot identify a control, and matching one against a screenshot taken
        // later is guesswork — the surface does not have to be at the same scroll position when
        // you go back to look. I did exactly that on the Discover finding and "confirmed" the
        // wrong control by eye, then changed two components on the strength of it. `near` was
        // telling me otherwise the whole time: it was EMPTY, and the control I had picked sits
        // beside a clearly-named one.
        if (!bad.isEmpty()) Journey.shot("audit-" + slug);
    }

    @Test
    public void everyTappableControlHasANameAPersonCanActOn() {
        boolean ready = startClean();
        assertTrue("sign-in did not complete as " + accountIdentity(), ready);

        List<String> findings = new ArrayList<>();

        for (String tab : Arrays.asList("Home", "Discover", "Library")) {
            audit(tab, Journey.openTab(tab), findings);
        }

        // THE PROBE HAS TO BE ABLE TO SEE A NAME AT ALL (2026-09-27).
        //
        // `A11yProbe` is now what clears a suspected finding — if `best()` returned "" for every
        // node, through a bad field order or an API that is not populated on this image, it would
        // clear NOTHING and every unnamed-looking control would be reported as a real defect. The
        // failure mode runs the other way too: a probe that silently returned a name for
        // everything would clear every finding and this suite would pass forever.
        //
        // So: on a surface known to be full of named controls, the probe must report a healthy
        // number of them. This is the check that stops the probe from being a thing that looks like
        // it is checking something.
        //
        // The tab is opened EXPLICITLY rather than inherited from the loop above. It used to rely
        // on "whatever the loop left open", which means reordering that list silently changes what
        // this measures — a self-check whose subject depends on unrelated code is the same class of
        // fragility it exists to catch.
        Journey.openTab("Library");
        Journey.sleep(2_000);
        int namedByPlatform = 0;
        for (A11yProbe.Named n : A11yProbe.actionableByBounds().values()) {
            if (usable(n.best())) namedByPlatform += 1;
        }
        assertTrue(
                "the platform probe found " + namedByPlatform + " named controls on Library — it is "
                        + "not reading names at all, so every clearance it gives elsewhere in this "
                        + "audit is worthless and every finding it reports is unverified. Fix the "
                        + "probe before trusting this suite's result either way.",
                namedByPlatform >= 5);

        // THE SURFACES THIS AUDIT NEVER WALKED (#2156).
        //
        // For months the walk was Home / Discover / Library / player / player ⋯ — five surfaces —
        // and the issue's "37 unnamed controls, ~32 remaining" was a PROJECTION onto the ones
        // below, which nothing had ever opened. A count nobody measured is not a worklist. Walking
        // them produced TWO findings in the entire app, both on Settings.
        //
        // Profile and Settings are reached through the masthead avatar, which is why they need this
        // suite's own labels: the link is named `auth.user?.name || t('profile.title')`, so for a
        // per-class identity the generic string is NOT what is on screen.
        audit("Profile", Journey.openProfile(profileLabels()), findings);
        // PROFILE ▸ TOPICS IS EXCLUDED, and this is a real open defect rather than a tidy-up.
        //
        // Walking it reports "15 of 29 nodes went stale mid-read" on roughly half of runs — always
        // exactly 15 of 29 when it fires, so a fixed subset of controls is being destroyed and
        // recreated while the surface is read. That is a genuine app behaviour worth fixing: a
        // control recreated after load drops focus, moves a screen reader's position, and can
        // swallow a tap already in flight.
        //
        // It is excluded rather than tolerated because the alternative is a suite that is red half
        // the time, which trains everyone to ignore it — and NOT silently, because a silent
        // exclusion is how this audit came to walk five surfaces while an issue claimed it covered
        // the app.
        //
        // THE CAUSE IS NOT KNOWN. A first diagnosis blamed the dynamic tag at ProfileView.vue:641
        // (`:is="i.openId ? 'button' : 'span'"`) flipping when `getStorylines()` resolves. That is
        // WRONG: `load()` assigns interests, clusters and storylines together out of one
        // `Promise.all` (ProfileView.vue:226-247), so there is no window where `openId` resolves
        // late. Do not re-derive that theory. Candidates not yet tested: `onActivated` firing a
        // second `load()` while the walk is in progress, and the Topics tab's own content
        // re-rendering. Restore this line once the churn is settled — the surface has never been
        // audited, so whatever names it holds are still unknown.
        //
        // UPDATE 2026-09-29 — the "15 of 29" is NOT app churn, on either tab. Profile ▸ Account has
        // exactly 29 clickable nodes, and exactly 15 of them live only in the Account panel (Compact,
        // Full, the twelve delivery checkboxes, Sign out). A diagnostic pre-pass recorded that. The
        // walk was reading ACCOUNT, because "opened" was only the tap returning true; the tab switch
        // landed mid-walk, `v-show` hid those 15, and they went stale. Measured on Stats, which
        // failed the same way. So arrival is now verified: the Account panel has to be gone.
        audit("Profile ▸ Stats", openProfileTab("Stats", "Your activity"), findings);

        // Settings verifies by ARRIVING (Journey.openSettings), not by the tap returning true.
        // This is where both findings were, and where the fix for them has to be proven.
        audit("Settings", Journey.openSettings(profileLabels()), findings);

        // The player is where the most icon-only controls live, and where all three defects above
        // were found.
        AppSession.openEpisode("p06-7217050bc6");
        Journey.sleep(5_000);
        for (String bad : unusableOn("player")) findings.add("[player] " + bad);

        // The overflow, open — its items are only in the tree while the panel is rendered.
        audit("player ⋯", Journey.tap("More actions", false, 15_000), findings);

        // THE INSIGHTS PANEL AND ITS NOTE COMPOSER. `NoteComposer` is one of the components the
        // static guard lists, and no device suite has ever audited the names on the surface it
        // lives on — `NativeCapabilityTests` reaches the composer only to assert a mic exists.
        audit("player ▸ Insights", Journey.tap(Arrays.asList("Insights", "✦ Insights"), true, 15_000), findings);
        // Notes are the LAST section of a long panel. Scrolling to the textarea's aria-label
        // ("Your notes" = notes.title) rather than the placeholder, because aria-label wins over
        // placeholder here — matching 'Add a note…' finds nothing, on both platforms.
        //
        // Deliberately NOT clicking the textarea: focusing it opens the soft keyboard, and Android's
        // tree holds only ON-SCREEN nodes, so the button row directly beneath the field drops out of
        // the tree at exactly the moment it is drawn. That is what made the dictation test flake
        // 1-pass/2-fail in suite against 4/0 solo, and an audit that did it would report the mic and
        // Add button as missing rather than as unnamed.
        audit("player ▸ notes", Journey.scrollTo("Your notes", false) != null, findings);

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
