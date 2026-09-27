package app.closelistening.player;

import org.junit.Rule;
import org.junit.rules.TestName;

/**
 * The base every Android device suite extends (#2139) — the sibling of `UITestCase.swift`.
 *
 * ## Why this exists
 *
 * The iOS tier had no shared setup for months, so each suite inherited whatever the previous one
 * persisted. Relaunching resets memory — but `localStorage`, Capacitor `Preferences` and the
 * account's data on the fixture api all survive it. Ordering decided results.
 *
 * Every native failure on 2026-09-16 was that, not a product bug: one suite left forced-offline ON
 * and three later tests failed with "no person row" / "no push cell" / an upload rejection. A
 * failure names the victim, never the culprit. Android starts with the lesson already applied
 * rather than learning it the same way.
 *
 * ## The two kinds of leaked state, and the two fixes
 *
 * 1. **Account data** (favourites, queue, captures, interests) lives server-side against a user id.
 *    Fixed by giving each suite its OWN identity, so suites cannot collide at all. The mock
 *    provider mints an account for any name, so the dev picker is the whole mechanism.
 * 2. **Device-local state** (`lp.forceOffline` in `localStorage`, dismissals) belongs to the APP,
 *    not the account, so a fresh identity does not clear it. Fixed by normalising it in
 *    {@link #startClean()}.
 *
 * ## Suites that WANT shared state say so
 *
 * A suite reading what an earlier target downloaded overrides {@link #accountIdentity()} to
 * {@link #SHARED_SEEDED_IDENTITY} — which makes the coupling a declaration rather than an accident.
 */
public abstract class UITestCase {

    /** The account the make-level targets seed. A suite reading seeded data opts in explicitly. */
    static final String SHARED_SEEDED_IDENTITY = "simtest";

    @Rule
    public TestName testName = new TestName();

    /**
     * The account this suite signs in as. Per-suite by default, derived from the class name.
     *
     * Override with {@link #SHARED_SEEDED_IDENTITY} in a suite that genuinely depends on seeded
     * state — and say why, because it re-enters the shared world this class exists to leave.
     */
    protected String accountIdentity() {
        // `DownloadThroughUITests` -> `downloadthroughuitests`. Lowercased and stripped to the
        // charset the mock provider accepts, matching how the browser tier derives its own ids.
        StringBuilder sb = new StringBuilder();
        for (char c : getClass().getSimpleName().toCharArray()) {
            if (Character.isLetterOrDigit(c)) sb.append(Character.toLowerCase(c));
        }
        return sb.toString();
    }

    /**
     * How to reach Profile for THIS suite's account.
     *
     * The masthead link is named `auth.user?.name || t('profile.title')`, so the label is the
     * account name once it resolves and the generic string only until then. Both have to be tried,
     * and only the suite knows the first one.
     */
    protected java.util.List<String> profileLabels() {
        return java.util.Arrays.asList(accountIdentity(), "Your profile");
    }

    /**
     * Bring the app to a known state, then sign in as this suite's account.
     *
     * NOT an `@Before`: several suites deliberately start with the app or the api in an unusual
     * state (offline, server degraded, cold install), and a base class that force-launched the app
     * would fight them. Call it first from a test that wants the guarantee.
     */
    protected boolean startClean() {
        AppSession.relaunch();
        // SIGN IN FIRST, then normalise the device switch — the opposite order to the iOS twin, on
        // purpose. The offline switch lives in Settings, Settings is reached through the masthead
        // avatar, and the avatar only exists when there IS a session. So a signed-out app cannot
        // reach the switch at all, and putting it first spends about a minute of swipes and
        // timeouts discovering that on every single test.
        boolean signedIn = AppSession.ensureSignedIn(accountIdentity());
        if (!signedIn) return false;
        // Forced-offline OFF: it is device-local, so it survives a relaunch AND an account change,
        // and left ON every later network assertion fails for a reason that has nothing to do with
        // the suite reporting it. That was every native failure on 2026-09-16 on iOS — one suite
        // left it on, three later ones failed naming themselves.
        Journey.setOfflineMode(false, profileLabels());
        return true;
    }
}
