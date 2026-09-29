package app.closelistening.player;

import android.app.UiAutomation;
import android.graphics.Rect;
import android.os.Build;
import android.view.accessibility.AccessibilityNodeInfo;

import androidx.test.platform.app.InstrumentationRegistry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads the accessibility tree that `UiObject2` only partly exposes (#2156, 2026-09-27).
 *
 * ## Why this exists
 *
 * `AccessibleNameAuditTests` judges a control by `UiObject2.getText()` and
 * `getContentDescription()`, because those are the only two name-bearing fields UI Automator's
 * wrapper offers. That is not where Android keeps every name:
 *
 *  - a text field's name can live in **hintText**;
 *  - a slider's in its **RangeInfo** plus **stateDescription**;
 *  - and an input wrapped in a `<label>` is named through **labeledBy**, a POINTER to the node
 *    holding the words — the words are never copied onto the control itself.
 *
 * So the audit reported Home's search field and the player's scrubber as unnamed even though both
 * carry correct markup, and ten of the original thirty-eight findings were that false positive. It
 * excluded `EditText` and `SeekBar` to stay honest, and recorded the gap: *"Verifying a text
 * field's or a slider's name on Android needs a probe that reads `AccessibilityNodeInfo` directly
 * — worth building, and NOT covered here."* This is that probe.
 *
 * It settles a specific open question. Settings' two switches
 * (`<label><span>Offline mode</span><input type=checkbox></label>`) report empty text AND empty
 * contentDescription, and adding an explicit `aria-label` to the input changed neither — measured,
 * with the rebuilt bundle confirmed served. Either the name is reaching the platform somewhere
 * `UiObject2` cannot see, or those controls are genuinely nameless to TalkBack. Those need opposite
 * fixes, and nothing already in the harness can tell them apart.
 *
 * ## What it does NOT do
 *
 * It does not decide what TalkBack would SAY — announcement composes name, role, state and
 * live-region policy, and this reads the inputs to that, not its output. A control named here can
 * still be announced confusingly. This answers "does the platform have a name for this node", which
 * is the question the audit is actually asking.
 */
final class A11yProbe {

    private A11yProbe() {}

    /** Everything the platform knows about one node's name, for reporting. */
    static final class Named {
        final String cls;
        final String text;
        final String desc;
        final String hint;
        final String stateDesc;
        final String labeledBy;
        final boolean checkable;
        final boolean hasRange;
        final Rect bounds;

        Named(String cls, String text, String desc, String hint, String stateDesc,
              String labeledBy, boolean checkable, boolean hasRange, Rect bounds) {
            this.cls = cls;
            this.text = text;
            this.desc = desc;
            this.hint = hint;
            this.stateDesc = stateDesc;
            this.labeledBy = labeledBy;
            this.checkable = checkable;
            this.hasRange = hasRange;
            this.bounds = bounds;
        }

        /**
         * The best name the platform holds, in the order assistive tech would resolve it.
         *
         * contentDescription wins over text when both exist — that is the platform's own
         * precedence, and it is the opposite of what `Journey.nameOf` does. `nameOf` reads text
         * FIRST on purpose, because the HARNESS wants the string a lookup can match; this wants
         * the string a SCREEN READER would use. Two different questions, deliberately two answers.
         */
        String best() {
            if (nonEmpty(desc)) return desc;
            if (nonEmpty(text)) return text;
            if (nonEmpty(labeledBy)) return labeledBy;
            if (nonEmpty(hint)) return hint;
            if (nonEmpty(stateDesc)) return stateDesc;
            return "";
        }

        /** Where the name came from, so a finding says what to fix rather than only that it is broken. */
        String source() {
            if (nonEmpty(desc)) return "contentDescription";
            if (nonEmpty(text)) return "text";
            if (nonEmpty(labeledBy)) return "labeledBy";
            if (nonEmpty(hint)) return "hintText";
            if (nonEmpty(stateDesc)) return "stateDescription";
            return "<none>";
        }

        @Override
        public String toString() {
            return cls + " " + bounds
                    + " name='" + best() + "' via=" + source()
                    + " [text=" + q(text) + " desc=" + q(desc) + " labeledBy=" + q(labeledBy)
                    + " hint=" + q(hint) + " state=" + q(stateDesc)
                    + " checkable=" + checkable + " range=" + hasRange + "]";
        }
    }

    private static boolean nonEmpty(String s) {
        return s != null && !s.trim().isEmpty();
    }

    private static String q(String s) {
        return s == null ? "null" : "'" + s + "'";
    }

    private static String str(CharSequence cs) {
        return cs == null ? null : cs.toString().trim();
    }

    /**
     * Every node the platform considers actionable, keyed by its screen bounds.
     *
     * Keyed by bounds because that is the one identifier shared with a `UiObject2` finding — the
     * audit reports a `Rect`, and this lets a finding be looked up against what the platform really
     * holds. Bounds are unstable across scrolls, which is why the lookup must happen in the SAME
     * pass as the finding and never against a dump taken afterwards.
     */
    static Map<String, Named> actionableByBounds() {
        Map<String, Named> out = new LinkedHashMap<>();
        UiAutomation ua = InstrumentationRegistry.getInstrumentation().getUiAutomation();
        AccessibilityNodeInfo root = ua.getRootInActiveWindow();
        if (root == null) return out;
        collect(root, out, 0);
        return out;
    }

    private static void collect(AccessibilityNodeInfo node, Map<String, Named> out, int depth) {
        if (node == null || depth > 60) return;
        try {
            if (node.isClickable() || node.isCheckable() || node.isFocusable()) {
                Rect b = new Rect();
                node.getBoundsInScreen(b);

                String hint = null;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    hint = str(node.getHintText());
                }
                String stateDesc = null;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    stateDesc = str(node.getStateDescription());
                }

                // THE POINTER, followed. `labeledBy` is why a `<label>`-wrapped input can be
                // correctly named while carrying no name of its own — the words live on the node
                // it points AT. Not following it is exactly how this audit mistook correct markup
                // for a defect ten times.
                String labeledBy = null;
                try {
                    AccessibilityNodeInfo lab = node.getLabeledBy();
                    if (lab != null) {
                        String t = str(lab.getText());
                        String d = str(lab.getContentDescription());
                        labeledBy = nonEmpty(d) ? d : t;
                    }
                } catch (Throwable ignored) {
                    // getLabeledBy can throw on a stale node; absence is reported as absence.
                }

                boolean hasRange = false;
                try {
                    hasRange = node.getRangeInfo() != null;
                } catch (Throwable ignored) {
                    // no range info available
                }

                out.put(b.toShortString(), new Named(
                        String.valueOf(node.getClassName()),
                        str(node.getText()),
                        str(node.getContentDescription()),
                        hint,
                        stateDesc,
                        labeledBy,
                        node.isCheckable(),
                        hasRange,
                        b));
            }
            for (int i = 0; i < node.getChildCount(); i++) {
                collect(node.getChild(i), out, depth + 1);
            }
        } catch (Throwable ignored) {
            // A probe that throws replaces the finding with its own failure — same rule as the audit.
        }
    }

    /**
     * The platform's name for the node at {@code bounds}, or {@code null} when it holds none.
     *
     * {@code null} and empty-string mean different things here: null is "the probe could not see
     * this node at all" (it moved, or the window changed), empty is "the platform has it and it has
     * no name". Collapsing them would turn a probe failure into a clean bill of health.
     */
    static Named at(Map<String, Named> probe, Rect bounds) {
        if (probe == null || bounds == null) return null;
        Named exact = probe.get(bounds.toShortString());
        if (exact != null) return exact;
        // Tolerate a pixel of drift between the two trees, which are read microseconds apart.
        for (Map.Entry<String, Named> e : probe.entrySet()) {
            Rect r = e.getValue().bounds;
            if (Math.abs(r.left - bounds.left) <= 2 && Math.abs(r.top - bounds.top) <= 2
                    && Math.abs(r.right - bounds.right) <= 2 && Math.abs(r.bottom - bounds.bottom) <= 2) {
                return e.getValue();
            }
        }
        return null;
    }

    /** Every checkable node the platform holds, for a one-shot diagnostic of a surface. */
    static List<String> checkableReport() {
        List<String> out = new ArrayList<>();
        for (Named n : actionableByBounds().values()) {
            if (n.checkable) out.add(n.toString());
        }
        return out;
    }
}
