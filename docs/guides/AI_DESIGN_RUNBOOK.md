# AI-Driven Visual Design Runbook

> **Source:** Adapted from Lenny Rachitsky's _"How to turn your AI into a world-class designer"_ ([Lenny's Newsletter](https://www.lennysnewsletter.com/p/how-to-turn-your-ai-into-a-world)). The thesis: models default to safe, generic design; deliberately pushing them off those defaults unlocks better work.

This guide structures a **Double Diamond discovery-and-delivery workflow** for raising visual design and UX quality of `web/learning-player` (the Vue 3 mobile-first consumer app) using AI agents. It is actionable by one person sitting at this repo with a set of specific techniques, a wired-in critic loop, and clear guardrails.

**When to use this:** Your goal is elevating the look and feel of an **existing surface** — making it bolder, more intentional, more memorable. Not for routine feature work or bugfixes.

**When NOT to use this:** This runbook does not apply to:
- Incremental tweaks to spacing or a single component
- Work that violates the frozen design token API or accessibility contracts
- Exploring new features (that is feature work, not design elevation)
- Routine responsive-layout fixes

---

## The Double Diamond, adapted to this codebase

**Discover → Define → Deliver.** Each stage has specific constraints in this app.

### Stage 1: Discover (divergent exploration)

The goal: **generate bold, unexpected directions without breaking the design system.**

Your starting point is the current state of one surface (`HomeView.vue`, `PlayerView.vue`, `SearchView.vue`, etc.). You will generate 3–5 creative directions, screenshotted, without shipping any yet. The tension here: this codebase has a deliberate, frozen visual identity (UXS-011 "Editorial Bold, dark-primary"; UXS-014 "Interaction Patterns"). You are **not** free to invent new colour tokens or typography rules.

**What you _can_ break:** layout, composition, negative space, motion, the *arrangement* of existing components, asymmetry, the visual hierarchy of sections.

**What you _cannot_ break:**
- The `--lp-*` semantic token layer (no raw hex in components; the names are the frozen API)
- Typography hierarchy (the `.lp-kicker` eyebrow, `.lp-section` calm display heading, `.lp-speaker` speaker label are canonical and tested)
- The dark-primary theme (light theme is post-MVP; do not add it here)
- Accessibility contracts (focus states, contrast, reduced-motion)
- Mobile-first viewport (Pixel 7 is the default Playwright project)

#### Technique 1: Seed strings

**Why:** Models are next-token predictors; a random external input forces decisions they would not make alone.

**How:**

1. Generate a random alphanumeric string via shell (e.g., `openssl rand -hex 8` → `a7f2b9e3c4d6f1a8`).
2. Show the string to the model: *"Here is a seed string: `a7f2b9e3c4d6f1a8`. Derive a bold creative direction (colour palette, layout principle, type treatment, motion philosophy) from patterns you see in it. Do not reveal the string to me; make the decisions yourself."*
3. The model builds a mockup without you knowing which patterns it chose, forcing it to reason backward from a constraint.

**Example:** The seed `a7f2b9e3c4d6f1a8` might yield: *"The pattern of warmth (f → 'fire') and cool geometry (d6, a8 → 'precision edges') suggests a direction: warm accent over sharp grid layouts, strong contrast, minimal adornment."*

#### Technique 2: Ambitious briefs

**Why:** Vagueness → safe defaults (purple gradients, text-left / graphic-right). Concrete examples unlock boldness.

**Lenny's examples:**
- *"Bold pixel art theme… each section should feel like a still from a video game."*
- *"Set in an isometric living 3D city, where features are neighbourhoods or buildings."*
- *"Radically asymmetric layout, dissonant colours and typography, uncomfortable negative space. Break all the rules but still make it look good."*

**How to write your own (three-step method):**

1. Ask the model for high-level ideas with no executable detail (e.g., *"What are 5 radically different visual languages I could apply to the player home screen?"*).
2. Visualize your favourites (ask for mockups or sketches); note your emotional reaction. Which directions surprised you? Which felt *wrong* but intriguing?
3. Have the AI write the executable brief: *"I want the player home to feel like `<emotional idea>`. Write a detailed visual prompt for a designer or AI that makes this concrete."*

**Rule of thumb:** *If you find yourself thinking 'there's no way this will work,' you're on the right track.*

#### Technique 3: Screenshot-based discovery

Use the existing validation harness (`e2e/validation/*.spec.ts`) to screenshot each divergent direction. Do not hand-edit mockups or describe designs in prose — every experiment must exist as **working code** (a Vue branch, a scoped CSS variant, etc.) that runs against the stack so you can evaluate it in context.

**Why:** Designs look different at size, under real network latency, and with real data. The screenshot harness gives you the critic loop's input and keeps your output reproducible.

---

### Stage 2: Define (convergent critique and selection)

The goal: **score each direction, identify the strongest one, and refine it in a loop.**

Use a **design-critic subagent** — a separate instance that sees ONLY screenshots, never the code or the brief. This breaks the feedback loop: if the agent sees the brief, it will police whether you "followed the rules"; if it sees only pixels, it will judge the **execution** and the **aesthetic**.

#### The critic agent file

Save this at `/Users/claude/.claude/agents/design-critic.md`:

```markdown
# Design Critic Agent

You are a world-class design critic for mobile-first consumer applications. Your role:

1. **YOU SEE ONLY SCREENSHOTS.** Never read the code, the creative brief, or the intended direction. Judge only pixels.

2. **You are strict and opinionated.** Default to criticism. Penalise:
   - Anything that feels "overdone, excessive, or obviously AI-generated" (glowing gradients, too many drop-shadows, bleeding accent colour, random icons as decoration, redundant contrast "fixes")
   - Visual noise; poor negative space hygiene
   - Typography that does not breathe (lines too tight, too-bold headlines, insufficient whitespace between sections)
   - Asymmetry without purpose; randomly placed elements
   - Components that feel tacked-on rather than intentional

3. **You must be bold and specific.** Not: "the layout feels off." Instead: "the episode card image is too large and crowds the title; reduce by 40% and align the text baseline with the navigation bar above."

4. **You score on a 0–10 scale:**
   - **0–3:** Fundamentally broken (crashes accessibility, violates platform convention, unreadable)
   - **4–5:** Safe and competent but forgettable; reads as "standard app"
   - **6–7:** Distinctive and intentional; a few rough edges
   - **8–9:** Cohesive, bold, memorable; all details serve the whole
   - **10:** Transcendent; every choice feels inevitable

5. **For each screenshot, you produce:**
   - A **numerical score** (0–10)
   - **Two specific strengths** (what actually works)
   - **Two specific weaknesses** (what to fix next iteration)
   - **One big swinging critique** (the thing that would elevate it the most)

6. **You IGNORE:**
   - Whether it is mobile vs desktop (just evaluate the viewport shown)
   - Whether the copy is real or placeholder (judge the layout, not the words)
   - Bugs or incomplete implementations (assume the code will ship as intended)

7. **You FAVOUR:**
   - Restraint and subtlety (a design that exercises *negative space* looks premium)
   - Intentional inconsistency (breaking rules *on purpose* for effect)
   - Motion and micro-interaction (even still screenshots hint at how it moves)
   - Typography hierarchy (clean, calm, confident)

You do not write code, propose implementations, or reference UX specs. You are a pure aesthetic judge.
```

#### How to run the critic loop

1. **Boot the stack** for your experiment: `make serve-for-validation` or `make serve`.
2. **Take a screenshot** of the variant using the validation harness:
   ```typescript
   // e2e/validation/design-critique.spec.ts (new file)
   test('variant A screenshot', async ({ page }) => {
     await page.goto('/')
     await page.waitForLoadState('networkidle')
     await page.screenshot({ path: 'validation-results/variant-a-home.png', fullPage: true })
   })
   ```
3. **Run Playwright:** `cd web/learning-player && npm run test:e2e:validation` (outputs to `validation-results/*.png`).
4. **Show the screenshot to the critic agent:**
   ```
   I'm working on a design direction for the player home screen.
   Here is a screenshot of variant A: [attach image].
   Critique it against the rubric above. Be tough.
   ```
5. **Iterate:** Read the score and feedback. If 6+, move to the next variant or refine this one. If <6, ask the model to suggest one specific change and loop back to step 1.
6. **Convergence target:** Aim for 9/10 on at least one variant, verified over 1–2 iterations that the loop is actually improving (not chasing diminishing returns).

**Cost note:** Use the expensive model (Claude Opus) ONLY for the critic (takes ~10% of your tokens). Use a faster model for the discover and deliver stages.

---

### Stage 3: Deliver (refinement and integration)

The goal: **polish the winning variant and merge it into the codebase without breaking anything.**

#### Technique: Ruthless subtraction

*"AI loves to add more, but it rarely takes away. A design that exercises restraint immediately looks premium and tasteful."* — Lenny Rachitsky

**Things to cut:**
- Unnecessary glows, gradients, or background effects (the `--lp-*` tokens already give you intention; do not add visual noise on top)
- Random or excessive accent-colour highlights on text
- Extra labels where the visual already communicates (e.g., a "new episode" badge AND an accent colour AND a label — pick one)
- Custom components worse than the native equivalent (if a standard Tailwind button reads better than your custom variant, use the standard)
- Excessive containers, whitespace, or padding (start with the minimum and add only where it *breathes*, not where it fills)
- "AI tells" (see the checklist below) — animations that feel random, overprocessed imagery, redundant visual cues

This is **not** about cutting features. It is about cutting decoration and choosing silence over noise.

---

## The critic loop: step-by-step command reference

### Prerequisites

1. **Validation harness:** Confirm you have `e2e/validation/*.spec.ts` and `playwright.validation.config.ts` in place.
   ```bash
   cd /Users/claude/projects/podcast-player/web/learning-player
   ls e2e/validation/ && ls playwright.validation.config.ts
   ```

2. **The critic agent file:** Paste the agent markdown above into `~/.claude/agents/design-critic.md`.

3. **Know your baseline:** Run the current app and screenshot the surface you are redesigning. This is your "before" for the critic.
   ```bash
   make serve-for-validation &
   # Wait for "Server running on ..."
   cd web/learning-player
   npm run test:e2e:validation -- --grep "listen-through"
   # Check validation-results/
   ```

### One iteration (Discover → Critique → Refine)

1. **Create a variant** in a branch or a scoped feature flag:
   - Edit `src/views/HomeView.vue` or the target surface
   - Use the ambitious brief technique or seed string to guide changes
   - Keep all design tokens, accessible names, and i18n calls as-is
   - Do NOT hardcode strings; use `t()` from i18n

2. **Screenshot the variant:**
   ```bash
   # Boot the stack
   make serve-for-validation &
   SLEEP 5
   
   # Run validation (this creates validation-results/*.png)
   cd web/learning-player
   npm run test:e2e:validation -- --grep "listen-through-real-corpus"
   
   # Find the screenshot
   ls -lt validation-results/ | head -3
   ```

3. **Open the critic agent** (in Claude Code, create a new thread or conversation; load the agent file):
   - Paste: `I am using the design-critic agent. Load the rules from the agent file I provide.`
   - Then: `Here is a screenshot of my variant: [attach image from validation-results/]`
   - Critic responds with a score and feedback.

4. **Act on feedback:**
   - If score 8–10 and you like the direction: move to **Deliver** (below).
   - If score 6–7 and the feedback is specific: implement the "one big swinging critique" and loop back to step 2.
   - If score <6: either kill this direction (go back to Discover with a different brief) or ask the critic for one specific pixel-level change and try it.

---

## The Subtraction Checklist

Before you ship, ask the model: *"Apply the subtraction checklist to this design. What should be removed or simplified?"*

- [ ] **Glows and drop-shadows:** Are they serving emphasis, or are they noise? (Hint: in this dark theme, a glow is almost never needed — the accent token already pops.)
- [ ] **Accent overuse:** Is every interactive element orange? Revert non-critical ones to muted.
- [ ] **Text variety:** Do you have 4+ font sizes or weights on one screen? Collapse to 2–3.
- [ ] **Gradient sludge:** Does every surface have a subtle gradient? Test solid colours first.
- [ ] **Whitespace:** Is there a 24px gap where 12px would breathe better? Shrink it.
- [ ] **Icon inflation:** Is there an icon next to every label? Try removing 30% and see if the page is clearer.
- [ ] **Animation:** Does every transition serve a purpose (state change, affordance), or are you animating things "because AI was generous with easing curves"?
- [ ] **Redundant affordances:** "New episode" badge + accent colour + label + icon = four cues for one thing. Pick one.

---

## Our AI-tells checklist (for this codebase)

This checklist is derived from this codebase's history and UX specs, **not from Lenny's paywalled Technique 7.** It catches common tells that design *looks* like it came from an AI rather than having arrived at a decision through taste.

- [ ] **Random asymmetry:** The layout has no clear principle; elements are off-center "for visual interest" rather than following a grid or compositional rule.
- [ ] **Typo surprise:** Headlines are suddenly serif, or script, or aggressively condensed, because the brief asked to "break conventions."
- [ ] **Accent spray:** Every interactive element is orange, because orange is the accent token and the model defaulted to "make it stand out."
- [ ] **Overrendered shadow:** Drop-shadows stack on drop-shadows; elements float unconvincingly.
- [ ] **Animation bloat:** Every page load has a 600ms stagger animation; every tab switch has a 400ms fade. Movement *feels* added rather than necessary.
- [ ] **Gradient fatigue:** The canvas is dark enough; adding a subtle radial gradient "for depth" reads as cheesy, not premium.
- [ ] **Icon overload:** Elements are decorated with icons that don't track the hierarchy (a tertiary element has a bigger icon than a primary one).
- [ ] **Text cringe:** Copy is suddenly uppercase / ALL CAPS / or uses special characters (◆ ✦ ★) for emphasis because the brief asked to "be bold."
- [ ] **Glass-morphism:** Semi-transparent frosted-glass surfaces everywhere (the dark theme does not suit this; it reads as trendy, not timeless).
- [ ] **Neon accents:** Accent token is redrawn as a brighter, more saturated version because "pop" was in the brief.
- [ ] **Inconsistent corners:** Some buttons are `rounded-none`, others `rounded-lg`, others `rounded-full`, all in the same interface, because "variety" was the aim.

**How to use it:** Before the final critique, ask the model: *"Check this screenshot against our AI-tells list. Which ones are present? Are they intentional, or are they mistakes?"*

---

## The "do not break" list

Commit to these before any redesign work:

| What | Where | Why |
|------|-------|-----|
| **Design token layer** | `src/theme/tokens.css` | The `--lp-*` semantic tokens are the frozen API. No raw hex in components. Names are the contract; values are open. |
| **Type treatments** | `src/style.css` (@layer components) | `.lp-kicker`, `.lp-section`, `.lp-speaker`, `.lp-nav` are canonical. Every use of these classes is tested (a11y, contrast). Do not restyle per-page. |
| **Per-show accent + contrast** | `src/theme/accent.ts`, `src/theme/contrast.ts` | The `deriveShowAccent()` logic maps show image → accent colour + contrast validation. Tests exist. Do not bypass. |
| **i18n strings** | `src/i18n/locales/en.json` + `t()` calls | All user-facing copy goes through `t()`. A redesign that hardcodes a string is wrong regardless of how it looks. |
| **Mobile-first viewport** | Playwright default project: Pixel 7 | All screenshots validate against Pixel 7 (375px width). Desktop layouts are secondary. Do not rearrange for desktop and assume mobile "will scale." |
| **Dark-primary theme** | `data-theme='dark'` (light is post-MVP) | Do not add new theme tokens or flip to light mode as part of a redesign. |
| **Accessibility specs** | `e2e/knowledge-panel-a11y.spec.ts` + others | Focus trapping, inertness on modals, keyboard navigation, colour contrast (4.5:1 for text), reduced-motion. Visual changes must not regress these. Run **`npm run test:a11y`** (if it exists; else check CI). |
| **E2E surface map** | `e2e/E2E_SURFACE_MAP.md` (if it exists for the consumer player) | Update this if you change accessible names, regions, or stable selectors. Do not break Playwright locators. |

**Command to verify you have not broken these:**
```bash
cd /Users/claude/projects/podcast-player/web/learning-player
npm run test:a11y 2>/dev/null || echo "No a11y suite; check CI"
npm run test:e2e 2>/dev/null || echo "No e2e suite; check CI"
```

If these fail after your changes, fix them before shipping.

---

## Limits and what we cannot do

### The paywalled technique (Remove AI tells — Technique 7)

Lenny's full rubric for this technique is behind a paywall and is not available here. The "Our AI-tells checklist" above is a standalone answer, not a summary of that paywalled content.

### No visual regression tooling

This codebase has a screenshot harness (`e2e/validation/*.spec.ts` → `validation-results/*.png`) but **no baseline diffing or visual regression detection**. You can compare before/after *by eye*, but:
- There is no automated visual diff.
- The screenshot harness is a **Tier-3 validation** tool (real backend, real corpus, nightly CI upload), not a regression detector.
- If you land a redesign, the next person to edit that surface will not be warned if they accidentally degrade it.

**Workaround:** Keep a before/after screenshot pair in your PR description or commit message so humans can eyeball the change.

### Images and assets

This runbook assumes **you work in code** (CSS, Vue templates, Tailwind utilities). It does not cover:
- **Image generation** (Lenny mentions using AI image gen for assets; this repo has none wired in yet)
- **Video/motion generation** (Lenny mentions looping clips and interpolated keyframes; beyond this runbook's scope)

If your direction needs custom imagery, you will need to either hand-create it, wire in an image-gen API, or do without.

### No light-theme redesign

Light theme is explicitly a post-MVP fast-follow. The design token structure supports it (the `.data-theme` hook is in place), but do not add it as part of a redesign.

---

## Checklist: Am I ready to redesign?

Before you start:

- [ ] I have a **specific surface** in mind (`HomeView`, `PlayerView`, `SearchView`, `LibraryView`, `ProfileView`, or a key component like `EpisodeCard`).
- [ ] I can **articulate my aesthetic goal** in 1–2 sentences (e.g., "bold, playful, game-like" or "minimal, typography-forward, editorial").
- [ ] I understand that I **cannot add new colour tokens**; I can only rearrange, scale, and emphasize the existing `--lp-*` palette.
- [ ] I know where **the critic harness lives** (`e2e/validation/`) and can screenshot a variant.
- [ ] I have loaded **the design-critic agent** (or have access to one) and can run feedback loops.
- [ ] I can **run the app locally** (`make serve-for-validation`) and **regenerate screenshots** in <2 minutes per iteration.
- [ ] I am **not redesigning during a crunch**; this work is iterative and needs breathing room.

---

## Example: The full loop (HomeView redesign)

**Goal:** Make the player home screen feel less "standard app", more "premium and intentional."

### 1. Discover

- Generate a seed string: `openssl rand -hex 8` → `b3f1e7a2d4c9f6e8`
- Brief: *"Extract a direction from this seed. I see an interplay of depth ('b3', 'f1', 'e7') and precision ('a2', 'd4', 'c9'). Make a player home that feels layered but clean — multiple cards stacked with careful shadow / depth, typography that breathes."*
- Model generates 3 variants (still code, not mockups).

### 2. Critique each variant

- Branch: `git checkout -b redesign/home-v1`
- Edit `src/views/HomeView.vue` per the brief.
- Screenshot via validation harness (3 minutes).
- Show to critic agent: score comes back as 7/2 ("strong grid, weak typography spacing").
- Feedback: *"The cards have good hierarchy. Reduce the gap between section title and first card from 24px to 16px; it will breathe better."*

### 3. Refine

- Adjust the spacing in HomeView.vue.
- Re-screenshot.
- Critic now scores 8/10 ("typography hierarchy is intentional; the stacked card layout feels premium").

### 4. Ship

- Run the full e2e suite (`npm run test:e2e`) to ensure no regressions.
- Update `e2e/E2E_SURFACE_MAP.md` if any selectors changed.
- Update `docs/uxs/UXS-011-consumer-learning-app.md` to document the change.
- Commit with a `docs(design):` or `style:` message.
- Screenshot before/after in the PR description for human review.

---

## Further reading

- **UX Specifications:** [UXS-011 Consumer Learning App](../uxs/UXS-011-consumer-learning-app.md), [UXS-014 Interaction Patterns](../uxs/UXS-014-interaction-patterns.md)
- **Design tokens:** `src/theme/tokens.css` (the frozen API)
- **Validation harness:** `e2e/validation/README.md` (if it exists) or `playwright.validation.config.ts`
- **E2E Testing Guide:** [E2E Testing Guide](E2E_TESTING_GUIDE.md)
- **A11y specs:** `e2e/*-a11y.spec.ts` (accessibility is not negotiable)
