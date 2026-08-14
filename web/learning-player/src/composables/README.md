# `src/composables/` — shared behaviour

Small reusable pieces of behaviour. Two of them encode product contracts that the app breaks
silently if you bypass them; the other three wrap browser APIs.

## The composables

| Composable | Purpose | Used by | Bypassing it means |
| --- | --- | --- | --- |
| [`useSignInGate.ts`](useSignInGate.ts) | Signed-out users see gated controls as **teasers** that route to sign-in | 7 files | A 401 the store swallows — the control flips, then silently reverts |
| [`useSectionState.ts`](useSectionState.ts) | `loading` / `ready` / `error` for a data-backed section | 7 files | An outage renders as an empty state — "you follow nothing" to someone with 30 follows |
| [`usePwaUpdate.ts`](usePwaUpdate.ts) | Service-worker update prompt | 1 | — |
| [`useShareCard.ts`](useShareCard.ts) | Renders a highlight to a shareable card (PRD-046 FR5) | 1 | — |
| [`usePushSubscription.ts`](usePushSubscription.ts) | Web Push subscribe/unsubscribe (PRD-046 FR1) | 1 | — |

## `useSignInGate` — the rule, not just the helper

**Auth-gated means *deferred*, not *hidden*.** Before #1590 every gated control rendered
`v-if="auth.isAuthenticated"`, so signed-out visitors saw no evidence that capture, queue, follows
or corpus-search existed — the differentiators were invisible to exactly the people deciding whether
to sign up.

The contract:

- the control **renders**, and states its requirement in its accessible name (`auth.signInTo*`);
- it **claims no toggle state** — omit `aria-pressed`, since nothing is toggled;
- the tap **routes to sign-in** with `redirect` back to `route.fullPath`;
- it **never calls the API**.

`gated()` resolves the session (`auth.ensureLoaded()`) *before* deciding, because `isAuthenticated`
is false until the shell's first refresh lands — without that wait, a user who taps quickly is sent
to log in while already holding a session.

A component that only renders the affordance and emits (the parent owns the API call, so the parent
owns the gate) takes a `gated` prop instead — see `TranscriptList.vue`.

Enforced by [`../__checks__/auth-gate.test.ts`](../__checks__/auth-gate.test.ts), per call site, in
both directions: an ungated write fails, and so does a component that wires the gate while hiding
another control behind `v-if="auth.isAuthenticated"` — that combination is always the half-wired
state, and it shipped twice.

## `useSectionState` — hide when the SYSTEM is empty, render when the USER is

`load()` never rejects and never collapses failure into the initial value; that collapse *was* the
bug. A rejected fetch lands in `error`, which renders differently from an empty success and offers
retry via [`SectionStatus.vue`](../components/SectionStatus.vue).

Deciding what to render when ready-and-empty:

- **system empty** (no corpus activity, no history) → hide; there is no action the user can take;
- **user empty** (follows nothing yet) → render, and **carry the action itself**, not a description
  of it.

Error state must never be silently equal to empty. That is the whole point of the composable.

## Related

- [`../stores/README.md`](../stores/README.md) — what the gated writes actually call
- [`../components/SectionStatus.vue`](../components/SectionStatus.vue) — renders the phases
- [UXS-011](../../../../docs/uxs/UXS-011-consumer-learning-app.md) — the gating rule as spec
- [UXS-012](../../../../docs/uxs/UXS-012-consumer-home.md) — the section state contract

## Known gaps

- **`usePushSubscription.ts` has no test.** The other four do.
