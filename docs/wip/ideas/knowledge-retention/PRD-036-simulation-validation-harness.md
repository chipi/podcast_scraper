# PRD-036: Simulation & Validation Harness

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: alongside Knowledge Retention v1 (internal)
- **Layer**: internal — synthetic data; encodes the value model.
- **Parent PRD**: `PRD-034-knowledge-retention-layer.md`
- **Architecture**: `RFC-087-simulation-validation-harness.md`
- **Relies on**: `RFC-086` (sandbox), `RFC-081` (substrate), `PLAN` (clock injection)

> **Proposed numbering** — `PRD-036` / `RFC-087` are placeholders; verify and renumber.

---

## Summary

The Knowledge Retention model makes strong claims — that signals fire usefully, that the
flywheel compounds, that value emerges in tens of episodes. This capability lets us **validate
those claims without a single real user**, by simulating mock-user cohorts against a synthetic
corpus with **planted ground truth**, so precision, recall, and lift are *computed*, not
guessed.

Simulation is treated as a **first-class, permanent capability**, not a one-off test for v1.

---

## First-class principle

Every future change to this model is validated through the harness *before* it reaches real
users: a new `λ`/`θ` setting, a new change-type (Dial B), a new reconcile tier (Dial A), a new
ranking. The harness is the **standing proving ground** — the same role the autoresearch loop
(RFC-056) plays for enricher quality, applied to retention *value*. It earns its own PRD/RFC
for the same reason the operator control room does: it is durable infrastructure, not a script.

---

## Two regimes (and which this owns)

- **Regime 1 — mechanism correctness** (deterministic; given known input, is the output
  exactly right?). Owned by **PLAN tests** (invariant recompute, reconcile selection, decay
  monotonicity, cost-bound, R1). Not this doc.
- **Regime 2 — value plausibility** (do the signals fire at useful rates with good precision as
  corpus and history evolve?). **Owned here.**

The enabler for Regime 2 is the pairing of **planted ground truth** + **compressed virtual
time** (RFC-087).

---

## Goals

1. Validate **value plausibility** of every operation (connect, reconcile, resurface, advise)
   against ground truth, without real users.
2. Answer the quantitative questions: **how much corpus** creates value, **what expansion
   rate** exercises fresh/stale/decay, **at what anchor count** each surface becomes useful.
3. Test the **flywheel thesis** falsifiably (density-vs-episodes, advise on/off).
4. Provide the **tuning surface** for `λ`/`θ` (working-set health under a cohort).
5. Be **reusable** for every future param/feature/tier change.

---

## Non-Goals

- **Not** a validator of *felt* value — whether the Map feels like your mind, whether resurface
  helps vs nags. That needs dogfood + humans (see §Validation ladder).
- **Not** a replacement for real-user telemetry once it exists (persona policies are guesses
  until calibrated).
- **Not** the autoresearch optimizer (RFC-056) — kindred, distinct.
- **Not** Regime-1 mechanism tests (those live in PLAN).

---

## Personas (the cohort)

Four archetypes, each chosen to **stress a different operation's corner case** (configs over
RFC-087's policy engine, not bespoke code):

| Persona | Behaviour | Stresses |
|---|---|---|
| **Narrow tracker** | deep on 1–2 entities | reconcile (best case), Map density |
| **Broad browser** | many shallow anchors | connect noise, cold-start, working-set bloat |
| **Lapsed user** | deposits then goes quiet | decay → dormancy, **watermark hole** |
| **Reinforcer** | re-encounters / acks often | salience persistence, decay resistance |

---

## Metric catalog (value signals, computed vs ground truth)

| Metric | Definition | Ground-truth basis |
|---|---|---|
| **Connect precision/recall** | of surfaced connections, share truly related | planted relations |
| **Reconcile yield** | events / active-user / week | arrival schedule |
| **Reconcile precision** | events on above-θ nodes the user actually tracks | persona interest + θ |
| **Resurface timing** | fires near natural relevance vs random-ping baseline | planted recurrence |
| **Advise lift** | top-ranked episodes' L2-touch vs chronological/random | held-fixed subgraph |
| **Working-set health** | active-set size stays bounded across cohort | decay + θ |
| **Cold-start curve** | first episode index each surface yields ≥1 useful output | per-persona run |
| **Flywheel** | subgraph density vs episodes, advise on vs off | identical-seed A/B |

The **flywheel** is the single most important measurement — it *is* the thesis. If density
does not grow faster with advise on, the compounding claim is false.

---

## Quantitative preconditions (the answers you asked for)

**How much corpus → recurrence, not episode count.** Value is created by **entity recurrence
in the dimensions a user tracks**, not raw volume. Connect needs a tracked entity in ≥2–3
episodes of the user's path; reconcile needs anchored nodes to receive *new* references
*after* anchoring. **Run the corpus value-potential diagnostic (RFC-087 — recurrence *and*
dispersion) over the real beta corpus (~680 eps / 14 shows / 5 clusters) before trusting it**:
recurrence confirms the connect/reconcile floor is cleared; **dispersion** confirms the
clusters aren't echo chambers (high overlap, low divergence) — the differentiated value needs
both.

**Per-user value emerges in the tens of episodes.** The Map reads as structure at ~30–50
anchored nodes; at 1–3 anchors/episode that is ~15–40 episodes consumed — matching the
"after 10–50–100 podcasts" intuition.

**Expansion rate → derive it.** Post-anchor arrivals must touch anchored nodes within ~1–2
half-lives or you only test decay. Pick a target cadence (e.g. ≥1 reconcile event / active-user
/ week), and from measured recurrence back out the new-eps/week/cluster needed (RFC-087 worked
example: ~1–2/week/cluster for typical params). In production this becomes the **corpus
freshness SLO**.

---

## Validation ladder (staged; sim is the lower rungs)

1. **Mechanism tests** (PLAN) — exact correctness.
2. **Cohort simulation** (this) — value plausibility, rates, precision, flywheel.
3. **Dogfood / user-zero** — you, via the operator viewer (PRD-035 inspect-as-user + λ/θ
   simulator). Felt value, first contact with reality.
4. **Tiny real cohort** — a handful of humans for felt value + deposit-rate calibration.

Sim validates that signals fire correctly, at useful rates, with measurable precision, and
that the flywheel compounds. It **cannot** validate felt value — that is rungs 3–4.

---

## Success Criteria

1. A cohort run produces **computed precision/recall** for connect and reconcile against
   planted ground truth (not estimates).
2. The **cold-start curve** is produced per persona — the empirical "episodes-to-value" per
   surface.
3. **Working-set size stays bounded** across the cohort under the seeded `λ`/`θ`; the harness
   surfaces a tuned `(λ, θ)` candidate.
4. The **flywheel A/B** shows (or fails to show) density growing faster with advise on — a
   clear thesis verdict.
5. The **corpus value-potential diagnostic** (recurrence **and** dispersion) has been run over
   the real beta corpus: per-cluster recurrence clears the floor, and dispersion is reported
   with echo chambers (high-overlap/low-divergence) flagged.
6. A **contradiction reconcile path** is validated via synthetic injection (R1) before the
   detector exists.

---

## Dependencies

- **Injectable clock** — RFC-081 decay/resurface/timestamps; added to **PLAN M2/M3** as
  acceptance.
- **RFC-086 sandbox** + `reconcile/inspect` + `decay/simulate`.
- **Real substrate functions** callable in-process at sandbox scope.
- **Corpus value-potential diagnostic** (recurrence + dispersion) runnable over the real beta corpus.

---

## References

- `RFC-087-simulation-validation-harness.md`
- `PRD-034-knowledge-retention-layer.md`
- `RFC-081-personal-knowledge-layer-reconciliation.md`
- `RFC-086-operator-data-access-and-tooling.md`
- `PLAN-knowledge-retention-v1-build.md`
- `RFC-056-autoresearch-loop.md`
