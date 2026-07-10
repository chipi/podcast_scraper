# Speaker resolution — reduce unknown speakers (roadmap + before/after measurement)

**Goal:** name more diarized voices (fewer `SPEAKER_NN`), and keep the ones that stay
unknown from polluting the corpus. Every step is measured **before/after on the same real
corpus** so we know what each one actually contributes.

**Measurement corpus:** `.test_outputs/manual/prod-v2/corpus` — **90 diarized episodes**,
**579 diarized voices** (the caption-derived episodes have no diarization roster and are
excluded). Method: replay `resolve_speaker_roster` on the cached segments + metadata (no
audio, no LLM, no GPU), A/B the one variable per step. Scripts:
`scratchpad/measure_1a.py`, `measure_1b.py`, `measure_reconcile.py` (promotable to
`scripts/` + a `make` target if we want this to be a standing gate).

Baseline is **not** a single number — separate "did we NAME more voices" (reducers) from
"did we stop MIS-LINKING the unknowns" (correctness). They are different wins.

---

## Shipped this branch — measured contribution

| Step | What it does | Measured effect (prod-v2, 90 eps / 579 voices) |
|---|---|---|
| **#1a** own-turn self-intro | Name a voice from *its own* turns' "I'm X", not just the opening host intro | **NAMES MORE:** 212 → **253** voices named (36.6% → **43.7%**), **+41 voices across 31 eps, 0 regressions**. Talk-time on a named voice **72.1% → 76.0%**. Source split: 29 direct self-intro + 12 improved guest matching. |
| **#1b** episode-scoped ids | Unresolved voice → `person:speaker-{ep}-NN`, not global `person:speaker-NN` | **CORRECTNESS:** the old shared ids fused **`person:speaker-00` = 16 episodes across 6 shows** into one phantom person; **9 ids collapsed ≥2 episodes, 7 collapsed ≥2 shows**. Episode-scoping → **all 0**. Does NOT name more voices. |
| **#2** publisher denylist | A network/publisher (`The New York Times`) is never a host/guest | Removes *wrong* names; not a reducer. |
| **#1c** diagnostics sidecar | Per-episode `what we tried / resolved / why unresolved` | The measurement substrate — makes "why still unknown" auditable. |
| **#3** host/guest role on card | Per-show role surfaced in the player | UX; not a reducer. |
| **voice_type** cameo/commercial/unknown | Classify each *unnamed* voice so noise is labelled and only real people count as "unknown" | **RECLASSIFIES:** of 326 unresolved voices, **196 (60%) are noise** — 195 cameo (<20s) + 1 commercial — leaving **130 real people** to chase. Carried on the roster + diagnostics; `display_label_for()` renders "Brief speaker"/"Advertisement". |
| **Step B** per-feed `known_hosts` | Name a network feed's recurring host that never self-introduces | **NAMES MORE:** 253 → **272** named (43.7% → **47.0%**), **+19** (recurring hosts named in episodes with no "I'm …": Alexi Horowitz-Ghazi ×5, Katie Martin ×4, Brandon ×4). Config-only, wired into `feeds.spec.yaml`. |

**Cumulative on prod-v2:** named **212 → 272** (36.6% → **47.0%**) via #1a + Step B; of the
**307 still unresolved**, **~196 are labelled noise** (cameo/commercial), leaving **~111 real
people** to name. #1b is a correctness fix orthogonal to naming.

---

## Where the remaining unknowns are (prioritisation)

After #1a: **326 / 579 voices still unknown (56.3%)** — but they hold only **~24% of talk
time** (named voices hold 76%). So most remaining unknowns are **short cameo interjections**
(a few seconds), not the people carrying the episode. Chase the high-talk-time unknowns
first; the long tail of 5-second cameos is low value.

---

## Reducer roadmap (ordered by leverage / cost) — each with a before/after plan

### Step B — per-feed `known_hosts` config  ·  cheapest, highest yield
Network feeds whose host never self-introduces (author tag is the org) stay `SPEAKER_NN`
forever. Hand-name the recurring hosts per feed in config; the roster already consumes
`known_hosts`. **Measure:** inject `known_hosts` for the top-N feeds, re-run `measure_1a.py`,
report the additional named voices + talk-time coverage.

### Step C — make host reconciliation usable on real data  ·  medium
`reconcile_hosts` (#1056) names a recurring unnamed host from a sibling episode — but
**measured fuel on this corpus = 0**: unnamed hosts are dropped by the
`startswith("speaker")` filter in `metadata_generation` before they ever reach the KG as a
`role=host` node, and the `person:speaker-NN` nodes that *do* exist come from the GI
(SPOKEN_BY) layer with **no role**. So the lever is real but currently starved. Precursor:
either (a) carry an unnamed *host* voice into the KG as a `role=host` node (so reconcile can
merge it), or (b) extend reconcile to GI person nodes keyed on feed. **Measure:** build the
corpus graph reconcile-off vs reconcile-on, count voices named/tagged.

### Step D — NER on the transcript intro for guests  ·  medium
The self-intro regex catches "I'm X"; it misses host-introduces-guest phrasings ("joining me
today is X", "my guest X"). Run NER / a pattern set over the intro window and feed matches
into `detected_guests`. **Measure:** re-run `measure_1a.py` with the enriched guest list.

### Step E — cross-episode voice fingerprinting  ·  out of scope (ML)
The real fix for recurring anonymous hosts, but explicitly a non-goal in #1056 (no ML voice
ID). Parked.

---

## Open decisions for the operator
1. Order: B → D → C (config-first, then guest NER, then the reconcile precursor)? B is a
   config change with no code risk and likely the biggest single jump on network feeds.
2. Promote the three measurement scripts to `scripts/measure_speaker_naming.py` + a
   `make measure-speakers` target so before/after is a repeatable gate, not scratchpad?
3. Chase only high-talk-time unknowns (ignore the <30s cameo tail), or full coverage?
