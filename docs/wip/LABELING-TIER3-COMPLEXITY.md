# Tier-3 complexity: swappable labeling strategies (A/B by profile, minimum code per experiment)

- **Status**: Analysis (ADR-140 tier 3). Not scheduled.
- **Date**: 2026-07-29
- **Goal being priced**: an agent proposes a *new labeling algorithm* (not just a knob), registers a
  new profile, and A/Bs it by switching `labeling_profile` — with **minimum code change per
  experiment**.

## What tier-2 already covers (cheap, done)

- **A new knob** (a threshold): add one field to `LabelingProfile`, read it at the site. ~5 lines.
- **A new additive step / feature flag** (a new intro form, a new name source, an on/off toggle):
  add the function, gate it on a profile flag, thread the flag. ~1 function. This is exactly how
  today's ADR-139 fixes + Pattern-B are already switchable.

So for the *common* agent experiment — "try this threshold", "turn this heuristic on/off", "add
this cue" — **tier 2 is already the A/B mechanism**. No tier-3 needed.

## What tier-3 actually requires — and why it costs more

Tier-3 is for swapping a whole **sub-algorithm** of a stage (a different way to *discover names*, a
different *classification scheme*), not a knob. The cost is not the swap — it is **defining the
seam** first, because the labeling stages are coupled through shared mutable state.

Measured coupling (roster.py): **13 mutation sites** of four shared structures —
`by_voice` (the roster being built), `used_lower` (claimed names), `voice_intro` (discovered
names), `nameable` (classification input) — threaded through **7 stage functions**
(`_self_intro_voice_names`, `_intro_reader_voice_names`, `_name_host_voices`, `_name_guest_voices`,
`_recover_stated_names`, `_classify_voice_types`, `classify_voices`). A stage is not a clean
`input → output` function today; it mutates the shared roster in place, in sequence.

**To make a stage swappable you first give it an explicit contract** (a Protocol: declared inputs →
declared outputs, no side effects on shared state), then register implementations and let the
profile select one. The refactor to remove the shared-state coupling *is* the cost.

## The seams, ordered by extraction cost (do cheapest first)

| Seam | What it decides | Coupling | Cost | A/B value |
|---|---|---|---|---|
| **Alarm / census policy** | defect-vs-total, threshold, census shape | already parameterised | **~0 (done)** | new alarm rule = a flag/fn on the profile |
| **Voice classification** | unknown / unidentified / cameo / commercial (Pattern-B lives here) | reads `by_voice`+talk+ad signals → typed `by_voice`; fairly clean I/O | **LOW–MED** — one Protocol, move `_classify_voice_types` + the nameable/promotion logic behind it | **highest** — this is where classification experiments cluster |
| **Name discovery** | self-intro + cue-intro + LLM → `voice_intro` | spread over 2 functions + LLM + montage/reclaim guards | **MED** | new discovery algorithm (e.g. embeddings, a different LLM prompt) |
| **Roster assembly** | `voice_intro` + name pools → `by_voice`, one-name-one-voice, forced match | mutates `by_voice`+`used_lower` in sequence; most tentacles | **HIGH** | rarely the experiment target |

## Recommended path (per-seam, incremental — not a big-bang)

1. **Extract the classification seam first.** It has the cleanest I/O, it is where Pattern-B and the
   next classification experiments live, and the profile already carries its flags. Define a
   `ClassificationStrategy` Protocol (`(by_voice, talk, signals, profile) → typed by_voice`), move
   the current logic behind a `pattern_b` strategy + a `legacy` strategy, and add a
   `classification_strategy: str` selector to `LabelingProfile`. After this, an agent's classification
   experiment = **one strategy class + one profile entry** — minimum change, A/B by switching.
2. **Extract name-discovery next, only when a discovery-algorithm experiment needs it.**
3. **Leave roster-assembly coupled** until an assembly experiment justifies the HIGH-cost refactor —
   YAGNI; it is rarely the experiment target.

## Cost estimate

- **Classification seam:** a bounded, few-hour refactor (Protocol + move logic + register 2
  strategies + thread the selector + a regression test that the default is byte-identical). This is
  the one worth doing to unlock automated classification A/B.
- **Full four-seam extraction:** larger, but **strictly incremental** — each seam is independent, so
  it is done when an experiment demands it, never all at once. The one-time cost is amortised across
  every future experiment; the per-experiment cost drops to "a strategy + a profile".

## Bottom line

The expensive part of tier-3 is **breaking the shared-state coupling to define each seam**, and that
cost is **per-seam and pay-as-you-go**, not a rewrite. Tier-2 already handles knob/flag/step
experiments. When you want to A/B a genuinely new *classification* algorithm, extracting that single
seam (~a few hours, one-time) is what makes every subsequent experiment a minimum-change strategy +
profile. I would extract the classification seam when the first such experiment appears, and not
before.
