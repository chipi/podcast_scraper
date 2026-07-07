# ADR-108: Reimagine the NLI enrichers — `topic_consensus` (embedding + low-contradiction composite, activated) + retire stance scoring for a read-time timeline

- **Status**: Accepted (2026-07-07; supersedes the 2026-07-07 "gated dark pending a precise scorer" decision). `topic_consensus` **validated + activated** (precision 0.91 on prod-v2); the `stance_timeline` enricher was **retired** in favour of a read-time CIL conversation/position timeline coloured by the deterministic `insight_sentiment` (VADER). See the 2026-07-07 updates below.
- **Date**: 2026-07-07
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**:
  - [ADR-104](ADR-104-enrichment-layer-boundary-vs-kg-direct.md) — the enrichment-layer
    boundary these enrichers live behind.
- **Related RFCs**:
  - [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md) — enrichment layer + the
    accuracy-gate amendment (`RFC-088-ENRICHER-ACCURACY-GATE-2026-07.md`) this ADR relies on.
  - [RFC-103](../rfc/RFC-103-momentum-layer.md) — the read-time time-series layer `stance_timeline`
    plugs into.

## Context

RFC-088 shipped `nli_contradiction` (cross-Person Insight *contradiction* pairs per Topic via a
DeBERTa cross-encoder `NliScorer`), and #1144 built `stance_disagreement` (a stance-level, no-LLM
successor that aggregates each speaker's stance and scores the pair symmetrically). Both aimed to
surface *"these two speakers disagree."*

**The accuracy evals killed both** — and the *why* is the load-bearing lesson:

- **`nli_contradiction`** — **0% precision** on prod-v2 (150-pair Opus silver, #1106). The
  cross-encoder scores "contradiction" on merely *topic-adjacent* insight pairs.
- **`stance_disagreement`** — **0% precision** (stance-aggregate) / 10% (atomic-max) vs the #1144
  gold. The designed positive (Cho-vs-Bessent debate, 0.999 atomic-max) is indistinguishable from the
  `no_shared_question` negatives (0.96–0.999).

The missing piece is a **shared-question gate**: do A and B engage the same contested *proposition*,
not just the same *topic*? That is a semantic judgment DeBERTa structurally lacks, and the operator
ruled out new LLM-dependent enrichers. So *asserting disagreement* is unreachable without an LLM.

**But the model isn't broken — the task framing was.** Sentence-pair NLI is answering the wrong
question well. Two moves make the same assets (the DeBERTa scorer, the pair/series + accuracy-gate +
gold framework, and the new RFC-103 time-series layer) productive **without an LLM**: pick tasks
where the shared-question gap either does not arise, or is approximable from signals we already have.

## Decision

**Reimagine both enrichers from scratch**, abandoning "disagreement-as-assertion" for the robust
**entailment** side of NLI. Each old enricher's core machinery maps to the task it can actually win:

### 1. `nli_contradiction` → `topic_consensus` (cross-speaker corroboration)

Flip the sign: instead of "who disagrees" (0%), detect **what the corpus agrees on**. Cross-speaker
Insight pairs (same Topic) whose relation is **entailment** → "N speakers corroborate X," "this claim
recurs in episodes Y, Z." Entailment is far more robust to topic-adjacency than contradiction
(aligned claims genuinely entail; adjacent ones merely co-occur). **Non-LLM shared-question proxy:**
NLI-entailment **×** tight embedding proximity (RFC-088 `topic_similarity` infra) — "same proposition,
not just same topic." Value: corpus consensus is a headline signal and a grounding/dedup input.

### 2. `stance_disagreement` → `stance_timeline` (per-person stance over time)

Reindex per-speaker stance by **time**: for each `(person, topic)`, track the speaker's stance across
episodes and detect **deviation** (e.g. *"AI is bad" → "AI is not realistic" → "AI is good"* — a
flip from opposed to supportive). **Same-speaker + same-topic makes the shared-question gate FREE**
(one person always argues their own take on T) — it deletes the exact failure mode.

- **Stance scoring (no LLM):** NLI-**entailment against fixed stance anchors** — `H+ = "{topic} is
  good/promising"`, `H− = "{topic} is bad/overhyped"`; **stance = entail(H+) − entail(H−) ∈ [−1, +1]**.
  This is the *canonical* NLI setup (premise = insight, hypothesis = anchor) — the half DeBERTa is
  genuinely good at.
- **Deviation (deterministic):** over the stance series, range, sign-flips, and regression slope →
  *"shifted from opposed to supportive."*
- **Plugs into RFC-103:** a `(person × topic)` stance series *is* a read-time time series; its
  **derivative** is a "**trending opinion shifts**" momentum signal.

### Common

Both keep the data-driven accuracy gate (`accuracy_gate(precision ≥ 0.5)`), so nothing ships until it
clears its gold — gold comes from the v3 **position-arc / debate** fixtures (#1148). `perspectives`
(#1146) stays the live neutral side-by-side; these enrichers **add** the agreement axis
(`topic_consensus`) and the trajectory axis (`stance_timeline`), each a claim the data can back.

## Update — 2026-07-07: real-corpus validation corrects the consensus signal + activates it

Ran both enrichers with the real DeBERTa over prod-v2 (99 bundles; full findings in
[docs/wip/ADR-108-REAL-CORPUS-EVAL-2026-07.md](../wip/ADR-108-REAL-CORPUS-EVAL-2026-07.md)). The eval
**refuted the original `topic_consensus` mechanism and confirmed a corrected one**:

- **`topic_consensus`: symmetric NLI *entailment* fails; the composite works.** The first cut gated on
  *symmetric entailment* ("symmetry IS the shared-question gate"). On real data that has **~0 recall** —
  1 pair over 2 903 scored — because genuine agreement between two speakers is phrased differently, and
  entailment ≈ paraphrase. The signal that actually recalls agreement is a **composite**: **embedding
  cosine ≥ 0.70** (the *shared-question* gate — same proposition) **AND NLI contradiction ≤ 0.5** (the
  *direction* gate — they don't disagree; this filters the similar-but-opposite pairs cosine alone
  admits). Measured **precision 0.91** (20/22) over a curated 28-pair gold → auto-promoted through the
  accuracy gate. So the ADR's "shared-question proxy" is **embedding proximity**, not symmetric
  entailment; the contradiction check replaces the (weak) entailment check. Implemented as a single
  `ConsensusScorer` provider (`consensus_local`: MiniLM + DeBERTa, both CPU-local — still no LLM).
- **`stance_timeline`: the signal is genuinely absent on prod-v2** — factual insights don't entail
  evaluative anchors, so stance ≈ 0 (and only 13 (person, topic) trajectories have ≥2 dated points). A
  `sign_flips`-on-noise bug even flagged 11/12 flat trajectories as "shifted". So auto-scoring a stance
  was the wrong frame here.

## Update — 2026-07-08: retire the `stance_timeline` enricher for a read-time timeline + sentiment colour

The stance *score* was the mistake, not the goal. A "how did this person's / the corpus's view on a
topic evolve" timeline already exists as **read-time CIL queries** — `GET /api/persons/{id}/positions`
(per-`(person, topic)` arc) and `GET /api/topics/{id}/timeline` (all speakers). The `stance_timeline`
enricher was a worse, gated reinvention of those. So it was **removed** (like `nli_contradiction`),
and the direction pivoted to what actually scales + is honest:

- **`insight_sentiment`** — a *deterministic* VADER enricher (pure-Python lexicon, no model / no
  network) tags every Insight with a compound + neg/neu/pos label. Unlike a stance *score* (useless
  when neutral), sentiment is a **decoration** — a neutral factual insight is a fine grey.
- **`GET /api/topics/{id}/conversation-arc`** — the *aggregate-first* answer for a big topic (1000s of
  insights): weekly **volume × sentiment** buckets, so the UI shows the *shape* of the conversation and
  drills into a week on demand, never rendering thousands of cards. The position-arc / topic-timeline
  insights now carry `sentiment` for per-card tint.

Net: per-person + global stance-over-time ships as timelines the corpus can actually back, without a
gated ML enricher or an LLM.

## Consequences

- The DeBERTa model is reused on its **strong** task (entailment vs a hypothesis / contradiction
  detection), not its weak one (cross-speaker contradiction as a positive signal). No LLM added; the
  shared-question problem is dissolved by task design (same-speaker) or by **embedding proximity +
  contradiction filtering** (consensus, per the 2026-07-07 update).
- The viewer's orphaned "Contradictions" render code is **repurposed/replaced** by consensus +
  stance-timeline surfaces, not left dark.
- The accuracy gate remains the durable firewall: each new enricher is a *data* exercise (clear its
  gold), never a hand-toggle. `stance_disagreement`'s `gold_v1.jsonl` is retained/relabelled as the
  `stance_timeline` regression bar.
- Cross-layer rebuild required: scorers (entailment + stance-anchor), manifests + IDs, gold + gate
  metrics, and the Person/Topic + Dashboard surfaces (stance sparkline; corpus consensus).

## Alternatives considered

- **Keep waiting for a precise *disagreement* scorer** (the prior decision) — rejected; the eval
  argues DeBERTa can't clear it without an LLM, so waiting is waiting forever. Changing the task is
  the unlock.
- **An offline LLM judge** (shared-question gate as an LLM prompt at enrichment time, never CI) —
  rejected by operator constraint (no new LLM-dependent enrichers).
- **Delete the enrichers** — rejected; the machinery + gold + gate are exactly what the reimagined
  tasks reuse.
- **Ship the imprecise detector** — rejected; recreates #1106 (fabricated disagreements).
