# v3 enricher content design — the six structures, grounded in the existing feeds

Design for the #1148 **content** pass (authoring), for operator review before I write
~hundreds of lines into `build_v3_spec`. The **plumbing** (schema + render + gold emission +
the eval framework/gate) is shipped and green; this sketches *what content to author* so the
enrichers produce real output and the scorers/gate read real `data/eval` metrics.

## Anchor the overlap on topics that already recur (don't invent)

The existing 9 feeds already cross on a few topics — reuse them as the master lever rather than
add new vocabulary:

| Shared topic | Already in | Use for |
|---|---|---|
| `topic:risk-management` (12×) | p05 investing, p01/p02 software, p04 agriculture | perspectives, cooccurrence, opposition, velocity |
| `topic:systems-thinking` (5×) | p01 software, p04 agriculture | second overlap axis, similarity neighbour |
| `topic:reliability` (11×) | p01/p02 software | grounding/density, cooccurrence with risk-management |

## The six structures → concrete authoring

1. **Time spread** — stamp `publish_offset_days` on the risk-management episodes across
   p05/p01/p04 at `0, 45, 90, 135, 180` (epoch `2026-01-01`) so `temporal_velocity` sees
   risk-management heating up.
   *Gold:* corpus `expected_enrichment.temporal_velocity` (risk-management velocity ratio ↑).

2. **Cross-speaker topic overlap** (master lever) — add `topic_claims` on `risk-management`
   attributed to **≥3 distinct speakers across ≥3 shows** (a p05 investor, a p01 SRE, a p04
   farmer), plus systems-thinking claims linking p01↔p04.
   *Gold:* corpus `topic_similarity` (risk-management ~ systems-thinking / reliability),
   `topic_cooccurrence_corpus` (risk-management × reliability, × index-investing), and
   per-topic **perspectives** (≥3 speakers on risk-management → de-mocks the #1146 e2e).

3. **Engineered opposition** (the nli/disagreement positive) — one **roundtable episode** where
   two named guests take opposite stances on the *same* proposition. Proposed proposition on
   `topic:risk-management`:
   - **A (investor):** "Diversification is the only real risk control — concentration is how you blow up."
   - **B (SRE/operator):** "Concentration with deep understanding beats diversification — spreading thin *is* the risk."
   *Gold:* corpus `contradiction_pairs` + `expected_enrichment.nli_contradiction` (the one true
   cross-person contradiction). This is what could **auto-promote nli** if a real DeBERTa run
   scores it ≥ 0.5 precision.

4. **Multi-guest** — the same roundtable carries `additional_guests=[second guest]` → a real
   co-appearance pair (two birds).
   *Gold:* corpus `expected_enrichment.guest_coappearance` (the pair).

5. **Grounding + density** — mark ~2 episodes `insight_density="low"`; author some
   `topic_claims` with `grounded=true` and some without.
   *Gold:* per-episode `expected_enrichment.grounding_rate` / `insight_density`.

6. **Seeded users** — 2 users whose heard sets cover the shared topics:
   - `u_risk`: heard the risk-management episodes across p05/p01/p04 → interests
     {risk-management, reliability}; scope=mine perspectives on risk-management.
   - `u_invest`: heard p05 only → interests {index-investing, macroeconomics}.
   *Gold:* `expected_interests`, `expected_ranking`, `expected_scope_mine_perspectives`.

## Footprint

Small + additive: ~1 new roundtable episode (opposition + multi-guest), `topic_claims` on ~4
existing episodes, `publish_offset_days` on ~5, `insight_density` on ~2, 2 seeded users +
`build_v3_corpus_meta` gold. No new topic vocabulary, no failure-mode-vocab change (keeps the
coverage tests green).

## Gold is authored, then reconciled by the loop run

The gold *values* above are authored as the *expected* enricher output, then the RUN-LOOP step
(build corpus → run enrichers → scorers) reconciles them: deterministic enrichers
(guest_coappearance, grounding, cooccurrence, velocity) are exact; the embedding/NLI ones
(topic_similarity, nli) get their gold from an independent judgment and are *measured* against
it (eval-first, same as #1105/#1106).

## Operator decisions before I author

1. **Anchor topic** — risk-management (recommended, most cross-domain) vs systems-thinking.
2. **Opposition proposition** — the diversification-vs-concentration one above, or your call.
3. **New episode vs augment** — add one roundtable episode, or fold the opposition into an
   existing p05/p01 episode.
