# ADR-165: The validation corpus grounds every insight in what is said

- **Status**: Accepted
- **Date**: 2026-10-11
- **Authors**: Marko Dragoljevic
- **See Also**: [ADR-163](ADR-163-the-jobs-of-discover.md) §4 (the extractive reason),
  [RFC-126](../rfc/RFC-126-discover-rails-reasons-paging-and-reels.md),
  `tests/fixtures/app-validation-corpus/README.md`, `tests/fixtures/FIXTURES_SPEC.md`

## Context & Problem Statement

The committed app validation corpus (`tests/fixtures/app-validation-corpus/v3`, 55 episodes) is
what the player's e2e suite, the rail evals and the screenshots run on. Measured on 2026-10-11:

- The builder handed quotes to insights round-robin (`insight_ids[qi % n]`): of 162 pipeline-shaped
  insights, 75 were backed only by a neighbouring insight's quote and 12 by none.
- 37 sponsor reads and 37 greetings ("Thanks, Sam. Great to be back.") were quotes.
- 31 "insights" were host questions and 2 were ad reads: transcript lines, not claims.
- Authored quotes were stamped at 0 ms; 7 of the 16 were paraphrases, not transcript text.

Surfaces built to be extractive (ADR-163 §4) therefore showed ad reads and greetings as the reason
an episode was recommended, and a topic reel had almost nothing it could honestly play.

## Decision

1. **A quote backs its insight or does not exist.** Each insight is supported by the transcript
   segment that contains its sentence, with that segment's offsets, times and speaker.
2. **Insights are claims.** Questions, greetings and sponsor reads are not insights.
3. **Authored claims are anchored.** An authored claim's quote is the one segment that contains it;
   a paraphrase gets the verbatim line it paraphrases (that speaker's segment sharing the most words,
   at least a third), and the insight keeps the paraphrase. No time is invented.
4. **The builder and the committed corpus follow the same rules.** The builder is fixed
   (`build_gi`, insight selection, `authored_quote_anchor`). Because a rebuild does not reproduce the
   committed corpus (later edits live only in the committed files), a migration applies the same
   rules in place (`scripts/tools/realign_fixture_gi.py`, dry run by default), and everything derived
   from the GI is regenerated: the per-episode enrichments, the corpus enrichments that read the GI,
   and the search sidecar.

## Consequences

**Positive**

- The fixture can prove extractive surfaces: a card's quote is something an insight says.
- Reels and Moments clip lines that mean something.

**Negative**

- Pinned evals and Moments e2e expectations change where the data honestly changed.
- Fewer insights per episode in some episodes (the non-claims are gone).

**Neutral**

- Production data is unaffected; the read-time support rule (`best_grounded_quote`) already guards
  production surfaces against the same failure.

## Alternatives Considered

- **Filter at read time only.** Kept as a guard, but rejected as the fix: the fixture would keep
  teaching tests that a greeting is a valid insight.
- **Regenerate the corpus from the builder.** Rejected: it does not reproduce the committed files
  (153 differ), so later fixes would be lost.
