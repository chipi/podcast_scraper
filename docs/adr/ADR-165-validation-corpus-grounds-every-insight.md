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
   segment that contains its sentence, with that segment's offsets, times and speaker. The speaker
   is the Person named as the segment's speaker label; the old quote edges are only a fallback,
   because they credited people who never speak.
2. **Insights are claims.** These are not insights:
   - questions, in any language (a question mark);
   - greetings, and thanking the host by name;
   - sponsor reads;
   - the generator's connective templates: a callback, a name mention, a segue and its generic
     reply;
   - host prompts that end in a full stop ("Unpack that.", "Walk me through a concrete example.").
3. **The claims chosen are representative.** Each speaker first contributes their first claim
   about one of the episode's topics (or their first claim), then the remaining slots go to topical
   claims. An episode keeps its insight count: removed non-claims are replaced by the next claims.
   Real extraction yields many insights across the speakers. Three that are all the host's left
   guests unquoted, which emptied the people and co-appearance surfaces.
4. **Authored claims are anchored.** An authored claim's quote is the one segment that contains it;
   a paraphrase gets the verbatim line it paraphrases (that speaker's segment sharing the most words,
   at least a third), and the insight keeps the paraphrase. No time is invented.
5. **Non-English episodes are grounded in their own language.** Insights are matched against the
   source-language segments (`<stem>.<lang>.segments.json`). A line is classified through its
   English translation only in the opening run, where index pairing with the translation's turns
   is verified by a shared name; further in, the pairing drifts.
6. **The builder's grounding variation is kept.** In `low_grounding_dialogue` episodes every second
   pipeline insight stays ungrounded, so the grounding-rate enricher still has variation to measure.
7. **The builder and the committed corpus follow the same rules.** The builder is fixed
   (`build_gi`, insight selection, `authored_quote_anchor`). Because a rebuild does not reproduce the
   committed corpus (later edits live only in the committed files), a migration applies the same
   rules in place (`scripts/tools/realign_fixture_gi.py`, dry run by default, idempotent), and
   everything derived from the GI is regenerated:
   - the per-episode `insight_density` and `insight_sentiment` envelopes, with the builder's own
     functions;
   - the corpus `grounding_rate` and `guest_coappearance`, with `cli enrich`. This also moves
     `grounding_rate` to the per-episode shape the server has read since #1927;
   - the search sidecar, which is gitignored and rebuilt by the tests' index fixture.

**Result** (2026-10-11):

- All 178 insights are backed by their own quote, with times.
- No question, greeting, sponsor read (in any of the six languages), connective or prompt insight
  or quote remains; ad-labelled lines are never claims.
- Every episode keeps its insight count (178 in all, as before), each backed by its own quote.
- Co-appearance went from 20 to 36 real speaker pairs, keeping all 20.

## Consequences

**Positive**

- The fixture can prove extractive surfaces: a card's quote is something an insight says.
- Reels and Moments clip lines that mean something.

**Negative**

- Pinned evals and Moments e2e expectations change where the data honestly changed (each re-pin
  was read before it was frozen).
- One property, "a storyline gathers a speaker its anchor topic does not have", held on the fixture
  only through a misattributed quote. Its test now runs on data built for it.

**Neutral**

- Production data is unaffected; the read-time support rule (`best_grounded_quote`) already guards
  production surfaces against the same failure.

## Alternatives Considered

- **Filter at read time only.** Kept as a guard, but rejected as the fix: the fixture would keep
  teaching tests that a greeting is a valid insight.
- **Regenerate the corpus from the builder.** Rejected: it does not reproduce the committed files
  (153 differ), so later fixes would be lost.

## Amendment 2026-10-11: three storylines

The corpus carried one storyline, so "Continue the thread" could never take turns, hiding the
storyline emptied that rail, and every storyline reel, path and door test landed on one subject
(measured: 1 storyline, 14 themes, 6 topics with a reel). The builder now authors three disjoint
storylines (`STORYLINES`), and `scripts/tools/regen_fixture_storylines.py` applies them to the
committed file from the committed graphs, refusing to move an existing one. *Second-order effects*
has a 24-clip reel and a seven-day path; *Building trails that last* has neither, on purpose. The
file's summary counts are recomputed from today's committed corpus (407 topics; the old 50 had gone
stale). One audit figure moved where it measures the corpus: the picker band gained the trail
storyline (112 → 113) and its top 12 lost one distinct feed set (10 → 9), because that storyline and
its broadest member cover the same nine episodes.
