# ADR-167: Identity guards run inside the join; a cautious split is acceptable, a wrong merge is not

- **Status**: Proposed
- **Date**: 2026-10-30
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-131](../rfc/RFC-131-person-identity-one-home.md)
- **See Also**: [ADR-166](ADR-166-person-identity-ledger-no-joins-without-it.md)

## Context & Problem Statement

The cross-episode identity map joins two person ids when their names pass a spelling rule. The
build (`kg/entity_clusters.build_entity_canonical_map`) joins a name to a cluster when it matches
**any** member. So clusters chain: A is joined through B to C, though A and C were never compared.

Spelling cannot tell a misspelling from two people with near-identical names. Two co-presenters
whose one-word first names differ by one letter are joined today, though the same episode places
them on different voices and the publisher lists them both. Removing such a join afterwards does
not work either: members chained through the removed one stay attached.

The operator's rule: a wrong name is worse than a hidden one.

## Decision

1. **A name joins a cluster only if it matches some member and no guard fires against any
   member.** This replaces "matches any member".
2. **Guards are deterministic evidence of two people.** Each blocks a join whatever the spelling:
   - **co-voice:** the two names are placed on different voices in one episode. There is no
     exemption: voices that diarization merged by fingerprint are already one voice before naming;
   - **publisher list:** the publisher's text names both in one episode, and each is placed on a
     voice or is a Person node there;
   - **mention co-occurrence:** both are distinct Person nodes in one episode, and at least one is
     only mentioned. This verdict is *unknown* (no join), and is reported apart from the two
     guards above, whose verdict is *two people*.
3. **Guard evidence is built once per episode, before clustering.** For a candidate and a member,
   the guards run only over the episodes the two share. Every blocked join is recorded in the
   identity ledger with its guard and episode (ADR-166).
4. **A split a guard causes is accepted and listed, never fixed by weakening the guard.** It is
   listed when the two names are spelling variants of each other. Those lists go to the operator.
   Success is measured as splits down with every guard-caused split listed, and wrong merges at
   zero.
5. **The residual no guard reaches is stated, not hidden:** two different people who never share
   an episode. There, only the rule's conservatism and read samples of new joins stand between
   them and a merge.

## Rationale

- **Guards inside the join make chains respect them.** A guard applied afterwards cannot undo a
  chain.
- **The guards use evidence already on disk:** the roster's placed voices, publisher text and graph
  nodes. They need no model and no reprocess.
- **The two kinds of error do not cost the same.** A split person is shown twice but correctly
  named. A merged person shows one human's words under another's name.

## Alternatives Considered

1. **Tighten the spelling thresholds.** Rejected: near-identical names of two people (one letter
   apart) and one-letter misspellings of one person are the same string distance, so no threshold
   separates them.
2. **Post-filter the finished map.** Rejected: members chained through a removed join stay
   attached.
3. **Exempt voice pairs that diarization's split-voice report marks as one person.** Rejected:
   those voices are merged before naming, so the exemption could never fire. That report is used
   as a negative check instead: voices merged by fingerprint must never appear as two roster
   entries.

## Consequences

- **Positive:** two people placed on different voices, or listed together by their publisher, can
  never be one person; every blocked join is explained in the ledger.
- **Negative:**
  - some true same-person pairs stay split: the same person as a voice and as a differently spelled
    mention, or one person diarized as two voices and named twice;
  - these are listed and counted, and the operator decides on them.
- **Neutral:** join order no longer decides a guard, because the evidence is per episode and the
  check runs against every member.

## Implementation Notes

- **Module:** the person-identity module of RFC-131, with
  `EpisodeEvidence(voice_of, mention_only, publisher_named)` keyed by episode.
- **Measure:** `name_report`:
  - R3a for splits, with guard-caused splits listed;
  - R3b for wrong merges per mechanism, chain joins included;
  - R6 for every join added or removed corpus-wide.
