# Advisor review of SPEAKER-NAMING-STRUCTURE-2026-10-02.md (Fable, read-only, prod-measured)

Reviewed at wt-kg HEAD `1f1cf5f5b`; prod `sha-0ee4580`. All prod work read-only (scratch in
`compose-api-1:/tmp/rp_adv`). Recorded verbatim in substance; the decision on what to do is open.

## 1. Diagnosis

Mechanics right; one case trace wrong; the largest cause of "host known, not placed" missing.

Verified: §3.1 (`hosts.py:1657-1663`, guards `:701-738`), §3.2 (`hosts.py:639`, `_NAME` `:626`,
`.search` `:693`; 30 feeds no host + 2 rejected = 32/80), §3.3 (`processing.py:804` no `nlp`;
`openai_provider.py:1457`, `:1467-1471`, `:1733` dead branch), §3.4 (`processing.py:1941-1946`,
`:1998`; `cues.py:46-48` parses only `Speaker N:` on purpose — `MG:` on the first cue only),
§3.5-3.7 (`roster.py:2686`, `:1747-1748`, `:1812-1856`, `:2886` vs `:2927`, `config.py:3996`).

- `_GUEST_HOST_EPISODE` IGNORECASE on 2,102 newest-run episodes: 18 matches, 8 genuine, 10 false
  ("sitting in some folks", "in for a snapback", "stand in for real evidence", "Tune in for this",
  "dropped in for the weekly"…). Which of the 10 blocked a name: not measured.
- Counts: 788 placement records, 58 with a host name and no placed host (doc: 786 / 56).

### Missing defect — role speech acts are cluster-global
Instrumented replay of the deployed roster over the 58 "host known, not placed" records:

| cause | n |
|---|---|
| host's own voice is in `conv_guests` (a guest speech act anywhere in its cluster) | 16 (+2 LLM called every voice guest) |
| seat exists, >1 pool name unclaimed → forced rule declines | 27 (a16z 8, Daily 5, Hard Fork 4, Journal 3, Latent Space 2, No Priors 2, Odd Lots, LAiF) |
| HEAD already places the host (record stale vs wt-kg) | 8 |
| genuine "guest host" refusal | 3 |
| veto / error / publisher path | 4 |

The 16: Dwarkesh ×3, Pragmatic Engineer ×3, The Daily ×2, Latent Space ×2, How I Write, Unhedged,
Ground Truths, Unbelievable?, Business of Africa, NN/G. `roles_from_conversation`
(`hosts.py:1404-1410`) matches over the whole cluster; steps 3/4 (`roster.py:2902`, `:2929`)
refuse any `conv_guests` voice. **How I Write is this**: the host voice (562 s, opens "Andy Weir is
on the show today") carries a bled guest line "Thank you. It's great to be here." at the end of its
cluster → `_GUEST_SPEECH_ACTS` (`hosts.py:800`) → no seat. Rule 2.6 as drafted inherits this.

### Other corrections
- Rule 0 "union of today's checks" does not reject `Two Carnegie Mellon`, `Timmerman Report`,
  `Premier Unbelievable?` — every existing predicate returns person. A new predicate is needed
  (number words, `Report`, trailing `?`, the show's own title echo). Today these are blocked only by
  first-hit order; Rule 1 removes that accidental protection.
- Same-person is exact-lowercase in `stated_non_host_voices` (`roster.py:2838-2846`): self-intro
  "Alistair Campbell" vs stated "Alastair Campbell" marks the real host not-a-host.
- Pool pollution: an a16z record has 11 `known_hosts` incl. the episode's own guest and
  `'Greg Brockman)'`. Source not verified.
- Empty pool → the LLM's "host" verdict is unchecked: Conversations with Tyler has 4 GUESTS
  published as host via `llm_resolution` (Julia Ioffe, Joel Mokyr, Craig Newmark, Annie Lowrey).
- "Rejected statement blocks author tags" costs nothing on today's 80 feeds (both rejected feeds
  have org-only authors). Structural, not currently harmful.
- Provenance mislabel confirmed (LAiF: Carin Zissis labelled `feed_statement`, came from
  episode-level authors, `processing.py:1252`).
- "83 publisher episodes without turns" vs advisor's 1 plain-text-with-labels + 4 cue ≤1-voice on
  newest-run records: different denominators, not reconciled.

## 2. Is the rule set simpler AND safe?

Simpler on paper; **not safe as written** — each rule has a measured prod loss.

- **Rule 1 collect** (replayed, 2,102 eps; pool changed in 252): gained 51 / lost 27 / changed 25.
  Gains: Tyler Cowen ×38, a16z hosts ×8, Tim Scarfe, Dan Hooper, William Armstrong ×7, Richard
  McColl, Matt Clifford, Dan Neidle; junk removed `Americas Online` ×21, `Machine Learning Street`.
  Real losses (~12): Alistair Campbell ×2 (spelling), `Timmerman Report` as HOST ×2 plus 5 Long Run
  guests relabelled host and 2 lost (junk author tag), Rory Stewart host→guest, Eric Olander →
  "Eric Olin", Robert Armstrong split into "Rob Armstrong" ×6. That is the trade the operator
  rejects — unless Rule 0 lands first and the pool merges spellings with `same_person`.
  #2075 / Carin Zissis: under HEAD seat logic her name in the pool painted zero wrong voices on 40
  LAiF episodes; the early-return reason holds only under the deployed seat logic (not replayed).
  The LLM closed-list effect of a larger pool is unmeasurable offline.
- **Rule 2.4 vocatives**: 31/788 records greet at the open. Duo: 9 agree, 2 disagree (a16z
  "Hey, Anish" — vocative would fix it). Panels: 2 agree, 2 wrong (No Priors "Welcome, Jared" → next
  voice is Elad; Planet Money two addressees). → duo-only, single addressee, open-anchored.
- **Promo rule**: 36 placed hosts self-introduce near a show mention, all THIS show but ASR-spelled
  ("Odd Lodge", "Odd Lads", "Google Deep Mind the podcast"). Needs fuzzy this-show match and
  first-person only (3 correct voices mention another show's host in the third person). Would catch
  Phoebe Judge, Catherine Benhold ×8, Tracy Mumford ×7.
- **Rule 2.6 one dominance guard everywhere** (replayed): gained 3 (none real), lost 7 incl. Shalma
  Wegsman (64%) and David Tizzard (74%). Do not put the dominance guard on the opener seat.
- **Rule 3**: `Name:` parsing must require labels on most cues (the `MG:` case).

## 3. Migration — ordered, each step isolated and replay-gated (old side = HEAD roster)

1. Immediate bugs: case-sensitive `in for [A-Z]`, drop `sitting in`; docstring; provenance label.
   Gate: 0 lost; the 10 false fragments as negative unit cases, "guest host Max Read" positive.
2. **Rule 0 alone**: one `is_person_name(name, show)`, one `same_person(a, b)`; apply `same_person`
   in `stated_non_host_voices`. Unit table: reject `Two Carnegie Mellon`, `Timmerman Report`,
   `Premier Unbelievable?`, `Americas Online`, `Norman Conquest`, `Machine Learning Street`,
   `'Greg Brockman)'`; accept `Christian Schmidt`, `Christopher Guest`, `Donald S. Lopez Jr.`;
   same_person true for Alistair/Alastair Campbell, Karin Zissas/Carin Zissis, Rob/Robert
   Armstrong; false for Katie Martin/Jay Powell. Gate: 0 real names lost.
3. **Role locality**: speech acts scored per turn; a host act in the voice's first K turns outranks
   a guest act anywhere. Gate: the 16 gain their host (David Perell, Dwarkesh ×3, Gergely Orosz ×3…)
   with 0 real losses; How I Write bled cluster as a fixture.
4. **Rule 1 collect** only after 2 and 3. Gate: Alistair Campbell, Long Run guests, Rory Stewart,
   Eric Olander unchanged; Tyler ×38, a16z ×8, Dan Hooper gained. Episode-description hosts are a
   separate class that never demotes a feed-stated host.
5. Promo rule (first-person, fuzzy this-show). Gate: Odd Lots/Journal/DeepMind hosts unchanged;
   Judge, Benhold, Mumford dropped.
6. Vocative placement (duo-only, single addressee, open-anchored). Gate: a16z Anish fixed; No
   Priors/Planet Money unchanged.
7. Title/description phrase widening ("with A and B", "Join X", more verbs). Gate: 80-feed diff,
   12 feeds gain, 0 change.
8. Rule 3 (done separately: `3d7c95937`).

Do NOT bundle: Rule 1 with seat changes; any dominance change to step 3; Rule 0 with anything.

## 4. Conversations with Tyler (36 no host) — traced

42 of 44 newest runs predate `834cab0a7` (2026-10-01, `engage` added to `_PRESENTS`). On that code
the statement was ∅, both author tags org/show, no NER → empty pool; Tyler never says "I'm Tyler
Cowen". The 2 episodes run 2026-10-02 on `sha-0ee4580` have `known_hosts=['Tyler Cowen']` and
Tyler placed. **Code fixed; the 36 are stale runs.** Healing needs a roster re-run (out of bounds)
or a text-only placement derivation (operator's call).

## Not verified
Which false `_GUEST_HOST_EPISODE` matches blocked a name; the a16z 11-name pool source; which
variant produced `/tmp/rp/seat.jsonl`; the 1,314 pre-#2075 episodes; the LLM effect of a larger
closed list; Rule 0 + Rule 1 together; the 83-vs-4+1 denominators.
