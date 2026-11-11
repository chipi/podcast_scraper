# ADR-180: Bylines join people within their publisher; publishers are organisations across media

- **Status**: Proposed
- **Date**: 2026-11-11
- **Authors**: Marko Dragoljevic
- **Related**: [RFC-134](../rfc/RFC-134-news-as-the-second-content-type.md) (news as the second
  content type), [RFC-131](../rfc/RFC-131-person-identity-one-home.md) (person identity),
  [ADR-167](ADR-167-identity-guards-inside-the-join.md) (guards inside the join)

## Context & Problem Statement

The operator wants one graph across media (2026-11-11): a person discussed on a podcast and in the
news is one person. News brings people two ways:

- **people mentioned** in the text;
- **authors** named in the byline.

Hosts already have a rule of this shape. A person seated as host joins a name only within that
show, against the show's own verified names (`identity.guards.host_join_pair_rule`). Elsewhere a
host joins only through the general rule and its guards.

Bylines are noisy. In a sample of 19 sources on 2026-11-11 they included:

- several authors in one string;
- a surname first ("Arnold; Mark");
- a date stuck to a name ("Perri Thaler November");
- the publisher as the author ("Meduza", "TechCabal; Adonijah Ndege").

A wrong merge is worse than a cautious split (ADR-167).

## Decision

1. **People mentioned in articles join through the existing joining and its guards,** like any
   person in the corpus.
2. **A byline is a person with the role author, and joins within its publisher only,** the way a
   host joins within a show. Its publisher's bylines are its verified names.
3. **Across publishers, or into podcasts, a byline joins only through the general rule and its
   guards.** There is no byline-specific shortcut.
4. **An organisation in the byline is the publisher, not a person.** It is removed from the byline
   before joining.
5. **Publishers are organisations (T-12).** A publisher that also publishes a podcast in the
   corpus (the BBC, Meduza) is one organisation across both.

## Consequences

- **Positive:**
  - one graph across media, with no new kind of matching;
  - the host rule's evidence and guards apply to bylines unchanged.
- **Negative:**
  - byline clean-up (splitting, order, organisation removal) is new code with its own fixtures;
  - a journalist who appears on a podcast joins only when the general rule allows it, so some true
    joins are missed at first.
- **Neutral:** the joining itself moves with the private extension. Public provides the byline as
  data and its role.

## Alternatives considered

- **Join bylines like any name.** Rejected. Common names across many publishers would merge
  different people.
- **Keep bylines out of the graph.** Rejected by the operator. Authors are part of one graph.
- **Join bylines to podcast people on name alone when the publisher also has a podcast.**
  Rejected for now. To revisit with measured joins after launch.
