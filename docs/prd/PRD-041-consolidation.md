# PRD-041: Consolidation (personal knowledge corpus)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P3)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-036 (per-user store), PRD-040 (highlights + notes), PRD-039 (playback history)
- **Reuses**: RFC-049 (GIL), RFC-055 (KG), RFC-072 (canonical identity), RFC-090 (hybrid search)

---

## Summary

Consolidation is the differentiator — the reason the platform exists rather than another player. It turns
captures and listening history into a **personal knowledge corpus**: a per-user projection over the same
GIL/KG ontology the pipeline already produces. The corpus answers "what have I learned about X / from this
guest," draws cross-episode connections grounded in the exact moments the user marked, and **resurfaces**
past highlights for reflection on a spaced schedule. Over weeks, the user's brain-corpus compounds instead
of evaporating after each episode.

## Background & Context

- PRD-035 thesis: listening is the input; a growing, connected, grounded personal corpus is the output.
  Capture (PRD-040) is half the loop; Consolidation closes it.
- **It reuses what we built** (Principle 2): the personal corpus is a per-user *projection* over the
  existing ontology — highlights and saved insights become nodes in *the user's* graph, grounded in their
  marked moments. We add a per-user layer, not new ML. Cross-episode traversal reuses the relational layer
  (RFC-072); recall reuses hybrid search (RFC-090) scoped to the user's heard/captured set.
- Scope boundary: the corpus is built only from episodes the user has **heard or captured** — grounded
  recall must cite the user's own experience, not the whole shared corpus.

## Goals

- Build a per-user knowledge graph from highlights, saved insights, notes, and listening history.
- Answer grounded recall queries ("what have I learned about X", "what has <guest> said") by retrieval over
  the user's corpus (no request-time LLM, D6), returning grounded insights/quotes/highlights with
  jump-to-moment citations.
- Surface cross-episode connections (a guest across shows, a topic across episodes) within the user's corpus.
- Resurface past highlights on a spaced schedule with reflection prompts.
- Surface enrichment-powered signals (RFC-088, built in parallel): trending topics, contradictions, and a
  grounding-rate credibility cue; a "Your Week" digest (RFC-068/023); clustered topic threads (RFC-075).
- (North-star) expose the personal corpus via MCP (RFC-095) so the user's own agent can synthesise.
- Maintain a personal topic/interest profile that can personalise ordering in Catalog/Discovery.

## Non-Goals

- Not a generic corpus explorer — that's the operator viewer (RFC-062). This is *the user's own* corpus.
- Not collaborative knowledge (no shared corpora / social) in v2.7.
- Not automated study scheduling beyond simple spaced resurfacing (full SRS algorithm tuning is later).
- Not content translation.

## Personas

**Active learner** (retain and connect across months) and **researcher** (synthesise across what they've
consumed, with citations).

## User Stories

- _As a learner, I can ask "what have I learned about transformers" and get a grounded answer drawn only
  from episodes I've heard, each point linking back to the exact moment._
- _As a learner, I can see everything <guest> has said across the episodes I've heard, connected._
- _As a learner, I get periodic gentle resurfacing of past highlights to reflect on ("2 weeks ago you
  marked …")._
- _As a listener, my Catalog/Home gradually reflects my interests as my corpus grows._

## Functional Requirements

### FR1: Personal knowledge corpus

- **FR1.1**: A per-user graph projection composed of the user's highlights, saved insights, notes, and the
  entities/topics they touch — built over the canonical identity layer (RFC-072) so a person/topic is the
  same node across episodes.
- **FR1.2**: The corpus includes only episodes the user has heard (playback history) or captured from.
- **FR1.3**: Every corpus node retains grounding (episode slug + timestamp + quote) for citation/replay.

### FR2: Grounded recall (retrieval, no request-time LLM)

- **FR2.1**: "What have I learned about <topic/person>" runs hybrid retrieval (RFC-090) + relational
  traversal (RFC-072) over the user's corpus and returns the matching grounded items — insights, quotes, and
  the user's own highlights — grouped (by episode/guest) and ranked, each linking to its captured/heard
  moment (jump-to-player). The "answer" is the assembled grounded set, not generated prose (D6).
- **FR2.2**: Results are extractive and verbatim, so there is nothing to hallucinate — no disclaimer and no
  request-time LLM. (A generative summary layer on top is a parked future option.)
- **FR2.3**: Zero-coverage queries say so honestly ("nothing in your corpus on this yet") rather than
  drawing from the global corpus.

### FR3: Cross-episode connections

- **FR3.1**: Person view: a guest's positions/insights across the episodes the user has heard (reuses
  RFC-072 relational traversal, scoped to the user's set).
- **FR3.2**: Topic view: a topic's threads across the user's heard episodes.
- **FR3.3**: Connections surface as suggestions ("you also heard <guest> discuss this in …").

### FR4: Spaced resurfacing & reflection

- **FR4.1**: A periodic surface (digest or in-app) resurfaces past highlights on a spaced schedule
  (simple intervals in v2.7; tunable SRS later).
- **FR4.2**: Resurfaced items carry a reflection prompt and a one-tap jump-to-moment to re-listen.
- **FR4.3**: Resurfacing respects user pacing controls (frequency, pause, dismiss).

### FR5: Interest profile

- **FR5.1**: A personal topic/interest profile is derived from captures + listening history.
- **FR5.2**: The profile can personalise ordering in Catalog/Home (PRD-038) and surface related episodes —
  drawn from the shared corpus but ranked by personal relevance. (Personalised ordering is opt-in.)

## API summary

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/user/corpus/graph` | Personal knowledge graph projection |
| `POST` | `/api/user/corpus/recall` | Grounded recall query over the user's corpus |
| `GET` | `/api/user/corpus/person/{id}` | A person across the user's heard episodes |
| `GET` | `/api/user/corpus/topic/{id}` | A topic across the user's heard episodes |
| `GET` | `/api/user/resurfacing` | Due resurfacing items + reflection prompts |
| `GET` | `/api/user/interests` | Personal interest profile |

## Success Metrics

- Recall returns grounded results drawn only from the user's heard/captured episodes, with working
  jump-to-moment links; zero-coverage queries are honest.
- A guest's cross-episode thread is correctly assembled within the user's corpus (canonical identity holds).
- Resurfacing delivers due highlights on schedule and respects pacing controls.
- As a user's corpus grows, personalised Home ordering measurably reflects their captured interests
  (opt-in; off by default if it can't be validated).

## Dependencies

- PRD-040 (highlights/notes), PRD-039 (playback history), PRD-036 (per-user store).
- RFC-049/055/072/090 (the ontology + retrieval the projection reuses).

## Open Questions

- Generative layer (parked, D6): if/when to add an optional LLM summary on top of retrieved results — it
  would reintroduce a request-time LLM and the no-LLM-in-CI fixture/stub requirement.
- Spaced-resurfacing algorithm: fixed intervals for v2.7; when to invest in a real SRS model.
- Personalised ordering: how to evaluate "better" without a multi-user feedback loop yet.

## References

- `docs/prd/PRD-035-learning-platform.md`, `PRD-040-capture.md`, `PRD-039-player.md`
- `docs/rfc/RFC-049-grounded-insight-layer-core.md`, `RFC-055-knowledge-graph-layer-core.md`,
  `RFC-072-canonical-identity-layer-cross-layer-bridge.md`, `RFC-090-hybrid-corpus-search.md`
- `docs/guides/GIL_KG_CIL_CROSS_LAYER.md`
