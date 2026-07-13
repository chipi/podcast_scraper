# What should an insight be? — routing, not ranking-and-truncating

The pipeline was optimising the wrong thing. We spent a day driving "grounded insights per episode"
from 16 to 28 and called it a 72% improvement. It was 72% more *rows*. Nobody reads 50 insights, and
a corpus of 5,000 of them is not knowledge — it is a spreadsheet.

This records what we actually want to surface, and why the machinery has to change shape to do it.

## The test

**What can we surface that a headline, a summary, or a transcript search cannot?**

Our differentiators are that we are **grounded** (a verbatim quote, timestamped), we are a **corpus**
(many shows, over time), and we know **who spoke**. An insight is only worth a slot if it exploits at
least one of those.

By that test, most of what we ship today fails. Sampled from real output:

```text
"OpenAI signed a $50 billion deal with Amazon."
"The 1873 railway crisis is considered the first global financial crisis."
```

A headline gave you the first. Wikipedia gave you the second. **We are extracting news, and the
summary layer already tells you what happened.** If a claim would appear in the summary, it is not an
insight.

## The taxonomy

Every extracted claim gets exactly one label, and the label decides where it goes.

### SURFACE — what a person said that you could not get elsewhere

| type | test | real example from our output |
| --- | --- | --- |
| **STANCE** | a position taken; you could disagree with it. A prediction is a stance about the future. | *"Autonomous AI systems writing prescriptions are currently unsafe despite pilot programs."* |
| **ARGUMENT** | a claim with a *because* — a mechanism, not just an assertion | *"The 1930 cutoff was chosen because works from that era are largely public domain."* |
| **EXPERIENCE** | firsthand, primary-source. Unavailable second-hand. | *"When we ran that team, what we found was…"* |

### CONNECT — true, often important, and the wrong shape for a first look

**EVENT** — a deal, launch, departure, ruling, or number. These are **not** waste: they are the
corpus's connective tissue. Five near-identical Pentagon/Anthropic claims across three episodes are
dilution as insights and a **story thread** as corpus structure — an arc you can follow across shows
and over time, which no single episode summary can give you. They feed the KG and the threads; they
never compete for a UI slot.

### DROP

**TRIVIA** (unique and inconsequential), **GENERIC** ("AI is advancing rapidly"), **FILLER** (ads,
chit-chat, topic labels).

## Two things the data said that we did not expect

**1. There is almost no junk to clean.** On raw, ungated output: 0% filler, 0% generic, 1.6% trivia.
The DROP bucket was supposed to be the cleanup lever and it is nearly empty — the extraction prompt
already carries a substance bar and the model is not emitting slop. **The dilution is not junk. It is
37% EVENTs** — real news, correctly true, wrong shape. Routing fixes that; cleaning cannot, because
there is nothing to clean.

**2. A stance without a speaker is unfalsifiable.** Asked to label claims with no speaker attached, a
judge returned **42% STANCE** — it became the new catch-all, exactly as `insight_type: observation`
had at 79%. A stance is a position someone *takes*; with no owner the category has no anchor. That is
not a prompt bug, it is a **missing field**. It is why speaker attribution had to be fixed first.

## No fixed cutoff

The obvious move is "rank and keep the top 12". **Do not.**

A cutoff baked into the pipeline is baked into the **corpus**: once a 100-episode build has cut at 12,
the 13th is gone and cannot be recovered without reprocessing everything. Truncation is cheap and
reversible in the **UI** and expensive and permanent in the **pipeline**.

So: **the pipeline ranks and tags. It never truncates.** "First look" becomes a UI decision we can
retune for free.

That only works if the ranking is honest, which means doing the reductions that are **not arbitrary**
and letting the count fall out of the content:

1. route EVENTs to CONNECT
2. drop within-episode near-duplicates — the same claim pulled from two chunks is a **defect**, not a
   choice
3. drop the ungrounded — no evidence, no speaker, nothing to show

If a 25-minute news roundup yields 8 and a 90-minute interview yields 19, then it **should** be 8 and
19. A fixed cap hides exactly that signal.

## The ranking signals

Applied to the SURFACE pool. None of them is a cutoff; they order the list.

| signal | cost | what it does |
| --- | --- | --- |
| **corpus novelty** | free (embeddings) | kills the near-duplicate — the same story told five ways. **Not** a value signal: the most "novel" claims are trivia and one-off headlines, so it ranks *redundancy*, not worth. |
| **speaker role** | free (deterministic) | the guest expert's stance outranks the host's framing. The guest is who the episode exists for. |
| **type** | from the classifier | an ARGUMENT (has a *because*) carries more than a bare STANCE |
| **grounded** | already computed | an unevidenced claim is not shown |

## Who classifies, in the real pipeline

The **value gate's judge** — a second provider call to a model pinned once in the registry
(`value_gate_provider` / `value_gate_model`), independent of whichever LLM writes the insights. That
component already exists; it does not need replacing, it needs a **better rubric**.

The current rubric is the bug. Its top tier reads: *"3 = CORE. A substantive claim about the world: a
decision, deal, launch, departure or result; a concrete number or finding."* **That is a definition of
news.** It scores the headline top marks, and the headline is the least novel thing in the episode.
The gate is not failing to select — it is selecting for the wrong thing.

**Not the extractor.** It is cheaper (the extractor already emits a type, so it costs nothing extra),
but the label decides what gets *shown*, and a model labelling its own output has every incentive to
call everything a STANCE. That is the #939 self-grading bias, applied to routing instead of scoring.

## Stage order, and one cost decision

```text
1. EXTRACT     chunked                              -> ~50 candidates
2. ATTRIBUTE   speaker, from the quote's offset     -> free, deterministic   [shipped]
3. CLASSIFY    the pinned judge, one batched call   -> SURFACE / CONNECT / DROP
4. NOVELTY     embeddings vs the corpus             -> free
5. RANK        novelty x speaker-role x type        -> no truncation
6. GROUND      verbatim quotes                      -> the expensive stage
```

Today we ground **all ~50** and gate afterwards. Grounding is the costliest stage after extraction
(~70s/episode). Grounding only what we would surface is ~4x cheaper — but grounding is itself a
quality filter, so selecting before grounding risks surfacing something unevidenced.

Suggested compromise: ground the top ~20, keep what grounds. Cost saving and grounding still has the
last word.

## Open

- **The classifier's rubric still is not crisp.** Even an independent judge returned 44.6% STANCE, and
  some of it is bleed ("OpenAI faces significant financial risks…" is an assessment of an EVENT). The
  next iteration gives the judge the **speaker and their role**, which is the anchor it was missing.
- Which local model is the judge — the bake-off decides it.
- Guest-vs-host weighting needs corpus-level speaker canonicalisation (ASR renders the same person as
  "Kevin Roos" and "Kevin Russo"; phonetic + edit-distance matching merges them).
