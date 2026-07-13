# The insight pipeline is two modules, and they must be measured separately

The grounded-insight pipeline does two jobs. We have been treating them as one, and every number we
produced mixed them together.

| | job | input | output | who does it today |
| --- | --- | --- | --- | --- |
| **Module 1** | **write the insights** | transcript | a list of claims | the summarising LLM (gemini / qwen) |
| **Module 2** | **find the quote that backs each insight** | transcript + one insight | quotes, with character offsets | a local extractive-QA model + an NLI model |

These are different skills. Writing a good claim is a reasoning task. Finding the exact sentence
that proves it, verbatim, in a 90 000-character transcript, is a retrieval task. A model can be good
at one and bad at the other — and qwen is exactly that model.

## Why this went wrong

The eval harness **re-implemented** the pipeline's wiring instead of using it. Its copy pointed
module 2 at the summarising LLM, so every eval had qwen (or gemini) hunting for its own quotes.
Production never does that: it uses the local QA/NLI models for module 2, whichever LLM wrote the
insights.

So the eval measured one thing and the pipeline runs another, and the numbers could not match. Every
grounding figure we have — grounding rate, quotes per insight, and the headline "grounded insights
per episode" — describes a module 2 production does not use.

**The fix (landed):** the eval no longer derives the evidence stack. It inherits whatever the config
resolved, exactly like production, and an experiment can override it *only by naming a provider on
purpose*. A test pins eval-equals-production so it cannot drift again.

## How each module is measured

Separately, because they fail in different ways.

### Module 1 — writing insights

Measured **before any grounding**, so a weak retriever cannot mask a strong writer.

- insights per episode, and per 10 000 characters (a raw count is meaningless without episode length)
- value distribution: how many are CORE / USEFUL vs FILLER, per an independent judge
- does the count scale with episode length, or does the model saturate?

This is where the real gemini-vs-qwen difference lives: **qwen saturates at roughly 18 insights per
call regardless of how much material it is given; gemini scales with the episode.** That finding is
unaffected by the eval bug, because it never involved module 2.

### Module 2 — finding the evidence

Measured on a **fixed insight set**, so the writer is held constant and only the retriever varies.
Otherwise a model that writes fewer insights looks like it grounds better.

Three numbers, and all three are needed:

- **coverage** — what share of insights get at least one quote. Alone, this is gameable: a lenient
  grounder scores 100% by attaching bad quotes.
- **fabrication** — does the returned quote actually appear in the transcript? This is checkable
  deterministically (string match), and it is the one an LLM can fail and a retrieval model cannot.
- **support** — does the quote actually back the claim, per an independent judge (not the model that
  produced it — self-grading is ~6x more lenient).

## The open decision: should an LLM do module 2?

Today module 2 is the local QA + NLI models. The alternative is an LLM. Both are real options and we
have **not** measured them against each other on a fixed insight set.

The case for the local models: retrieval is what they are for; they cannot fabricate a quote,
because they select a span out of the transcript rather than generating text; they are cheap and run
on the box.

The case for an LLM: it may find evidence for claims that are supported by a passage but not by any
single contiguous sentence — the QA model can only return a span.

The known risk on the LLM side is fabrication: an LLM asked for a verbatim quote can produce one that
is *nearly* right, and a near-miss fails the offset match and is dropped silently. That is a real
failure mode we already have — quotes the retriever returns but that cannot be located are discarded
today with no counter at all.

**This is now an answerable question**, because module 2 is a named choice rather than an accident.
The experiment is: one fixed insight set, four grounders (local QA/NLI, qwen, gemini, anthropic),
scored on coverage / fabrication / support.

## What this means for prod v3 on the DGX

The question "can qwen replace gemini" splits in two, and the answers are probably different:

- **Module 1 (write insights):** this is the real contest, and it is where qwen's saturation
  matters. Chunking is the lever.
- **Module 2 (find quotes):** if the local QA/NLI models do this, then **it does not matter which
  LLM wrote the insights** — the retriever is the same either way. qwen's apparent "grounding
  problem" would then be an artifact of the eval, not a property of the DGX stack.

Which is why module 2's owner has to be decided before the 100-episode run, not after.
