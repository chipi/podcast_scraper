# Who should find the quote? — the grounding bake-off

**Date:** 2026-07-13
**Run:** `bakeoff_module2_v1` — `scripts/eval/experiment/grounding_bakeoff.py`

## Answer

**The summarising LLM should ground its own insights.** The local extractive-QA + NLI stack finds
evidence for **8%** of insights; qwen finds it for **82%**, and every quote it returns is verbatim.

| grounder | coverage | quotes/insight | verbatim | drift | fabricated | s/episode |
| --- | --- | --- | --- | --- | --- | --- |
| **ollama / qwen3.5:35b** | **82.0%** | 1.08 | **100%** | 0% | **0%** | 71 |
| transformers (QA + NLI) | 8.0% | 0.08 | 100% | 0% | 0% | 63 |
| gemini-2.5-flash-lite | **not measured** | — | — | — | — | — |

100 insights, 10 pinned episodes, one **frozen** insight set — every grounder is handed the
*identical* claims, so the only thing that varies is who does the retrieval. Without that, a model
that writes fewer or blander insights looks like a better grounder.

**gemini was not measured.** Its arm hit 70 × `503 UNAVAILABLE` ("this model is currently
experiencing high demand") and exhausted its retries. Any coverage figure from that run would have
measured Google's capacity, not gemini's grounding, so the arm was discarded rather than reported.
It needs a re-run when the API is healthy. It does not block the decision: the DGX profile grounds
with qwen, and the cloud profiles already ground with their own LLM.

## Why this had to be measured separately

The pipeline does two jobs, and they had never been separated:

1. **write the insight** — an LLM reads the transcript and states a claim
2. **find the quote** that backs it — a retrieval task over ~70 000 characters

They are different skills, and a model can be good at one and bad at the other. Every eval before
this one ran both together, so a grounding failure and a generation failure were indistinguishable.

## The fabrication question — the reason not to trust an LLM here

An extractive model **cannot** fabricate: it returns a span it selected out of the transcript. An
LLM *generates*, so it can emit a quote that is nearly-but-not-quite verbatim — and a near-miss
fails the character-offset match and is dropped **silently**, with no counter. That is the standing
argument for keeping evidence-finding away from an LLM.

It does not survive contact with the data. Three numbers, not one:

- **verbatim** — the quote is an exact substring of the transcript
- **drift** — not verbatim, but the tolerant matcher salvaged a sub-phrase (the stored quote is then
  *not* what the model returned)
- **fabricated** — not locatable at all

qwen scored **100% verbatim, 0% drift, 0% fabricated**, on every one of the 10 episodes. It copies
exactly. The risk is real in principle and absent in practice for this model.

## Why the local QA + NLI stack fails

Not a threshold. Two structural faults, and neither is fixed by tuning:

**There is no retrieval step.** `deepset/roberta-base-squad2` answers a question *within* a window.
With a 1800-char window over a 70k transcript there are ~40 windows, every one of them is forced to
answer, and nothing ever asks *which window is about this claim*. QA confidence is not relevance, so
the winner is whichever irrelevant window happened to be most confident.

**The verifier asks the wrong question.** The NLI model works — it scores textbook pairs 0.98 / 0.99
/ 0.00 — but it demands strict logical entailment, and insights are abstractive:

```text
premise    "this was POSSIBLY creating SOME tension between Sam Altman and his CFO"
hypothesis "OpenAI's CFO IS experiencing tension with Sam Altman"        -> 0.007
```

The transcript hedges; the insight asserts. Strict entailment says no, and it is technically right.
But the pipeline means *"is this quote evidence for this claim?"*, not *"does this premise entail
this hypothesis?"* — which is exactly the lesson #1179 recorded for the LLM entailer and which was
never applied to the NLI model, because an NLI head cannot ask a graded question at all.

An embedding prototype retrieves the correct passage (it pulled the literal supporting sentence for
a claim the QA model missed entirely), so the retrieval half is salvageable. Fixing the ML grounder
means adding embedding retrieval **and** replacing the verifier. Tracked separately; it only affects
the local/offline profiles.

## Two bugs fixed along the way

Both were invisible because no eval had ever exercised this code path — the eval harness pointed
grounding at the LLM under test, so the stack production actually ran was never executed once.

1. **`top_k` was discarded whenever windowing was active** — and windowing is always active. The
   grounder got one candidate per insight no matter what it asked for.
2. **Span scores were softmaxed *within* each window's own top-k**, so the best span in *any* window
   came back at ~1.0. A window about nothing scored as high as the window holding the evidence, the
   cross-window winner was arbitrary, and `gi_qa_score_min` gated on nothing. (This is why a 70k
   transcript returned the single word `"Codex"` at `qa_score = 1.000` for every insight.)

Scores are now absolute probabilities, comparable across windows. It is not enough to make the ML
grounder usable — see above — but the numbers it reports are now real.

## Caveats

- One show (Hard Fork), 10 episodes, 100 insights. Enough to separate 82% from 8%; not enough to
  rank two close grounders.
- The frozen insight set is gemini's. A qwen-written insight set may ground differently — the claims
  are phrased differently. Worth a second pass before the corpus build.
- `gi_qa_score_min = 0.3` is now stale (it was tuned against the broken ~1.0 scores). It must be
  recalibrated *with* any ML-grounding work, not before — a retrieval stage will change the
  distribution again.
