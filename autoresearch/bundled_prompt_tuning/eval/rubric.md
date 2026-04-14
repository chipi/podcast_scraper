# Judge rubric (immutable during a run)

Score how well the **candidate summary** reflects the **transcript**. Output must follow the JSON
contract in the user message.

## Dimensions (score each independently, then report a combined score)

1. **Coverage** — All main themes from the transcript appear; nothing central is missing.
2. **Accuracy** — No contradictions or invented facts vs. the transcript.
3. **Efficiency** — Each sentence and bullet contributes unique information not found elsewhere
   in the output. No padding, repetition, or filler. Length is appropriate to the content depth —
   a richer transcript warrants a longer summary; do not penalise length if the content justifies it.

A perfect summary hits all three. A failing summary hallucinates, omits the core story, or pads
with fluff.
