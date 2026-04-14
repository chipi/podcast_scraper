# AutoResearch — OpenAI Bundled Prompt Tuning (Paragraph Track)

## Objective

Improve the paragraph/summary quality of bundled OpenAI output, measured against
`silver_sonnet46_smoke_v1`. Shares prompt templates with the bullets ratchet — experiments
must be checked against both ratchets before acceptance.

- Templates:
  - `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2`
  - `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2`

- Ratchet configs:
  - Paragraph (primary): `data/eval/configs/summarization/autoresearch_prompt_openai_bundled_smoke_paragraph_v1.yaml`
  - Bullets (regression check): `data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_smoke_bullets_v1.yaml`

- Results log: `autoresearch/bundled_prompt_tuning/results/results_openai_paragraph_r1.tsv`

Current paragraph baseline: **0.455919** (ROUGE-L 26.2%, judge_mean 0.909, 0/5 contested).
Current bullets champion: **0.524266** (must not regress > 1%).

## Dual-metric acceptance rule

Accept an experiment only if **both** hold:

- Paragraph ratchet delta ≥ +1%.
- Bullets ratchet delta ≥ −1% (no significant regression).

## Round 1 Hypotheses

Silver paragraph style (observed): 5–6 paragraphs, 335–410 words. P1 = thesis sentence naming
episode domain + central argument. P2..Pn each cover one topic, often opening with a topic
anchor ("On [topic]...", "[Topic] is framed as..."). Final paragraph = practical takeaway.
Current bundled output already uses 4–6 paragraphs but lacks structural guidance.

1. **Structure narration for summary** — add prose guidance in system prompt: "Each paragraph
   covers one distinct topic; P1 states the thesis; subsequent paragraphs cover topics in the
   order they appear; final paragraph states the practical takeaway."

2. **Topic anchor openers** — add user prompt rule: "Open each non-thesis paragraph with a
   topic-anchor phrase ('On [topic]...', '[Topic] is framed as...', 'A recurring theme...')."

3. **Opening sentence pattern** — add instruction: "Begin the summary with a sentence naming
   the episode's domain and its central argument or premise."

4. **Silver-quality paragraph example** — add a complete silver-style 4-paragraph example to
   the system prompt (same show-don't-tell approach that worked for bullets in r3-1 / r7-1).

5. **Thesis + takeaway dual rule** — require both: first sentence names domain+thesis; last
   paragraph states practical takeaway. Tests whether the two anchors together close the gap.

## Experiment Loop

For each experiment:

1. Edit the bundled templates.
2. Run paragraph ratchet:
   `make autoresearch-score-bundled CONFIG=data/eval/configs/summarization/autoresearch_prompt_openai_bundled_smoke_paragraph_v1.yaml REFERENCE=silver_sonnet46_smoke_v1`
3. Run bullets ratchet (regression check):
   `make autoresearch-score-bundled`
4. Decision:
   - Paragraph ≥ +1% AND bullets ≥ −1%: **accept**, commit both templates + results TSV.
   - Else: **reject**, `git checkout HEAD -- <templates>`.
5. Append one row to `results_openai_paragraph_r1.tsv` with both paragraph and bullets scores
   in the notes.
6. Stop on 3 consecutive rejections.
