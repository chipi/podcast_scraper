# Design RFC Prompt Template

You are helping design a change in a production Python project.

## Constraints

- Maintainability > cleverness
- Prefer minimal dependencies
- CI and Docker must remain stable
- Backward compatibility matters

## Task

1. Restate the problem in your own words
2. Define explicit goals and non-goals
3. Propose 2–3 viable designs
   - Explain tradeoffs (complexity, risk, testability)
4. Recommend one design and justify it
5. Outline a concrete implementation plan
6. Identify risks and how to mitigate them
7. Propose a test strategy

## Rules

- Do NOT write code yet
- Focus on decisions and reasoning

**Use when:** starting features, refactors, or PRDs

**Effect:** Pushes Auto to Opus / GPT-5.2-level reasoning
