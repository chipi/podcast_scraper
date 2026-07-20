# markdownlint rule audit

Rationale for every non-default rule in `.markdownlint.json`. `.markdownlint.json` is strict JSON and can't carry inline comments, so the reasoning lives here.

Last audit: **2026-07-19** — #1028.

## Disabled rules — legitimate project decisions

Each disabled rule has been verified against the tracked docs. Violation counts measured 2026-07-19.

### `MD013` — line length (13 034 violations)

Legit disabled. Long-form docs, wide tables, URL-heavy prose. Enabling would rewrite most of `docs/`. Prose docs are read on the mkdocs site where line length is irrelevant.

### `MD036` — emphasis used as heading (170 violations, deferred fix)

**Deferred, not endorsed.** Sample audit (2026-07-19) confirms most violations are semantic sub-headings that *should* be `### Foo` or `#### Foo`. Example from `docs/guides/eval-reports/EVAL_RFC097_CHUNK7_VLLM_WORKAROUNDS_2026_06_21.md` (17 violations, top offender):

```markdown
## Model 1: `google/gemma-4-26B-A4B-it`

**Failure signature**    <-- should be ### Failure signature

...

**Two fixes vLLM exposes**    <-- should be ### Two fixes vLLM exposes
```

Fixing all 170 mechanically is a `sed`-with-thought exercise: promote each `**Foo**\n` to `### Foo\n` or `#### Foo\n` depending on the enclosing heading depth. Doing that here would touch ~30 files across `docs/guides/`, `docs/architecture/`, `docs/api/`, `web/gi-kg-viewer/e2e/` with no per-doc rendering review. Deferred to a dedicated cleanup PR.

Top offenders when we tackle it:

- `docs/guides/eval-reports/EVAL_RFC097_CHUNK7_VLLM_WORKAROUNDS_2026_06_21.md` — 17
- `docs/api/CONFIGURATION.md` — 11
- `docs/architecture/VIEWER_ASYNC_STABILITY.md` — 8
- `docs/architecture/VIEWER_GRAPH_SPEC.md` — 5
- Six other files at 4 each.

### `MD051` — link fragments must exist (13 violations, all false positives)

Legit disabled. Every hit is a `[Foo](#foo)` link into a heading that uses mkdocs-material's `attr_list` custom-anchor syntax `## Foo {: #foo }`. markdownlint doesn't parse `attr_list`, so it doesn't know the anchor is real. mkdocs-strict build confirms every "broken" link resolves at render time. Re-enable if we drop mkdocs-material.

### `MD033` — inline HTML (19 violations)

**Legit disabled.** Two patterns account for every violation:

1. `<a id="anchor-name"></a>` — bare deep-link anchors at arbitrary positions in prose. CommonMark has no way to attach an id to a paragraph without a heading; the tag is portable across renderers. Example: `docs/api/CONFIGURATION.md:1057` `<a id="llm-cleaning-length-guard-issue-564"></a>`. Migrating to mkdocs-material's `attr_list` syntax would work but ties the anchors to that specific renderer.
2. `<br>`, `<code>`, `<sub>` inside table cells where CommonMark syntax can't do the job (line breaks in a cell, code with pipes, subscripts).

Both are structural rather than stylistic. Re-enable only if we standardise on mkdocs-material and can migrate pattern 1 to `attr_list` in the same pass.

## Re-enabled by this audit

### `MD025` — single top-level H1 per file (2 violations, both fixed)

**Re-enabled 2026-07-19.** The 2 violations were an accidental duplicate `# CI/CD Pipeline` H1 in `docs/ci/OVERVIEW.md` and `docs/ci/WORKFLOWS.md`. Each file's title was correct (`# CI/CD Overview`, `# CI/CD Workflows`); a stray `# CI/CD Pipeline` heading appeared before the first `## Overview` section. Both fixed inline this PR by removing the redundant H1.

## How to re-run the audit

For each disabled rule, count violations across the tracked docs:

```bash
CFG=$(mktemp -t md.XXXX.json)
cat > "$CFG" <<'EOF'
{"default":true,"MD013":false,"MD022":true,"MD024":true,"MD025":true,"MD029":true,"MD031":true,"MD032":true,"MD036":false,"MD040":true,"MD041":true,"MD047":true,"MD051":false,"MD033":false,"<RULE>":true}
EOF
env -u NODE_OPTIONS npx markdownlint-cli "**/*.md" \
  --ignore node_modules --ignore "**/node_modules/**" --ignore .venv --ignore "**/.venv/**" \
  --ignore .build/site --ignore "docs/wip/**" --ignore "tests/fixtures/**" --ignore "data/eval/runs/**" \
  --ignore "web/gi-kg-viewer/playwright-report/**" --ignore "web/gi-kg-viewer/test-results/**" \
  --config "$CFG" 2>&1 | grep -c "<RULE>/"
rm -f "$CFG"
```

Where `<RULE>` is `MD013`, `MD025`, `MD036`, `MD051`, or `MD033`.

## When to re-audit

- When we add a new rule to `.markdownlint.json` — verify the initial violation count matches expectations.
- When a violation count for a "legit disabled" rule changes materially — the disable might no longer be justified. E.g. if `MD051` violations drop to 0, we could re-enable and drop mkdocs-material's `attr_list` custom anchors.
- Yearly cadence otherwise — patterns shift; today's "50 violations" may be tomorrow's "5".

## History

- **2026-07-19** — Initial audit. `MD025` re-enabled after fixing 2 violations. Other 4 rules verified as legit disabled. See [#1028](https://github.com/chipi/podcast_scraper/issues/1028).
