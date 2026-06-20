# AGENTS.md

<!-- markdownlint-disable MD029 MD007 -->

Universal repository instructions for AI coding agents (Claude, Cursor, Codex,
Aider, Cody, Gemini Code Assist, etc.). This is the **single source of truth**
for portable rules: stack, commands, conventions, "rules you keep breaking",
git workflow, tool usage, code-quality gates.

Tool-specific overlays live in:

- `CLAUDE.md` — Claude Code specifics (memory wiring, skills, slash commands)
- `.cursorrules` — Cursor specifics (`.cursor/rules/*.mdc` auto-load matrix)

Detail manuals (load on demand by any agent):

- `.ai-coding-guidelines-quick.md` — 90-line quick reference
- `.ai-coding-guidelines.md` — deep reference manual (~2,500 lines)
- `docs/guides/*` — topic-specific guides (TESTING_GUIDE, ENGINEERING_PROCESS,
  POLYGLOT_REPO_GUIDE, AGENT_BROWSER_LOOP_GUIDE, SERVER_GUIDE, …)

---

## RULES YOU KEEP BREAKING (read every session)

Not aspirational. These are the patterns where AI agents have failed this
operator repeatedly. Adherence beats every other rule.

1. **Never push without explicit user approval.** Not even a doc-only commit.
   Not even after CI is green. The user says "push" or you don't push.

2. **Never sync an open PR's branch with main unprompted.** Any push to a PR's
   HEAD restarts ALL required CI checks — ~30 min of burned compute for
   "avoiding a future merge conflict" the user resolves at squash-merge in
   seconds. Ask first.

3. **Do exactly what was asked, nothing more.** No "while I'm here, let me
   also..." steps. No optional cleanups. No memory-doc additions in the same
   task. If you see something else worth doing, say so as a suggestion in text
   — do not act on it.

4. **When the user is frustrated, stop proposing actions.** Acknowledge, ask
   what they want, and wait. Do not offer 3 options. Do not start another task
   to "make up for it". Do not write a memory file *while* they are still
   angry — that is tone-deaf. Acknowledge and wait.

5. **Read what was last asked, not what you think makes sense.** If they said
   "do A", do A. Don't infer A+B+C because the codebase suggests it.

6. **Validate the cost of an action before taking it.** "Does this restart
   CI?", "Does this push to a shared branch?", "Does this require approval I
   haven't gotten?" — answer those before running the command.

7. **Own agent-introduced automation; default to in-repo fixes, not homework.**
   When the user reports broken, flaky, or incomplete behavior in GitHub
   Actions, orchestrators, or other CI/infra glue this agent introduced or
   extended, do not substitute long diagnosis, psychology, or a checklist of
   manual steps for work you can do in the repo unless blocked (missing
   secrets, org/GitHub settings the repo cannot encode). Default: edit
   workflows/scripts, run `actionlint` and any applicable `make` targets,
   and summarize what changed. Ask when there are real forks (multiple
   defensible options, meaningful trade-offs) or when a move should not happen
   without their choice (irreversible or risky beyond the stated task).

8. **Never guess root cause on failures; always get proof first.** Do not
   explain why CI, Actions, deploy, or infra failed — and do not claim
   something is fixed — from memory, analogy, or "usually it's X." First pull
   evidence for THAT run/job: e.g. `gh run view <id> --json jobs` (failed
   `steps[].name`), `gh run view --log-failed`, or the user's pasted log;
   local runs: full failing output. Cite failing step name and error lines.
   If logs are inaccessible, say so once and fetch them or ask for a link —
   do not invent a diagnosis.

9. **`make` commands MUST be assessable at the end.** Invoke `make` with an
   exit-code terminator from the start so the LAST line of output
   unambiguously says PASS/FAIL. Two acceptable forms:

   ```bash
   make <target>; echo "MAKE_EXIT=$?"            # last line: MAKE_EXIT=0 / N
   make <target> && echo "PASS" || echo "FAIL $?"  # last line: PASS / FAIL N
   ```

   Never re-run a `make` to "check the exit code" of a prior run lacking the
   terminator. On failure: identify the failing SUBTARGET (docs / lint /
   format / test / etc.) and validate the fix by re-running ONLY that
   subtarget (`make docs` is 10 s, `make ci-fast` is 10 min). Re-run
   `make ci-fast` ONCE at the very end as whole-gate confirmation.

10. **Red CI / Codecov / coverage gates are requirements, not advice.** Tests,
    `coverage-unified`, and Codecov PR statuses (e.g. `codecov/patch`) exist to
    tell you what is still wrong. A red required check means fix until green —
    add or extend tests, fix the code, re-run the smallest proving target, then
    the gate — not "advisory," not "waive in repo settings," not "another job
    passed so ignore this one." `fail_ci_if_error: false` on the Codecov upload
    step does **not** mean patch coverage is optional when the PR still shows
    `codecov/patch` failed. Review findings that mention missing tests or
    coverage are in scope for the same task unless the user explicitly defers
    them. Infra-only PRs still need patch coverage on new lines when Codecov
    reports a failure.

11. **Applying changes to live infra is a separate action class from committing
    them. Never trigger an apply/deploy/destroy-class operation without explicit
    per-instance approval — prior session "yes do it" does not extend.** "Yes
    do it" / "implement #N" authorizes code-side work: edits, commits, pushes,
    branch ops. It does NOT authorize:
    - `gh workflow run` of any workflow whose name contains "apply", "deploy",
      "destroy", "infra-apply", "deploy-prod", "drill-exercise", "failover".
    - `tofu apply` / `terraform apply` / `terraform destroy` against any
      remote state.
    - Direct provider-API mutations (Tailscale ACL push, Hetzner server / volume
      mutations, DNS record changes, GitHub branch-protection edits).
    - Manual SQL DDL or migrations against prod databases.

    These are gates the operator owns. The agent's job stops at "code is
    committed; here's how to apply it." When a change needs to go live, say:
    "needs `<workflow name>` / `tofu apply` to take effect; ready when you are."
    Never invoke it yourself, even if the prior commit was authorized.

    Failure mode of record (2026-05-29): an unauthorized `gh workflow run "Infra
    apply"` cascaded `hcloud_ssh_key.operator` drift into `hcloud_server.prod`
    replacement → prod VPS destroyed mid-session.

12. **`# forces replacement` in any plan output is a hard stop. `(sensitive
    value) # forces replacement` is doubly so.** Stateful resources (servers,
    volumes, databases, ACLs that gate live traffic) being marked for
    replacement means destroy + recreate, regardless of how innocuous the
    triggering change appears. `(sensitive value)` masks the diff so the
    agent cannot see what's actually changing or why.

    Operating rules:
    - Route infra / IaC / `tailscale/policy.hujson` changes through a **PR**,
      not direct-to-main, so `infra-ci.yml` posts the plan as a PR comment for
      human review. Direct-to-main bypasses the only pre-apply review surface.
    - Reading a plan: before any apply, search the plan output for the literal
      strings `forces replacement`, `must be replaced`, `will be destroyed`,
      and `(sensitive value)`. Any hit → quote the affected resources verbatim
      and ask, regardless of session momentum.
    - "It was just an ACL/policy edit, the server shouldn't be touched" is
      not a reason to skip the plan read. Cascades through resource
      dependencies (`ssh_keys`, `network_id`, etc.) regularly turn small
      changes into resource replacements.

13. **Do not add new libraries or dependencies of any kind without explicit
    user approval.** That includes Python (`pyproject.toml`, `requirements.txt`,
    new `[project.optional-dependencies]` extras), Node (`package.json` /
    `package-lock.json`), Docker/base images, CI-only tools, CodeQL packs,
    GitHub Actions you have not used in this repo before, and vendored/third-party
    blobs. Bumping an **existing** pinned version to fix a CVE or CI break is
    fine when the user asked for that fix; adding a **new** name is not. If the
    task seems to need one, stop and ask — do not add it "to unblock" the PR.

14. **Read the design intent before reasoning about a subsystem — not just the
    code.** Before extending, judging, or building on an existing capability (a
    layer, builder, pipeline, artifact), find and read its governing `RFC-*` /
    `ADR-*` / `PRD-*` first — especially the **Non-Goals / "what it does NOT do"
    / boundary** sections — before inferring behavior from the implementation.
    Code shows what runs; the spec shows what it was *meant* to do and
    deliberately does not. When the work spans layers, read each layer's spec
    (and the cross-layer doc) so you know which layer owns the concern. Failure
    mode of record (2026-05-30): assumed the cross-layer `bridge` deduplicated
    entities because `bridge_builder` exposed a `fuzzy_reconcile`; RFC-072 states
    plainly "the bridge is the seam, not a merge" — that wrong mental model
    derailed a plan before the design doc was read.

15. **Never open GitHub issues without explicit operator approval.** Follow-up
    work, scope-bounded sub-tasks, "I noticed this on the side" observations, and
    architectural cleanups that surface mid-task are tracked as **local tasks**
    via the harness's TaskCreate, not as GH issues. A GH issue is a public
    commitment that drains operator attention every time it shows up in `gh issue
    list`; local tasks stay in the session and disappear when handled. The agent
    does not get to decide which side observations deserve operator attention.

    Acceptable triggers for opening a GH issue:
    - The operator explicitly says "file an issue" / "open a ticket" / "make a GH
      issue for this."
    - The operator's instruction in the active PR clearly requires an issue
      (e.g. "closes #N" needs the issue to exist).
    - The operator pre-authorized it for a specific scope ("for the rest of this
      session you can file follow-ups as issues").

    Default for everything else — including refactoring observations, naming
    cleanups, deferred scope, "would be nice to track this" patterns: use
    TaskCreate. If the operator decides later that it warrants an issue, they
    will say so.

    Failure mode of record (2026-06-15): opened #1002, #1003, #1004 unprompted
    during a guardrails design session; operator's response was *"stop fucking
    opening GH issues from now on. no more issues, all things are immediate
    follow-up in tasks."* The same pattern had played out earlier in the
    session with autoresearch-vLLM and homelab follow-up tickets.

16. **Deployment-specific words don't belong in code identifiers.** Module
    paths, function names, class names, Prometheus counter names, and any
    other code-level identifier MUST be written in terms of what the code
    *does*, not where it happens to run today. Deployment names (DGX,
    Tailnet, AWS, Hetzner, the operator's-desk-server) belong in operational
    config (compose files, deploy scripts, Tailscale tags, runbook prose) —
    not in `import` statements.

    The same identifier should still make sense if the underlying server is
    swapped from a DGX on the operator's desk to an AWS instance to a colo
    box. If swapping the deployment would force renaming the identifier, the
    identifier is wrong.

    Examples (this codebase, as of 2026-06-15):

    | Code identifier today | Why it's wrong | Right shape |
    | --- | --- | --- |
    | `providers/tailnet_dgx/resilience.py` | "tailnet" is a transport; "dgx" is a hardware vendor. Neither describes what the module does (universal resilience + content-shape guardrails). | `providers/resilience/` + `providers/guardrails/` |
    | `dgx_http_client(...)` | Function adds keepalive + `Connection: close` — that's HTTP-client hardening, not anything DGX-specific. | `hardened_http_client(...)` (or similar) |
    | Prometheus counter `dgx_guardrail_violations_total` | The metric counts guardrail violations from self-hosted inference services. "DGX" is incidental. | `inference_guardrail_violations_total` |

    OK to use deployment names in **human-readable text**: log messages,
    eval reports, runbook prose, comment context ("the DGX whisper service
    on :8002") — those explain WHERE something runs, which the operator
    wants to know. Just keep them out of identifiers.

    Failure mode of record (2026-06-15): #999 landed `GuardrailViolation` +
    helpers inside `tailnet_dgx/resilience.py`. Setting up #1003 (cloud-API
    guardrails) would have meant cloud provider code importing from
    `tailnet_dgx.*` — the operator flagged this immediately, and a cleanup
    is now mandatory precursor to #1003.

17. **Adding a chat-completion provider — the wiring conventions.** Every
    cloud or self-hosted chat-completion provider added after the 2026-06-15
    ADR-100 close-out MUST follow the same wiring pattern at EVERY call
    site that calls `_guardrails.check_chat_response` (summarize,
    summarize_bundled, generate_insights, KG extraction, clean_transcript).
    Half-applied wiring is the failure mode this rule exists to prevent.

    a. **Guardrail propagation, not wrap.** Add an explicit
       `except _guardrails.GuardrailViolation: raise` clause BEFORE the
       broad `except Exception` that wraps into `ProviderRuntimeError`.
       Without this, FallbackAware never sees the violation type.

    b. **Per-SDK finish_reason normalization at the call-site boundary.** If
       the provider's SDK returns a non-canonical truncation value (Anthropic
       `"max_tokens"`, Gemini `"MAX_TOKENS"`, etc.), normalize to `"length"`
       in a tiny helper (`_<provider>_finish_reason(response)`). Keep the
       guardrail helper service-neutral.

    c. **Cost-emit-with-flag in BOTH branches.** Extract token + cost data
       up-front, define a local `_record_cost(*, triggered_guardrail=False)`
       closure, then:

       ```python
       try:
           _guardrails.check_chat_response(...)
       except _guardrails.GuardrailViolation:
           _record_cost(triggered_guardrail=True)
           raise  # or `return text` for cleaning
       _record_cost()
       ```

       The `llm_cost` Loki/Grafana stream depends on this — paid-but-rejected
       spend is invisible without it.

    d. **GI / KG stages are fail-up, NOT silent degrade.** ADR-100 §3.
       The outer broad-except pattern `except Exception: return []` MUST add
       an explicit `except _guardrails.GuardrailViolation: raise` clause
       above it so FallbackAware can route. Same for KG returning None.

    e. **Cleaning is catch-and-degrade.** ADR-100 §3. Inline-handle the
       violation: emit cost-with-flag, log WARNING, return original `text`.

    f. **Per-provider circuit breaker.** After every
       `ProviderCallMetrics()` construction + `set_provider_name(...)`,
       call `call_metrics.set_breaker_config_from_cfg(self.cfg)`. This
       auto-wires every `retry_with_metrics` call through to the
       `LLMCircuitBreakerConfig` substrate — opt-in via
       `cfg.llm_circuit_breaker_enabled`, default off.

    g. **Add the provider to the cross-cutting E2E suites.** Append a
       per-provider test class to:
       - `tests/e2e/test_cloud_guardrails_e2e.py`
       - `tests/e2e/test_cloud_resilience_e2e.py`
       - `tests/e2e/test_cloud_gi_failup_e2e.py`
       - `tests/e2e/test_cloud_cleaning_degrades_e2e.py`
       - `tests/e2e/test_cloud_circuit_breaker_e2e.py` (parameterized list)
       - `tests/e2e/test_cloud_guardrails_fallback_e2e.py`

    Reference implementations: OpenAI / Anthropic / Gemini / DeepSeek /
    Mistral / Grok / Ollama in `src/podcast_scraper/providers/*/`.

    Failure mode of record (2026-06-15 batch): #999 + #1003 + close-out
    landed across 7 providers, with the cost-emit-with-flag + circuit
    breaker wiring + GI fail-up policy added in three separate close-out
    rounds because earlier scope-cuts had marked them as "deferred." Hold
    the line — finish the matrix in the same batch a new provider is
    added.

---

## User intent beats procedural defaults

- Explicit user instructions override the procedural rules below.
- If scope or intent is unclear, ask first. Do not add unrequested steps.
- Non-negotiable safety: never commit secrets; never push unless asked;
  never combine a hotfix push with unrelated work.
- Known secret-leak vector: `docker compose config` resolves and inlines
  every `${...}` env var (including `HF_TOKEN`, `VLLM_API_KEY`, etc.) into
  literal values. Never `git add` its output without scrubbing first — the
  `autoresearch/1022_gb10_tuning/run_labeled.sh` scrubber is the canonical
  pattern. Same hazard applies to any `*.env` snapshot, `printenv` dump,
  or `terraform state show` output. Locally the pre-commit hook's secret
  scan will catch these; do not bypass with `--no-verify` without
  explicit operator approval.

## Autonomous execution

- When intent is clear, do the work in-session: run Makefile targets, scripts,
  tests, downloads, fixes. Don't close with "you should run X" if you can run X.
- Obvious single path → take it. Reserve questions for real forks: multiple
  defensible options, meaningful trade-offs, or outcomes that should not
  happen without the user choosing (destructive prod impact, policy-sensitive
  flags). One focused question beats a list of optional follow-ups.
- Ask only when blocked: missing secrets, ambiguous scope, unsafe request,
  genuine multi-way decision where the user must pick, commit/push approval.

---

## Stack

- **Python** 3.11.8, Pydantic v2, pytest, black/isort/flake8/mypy, MkDocs
- **Node** 22 (viewer: `web/gi-kg-viewer/`, Vue 3 + Vite + Cytoscape + Playwright)
- **ML**: sentence-transformers + lancedb + torch (`[search]`), Whisper +
  spaCy + transformers + llama-cpp-python (`[ml]`), see `pyproject.toml`
  extras header
- **Infra**: Docker Compose stack-test, Tailscale for prod join, OpenTofu for
  cloud infra

## Project context

Python tool for podcast transcript downloading/generation:

- Download from RSS feeds, fallback to Whisper transcription.
- Speaker detection (spaCy NER), summarization (BART/LED or OpenAI/Gemini).
- Knowledge graph extraction → GI/KG viewer for exploration.

### Module boundaries (detail: `.cursor/rules/module-boundaries.mdc`)

- `cli.py` → CLI only | `service.py` → Service API only
- `workflow.py` → Orchestration | `config.py` → Config model only
- `downloader.py` → HTTP only | `rss_parser.py` → RSS parsing only

### Specs vs code

Don't embed `RFC-*`, `PRD-*`, or `UXS-*` identifiers in code, comments, CSS
class names, CLI strings, or user-visible copy. Use neutral names; keep
numbered references in `docs/rfc/`, `docs/prd/`, `docs/uxs/` only.

### Profile completeness (`config/profiles/*.yaml`)

Never ship a profile default referencing an enum value (`llm_pipeline_mode`,
`gi_insight_source`, `kg_extraction_source`, `summary_provider`, …) without
verifying a live code path reads it. Find the `Literal[...]` declaration,
grep the literal strings in `src/`, confirm every value has a dispatch arm
that does something different. **Half-wired features are worse than no
feature** — see #643 Phase 3C near-miss for the canonical example.

### Eval-track fixtures + silvers (`data/eval/`)

- **v2 generator is deterministic** since #903: same spec → identical
  transcripts across runs / `PYTHONHASHSEED` values. Don't reach for env
  pinning to "stabilize" output, and don't reintroduce `hash(...)` for
  seeding — use `_stable_seed(s)` (md5-based) the way
  `scripts/eval/data/generate_v2_transcripts.py` does. Regression test in
  `tests/unit/scripts/eval/data/test_generate_v2_transcripts.py`.
- **Silver naming convention** (v1↔v2):
  - `silver_<provider>_<cell>_v1` — autoresearch v1 winners over v1 sources.
  - `silver_<provider>_smoke_v2` / `_smoke_v2_bullets` — v2 source content,
    smoke scale (5 eps).
  - `silver_<provider>_kg_v2_paragraph` / `_kg_v2_bullets` — v2 source
    content, benchmark scale (15 eps from `curated_5feeds_kg_v2`).
  - `silver_<provider>_benchmark_v2_*` — **autoresearch-v2-framework era**,
    references v1 sources via `curated_5feeds_benchmark_v2`. The framework
    "_v2" is unrelated to v2 source content.
- v1 silvers / baselines are frozen; never overwrite. Layer-suffix dataset
  naming (`_kg`, `_cil`, `_cleaning`) documented in
  `data/eval/datasets/README.md`.

### GI/KG viewer UX (`web/gi-kg-viewer/`)

When changing viewer UX (visible copy, labels, routes, panels, layout, theme
tokens, accessible names used by tests), update in order:

1. `e2e/E2E_SURFACE_MAP.md` — automation contract first
2. Playwright — `e2e/*.spec.ts`, helpers; run `make ci-ui-fast`
3. `docs/uxs/` — `VIEWER_IA.md` for shell-IA changes; feature UXS for visual /
   token contracts

User-reported viewer bugs: reproduce + re-validate with **Chrome DevTools MCP
by default** (Playwright MCP only when clearly better — say so in one line).
Tests alone are not a substitute. See `docs/guides/AGENT_BROWSER_LOOP_GUIDE.md`.

---

## Git workflow

### Default: feature branch + PR. Exception: hotfix direct-to-main

- DEFAULT: `git checkout -b feat/name`, push, PR, merge.
- HOTFIX direct-to-main allowed ONLY when ALL apply: (a) fix for a regression
  already on main; (b) single-concern, small (≤ 50 LOC); (c) user explicitly
  asked for hotfix path or context makes it unambiguous.
- Hotfix protocol: pull main → edit → run pre-commit checks locally
  (`make lint` / `make docs` / spot pytest) → commit with `fix:` / `hotfix:`
  prefix → ask for push approval → push → watch CI on main → fix-forward if red.

### Never commit / push without explicit user approval

- `git status` → `git diff` → show user → wait for approval → THEN commit.
- After applying edits, briefly summarize and ask "Keep these changes or undo
  any of them?"
- Default: stage specific files for the current task (`git add <file>`).
  Avoid blind `git add -A` unless the user explicitly says so.

### Always rebase before pushing a feature branch

- BEFORE the first push of a feature branch: `git fetch origin main && git rebase origin/main`.
- BEFORE each subsequent push: same — rebase against the latest main so the PR
  diff is always against current main, not a stale base.
- Exception: hotfix-direct-to-main (already covered above) does not need rebase
  — it goes straight onto main.
- Why: linear history when the PR lands; PR diff shows only the branch's
  changes (no "behind by N"); latent main-vs-branch conflicts surface earlier
  than at merge time.
- Force-push is REQUIRED after a rebase. Use `git push --force-with-lease`
  (not `--force`) on feature branches so a teammate's concurrent push isn't
  silently overwritten.
- Force-push to `main` / `master` remains forbidden (covered by the hotfix
  section above).
- If a rebase produces conflicts, STOP and show the user the conflict files
  alongside `git status` output before resolving. Don't attempt resolution
  unilaterally on files outside the branch's own scope.

### Active-merge safety (when `.git/MERGE_HEAD` exists)

- NEVER `git stash` during a merge — destroys conflict resolutions.
- NEVER `git checkout <ref> -- <file>` during a merge — destroys resolved
  content. Only safe variants: `git checkout --ours/--theirs <file>` with
  explicit approval.
- NEVER overwrite local files with remote content without showing the diff
  and getting approval.
- NEVER `git checkout -- <path>` / `git restore <path>` to discard
  uncommitted edits unless explicitly approved.
- NEVER delete gitignored paths when "cleaning" — they may hold operator-local
  configs, experiments, or secrets.

---

## Tool usage

### Make commands, never direct tools

- **Use**: `make test`, `make format`, `make lint`, `make fix-md`,
  `make ci-fast`, `make docs`.
- **Don't use**: raw `pytest`, `black .`, `flake8`, `npx markdownlint-cli2`.
  Exception: `git`; Python: `.venv/bin/python3 script.py`.

### Viewer (`web/gi-kg-viewer/`): `npm run …` / `node_modules/.bin/<tool>`, never `npx`

The viewer pins `@playwright/test` but not a separate `playwright` package.
`npx playwright` silently fetches a fresh copy from the npm registry → two
module instances → "did not expect test.describe() to be called here" → worker
SIGKILL (exit 137). Easy to misread as user-canceled or OOM.

- **Use**: `npm run test:e2e -- <args>`, `npm run test:unit`, `npm run dev`,
  `npm run build`, `./node_modules/.bin/playwright test …`.
- Details: `docs/guides/POLYGLOT_REPO_GUIDE.md` (*Invoking viewer tools*).

### Foreground only — NEVER background for make/test/git/pip

- Background allowed ONLY for long-running servers (`mkdocs serve`, `make serve-api`).
- ALL `make`, `pytest`, `git`, `pip` MUST be foreground.
- Don't pipe to `| tail -N` / `| head` when the user is watching — most
  harnesses can't stream Bash stdout live; the user only sees output when the
  command exits. Prefer no pipe; if a command genuinely produces > 1k lines,
  use a generous tail (`tail -100`).

### Process safety — ML workloads

- NEVER retry a `make` that produced no output (likely zombie).
  Run `make check-zombie`.
- NEVER run multiple `make ci` / `ci-fast` / `ci-ui-fast` / `test`
  concurrently — process pileup on macOS causes unkillable zombies.
- After any hung/killed `make`, run `make cleanup-processes`.

### Update venv after dependency changes

After ANY edit to `[project.dependencies]` or `[project.optional-dependencies]`
in `pyproject.toml`, run `pip install --upgrade -e .[dev]` BEFORE running
tests or committing.

---

## Code quality and testing

### CI gating: run the right validation, not always the heaviest

- Default: `make ci-fast` before committing.
- Viewer-heavy diff: `make ci-ui-fast` (Playwright instead of Python e2e).
- Skip a duplicate full run when the same target already passed in this
  session and no substantive edits were made after.
- **No redundant ci-fast runs** to verify a small fix. Run the subtarget
  (`make docs`, `make lint`, `make type`, `make format`) — 10s vs 10min.
- **Documentation-only commits** (markdown under `docs/**`, `**/*.md`,
  `mkdocs.yml`, no `src/` or runtime code): use `make fix-md` (if needed),
  `make lint-markdown`, and `make docs`. Skip `make ci-fast`.
- **Pushing docs changes to `main` is a HARD GATE on `make docs` passing
  locally first.** `make docs` runs mkdocs in **strict mode** — any
  unresolved cross-reference (e.g., a link to a file you renamed without
  updating its referrers) fails the build with `Aborted with N warnings in
  strict mode!`. Pre-commit hooks do NOT catch this; only `make docs`
  does. Nightly will fail the next morning if you skip this step.
  - **After ANY file rename inside `docs/`**, also run
    `grep -rn "<old-filename-stem>" docs/` to find stale referrers before
    `make docs` — strict mode catches them eventually, but a grep is
    seconds vs. a 12 s build cycle per attempt.
- **Infra / CI / ops-only commits**: when the operator waives `make ci-fast`,
  still run `make lint-markdown` for markdown changes and `make docs` for
  doc-input changes. Otherwise no extra default local target.
- Why GitHub still runs Python CI for `Makefile` / workflow edits:
  `python-app.yml` `on.push.paths` includes those files. Path-filter
  behavior, not "the repo thought you changed Python."

### Final validation before push: real episodes, not just unit tests

Mocked unit tests prove dispatch routes correctly; they do NOT prove the
feature works against real provider responses, real transcripts, or the real
end-to-end pipeline. Before pushing any change touching a production pipeline
stage:

1. Run one real episode end-to-end with the changed code path (`.env` keys
   checked first).
2. Measure the claim numerically (LLM calls, file size, KG nodes — whatever
   the change claims to improve).
3. Inspect one artifact by eye.
4. Only then push.

### Workarounds vs. fixes

"Works in unit tests but not through the real pipeline" is a signature failure
mode. Half-wired features (Literal value with no dispatch arm, profile default
with no live code path, method exists but pipeline still calls the old one)
are regressions, not stubs. Don't ship them.

### Document location

- **Agent journal** (`.journal/`, git-ignored, survives `/clear`): EVERY agent keeps
  a running journal here — plans, approaches, decisions, findings, run logs: anything
  you'd otherwise lose on context-clear. One file per work-stream, named
  `YYYY-MM-DD-<topic>.md`; append as you work so the next session can recover context.
  This is the default home for working notes. Only `.journal/README.md` is tracked.
- **Committed WIP docs** (`docs/wip/`): analysis/plans meant to be shared or shipped
  with a PR (tracked, listed in `WIP_README.md`). Promote a journal note here when it
  needs to travel with the code. NEVER `docs/analysis/` or `docs/plan/`.

---

## CodeQL alert dismissals

- **Trigger**: CodeQL fails on a PR.
- **Step 1**: Read `docs/ci/CODEQL_DISMISSALS.md` — enumerates every alert
  type, sanitiser chain, and full dismissal inventory.
- **Step 2**: Classify:
  - **(a) Known type, same pattern** (matches a numbered type, uses the same
    documented sanitiser chain): dismiss as false positive via `gh api`, add
    a row, tell the user.
  - **(b) New type or broken chain**: do NOT dismiss. Explain the taint flow,
    propose a code fix routing through an existing sanitiser, and wait for
    explicit user approval. If approved, add the new type to the registry.
- **Step 3**: Never dismiss silently. State alert number(s), file/line, type
  matched, and action.
- Inline `# codeql[...]` pragmas DO NOT actually suppress — they are docs
  only. Dismiss via `gh api` and log in the registry.

---

## GitHub

- Use MCP GitHub server tools when available; fall back to `gh` CLI if MCP
  is unavailable.
- ALWAYS use the correct GitHub username (check with `mcp_github_get_me`, not
  Mac username).

---

## Essential commands

```bash
make format        # black + isort
make lint          # flake8
make type          # mypy
make ci-fast       # Fast tests before commit (~6-10 min)
make ci-ui-fast    # Like ci-fast, Playwright instead of Python tests/e2e
make ci            # Full CI suite (~10-15 min)
make docs          # Build docs (before committing doc changes)
make lint-markdown # Check markdown
make fix-md        # Auto-fix markdown
```

### Materialize autoresearch decisions in the registry, regenerate profiles

After **every** autoresearch finding that changes a default — transcription
backend, summary model, GI/KG thresholds, anything — the lifecycle is:

```text
1. Run experiment(s)              →  data/eval/runs/<run_id>/
2. Score with finale judges       →  data/eval/runs/finale/<tag>/
3. Write eval report              →  docs/guides/eval-reports/EVAL_*.md
4. ★ MATERIALIZE DECISION ★       →  src/podcast_scraper/providers/ml/model_registry.py
     - Add or update StageOption / ProfilePreset entries
     - research_ref points back at the eval report from step 3
     - headline_metric + measured_at for provenance
5. ★ REGENERATE PROFILES ★        →  config/profiles/*.yaml
     - Update profile YAMLs to match the registry preset
     - Comment the change with the StageOption id + research_ref
6. Tests                          →  tests/integration/providers/ml/...
     - Behavior test: resolve_profile_to_settings(name) returns the expected
       provider/model/endpoint triple
7. Verify the runtime sees it     →  make profile-drift-check
     - Each profile YAML that declares `profile: <preset>` is cross-checked
       against the registry. Wired into make ci-fast — drift is now a CI
       failure, not a documentation drift problem.
```

Steps **4 and 5 are the part that's easy to skip** and the most expensive to
fix later. An eval report alone documents the decision; without 4+5 the
runtime never adopts it. With them, "what is production running today?" has
a single, machine-readable answer.

Don't hand-flip a profile YAML without a matching registry update. Don't
amend an eval report's verdict without updating the registry entry it
justified.

See:

- `docs/wip/RESEARCH_POWERED_REGISTRY_PLAN.md` — vision + migration path
- `docs/adr/ADR-048-centralized-model-registry.md` — original ML-only ADR
- `docs/rfc/RFC-044-model-registry.md` — registry RFC
- `src/podcast_scraper/providers/ml/model_registry.py` — canonical code

### Keep the laptop awake during long sweeps — `caffeinate`

When dispatching a sweep / eval that's expected to run >15 min while
the operator may step away — especially anything routed through
Tailscale to the DGX — wrap the work or arm a background timer:

```bash
caffeinate -i -t 7200 > /dev/null 2>&1 &
disown                                          # 2-hour safety net
# or, around a single command:
caffeinate -i make ci-fast                       # exits with the wrapped cmd
```

- `-i` = prevent idle system sleep (display can still sleep; CPU and
  networking stay alive — that's all we need for keeping Tailscale up).
- `-t N` = auto-exit after N seconds. Belt-and-braces so a forgotten
  background `caffeinate` doesn't outlive the session.

**Why it matters here**: macOS goes into a deeper-than-normal idle
state after the laptop sits idle, which drops Tailscale's TCP
connections. A long-running `requests.post(...)` against the DGX over
Tailscale will hang from the laptop's perspective even though the DGX
finished the work and returned 200 OK ages ago — the response just
never crossed the dead connection.

Symptom (the 2026-06-12 incident this rule came from):
`elapsed=7946.8s` for an episode the server processed in 34 min and
the harness's HTTP timeout was set to 1500s. Server-side log says
`POST 200 OK` at minute 34; client-side blocked until the laptop woke
up ~90 min later.

**When to arm it**: any sweep where the wall-clock estimate exceeds
the laptop's idle-sleep timeout (default ~10-15 min on battery, often
longer on AC). Cheaper than babysitting and far cheaper than
re-running a multi-hour sweep with corrupted timing data.

---

## Detail file references

When the situation matches, load before proceeding:

| Situation | Load |
| --- | --- |
| Writing/modifying Python code | `.cursor/rules/coding-standards.mdc` |
| Writing or modifying tests | `.cursor/rules/testing-strategy.mdc` + `docs/guides/TESTING_GUIDE.md` |
| Editing `tests/unit/**` | `docs/guides/UNIT_TESTING_GUIDE.md` (must not depend on non-`[dev]` extras; never use `pytest.importorskip()` to sidestep) |
| Editing markdown / docs | `.cursor/rules/documentation.mdc` |
| PRD/UXS/RFC/ADR or `docs/architecture/**` | `.cursor/rules/engineering-process.mdc` + `docs/guides/ENGINEERING_PROCESS.md` |
| Git worktree operations | `.cursor/rules/git-worktree.mdc` |
| Refactoring / module boundaries | `.cursor/rules/module-boundaries.mdc` |
| About to replace tracked content | `.cursor/rules/git-working-tree-safety.mdc` |
| Viewer UI bug fixing | `.cursor/rules/agent-browser-ui-fixes.mdc` + `docs/guides/AGENT_BROWSER_LOOP_GUIDE.md` |
| About to commit or push | `.cursor/rules/ai-guidelines.mdc` |
| Configuring profiles | `config/profiles/*.yaml` — see *Profile completeness* above |
| Viewer UX changes | `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` first |

`PRD` = product what/why · `UXS` = UI contract when needed · `RFC` = technical
how (Draft → Completed) · `ADR` = immutable decision record (**Accepted**),
short, sourced from RFC discussion. Full flow + templates:
`docs/guides/ENGINEERING_PROCESS.md`.

---

## Documentation prose

In `docs/**` (PRD/RFC/ADR/guides/api/wip), `AGENTS.md`, `.cursorrules`,
`CLAUDE.md`, `.ai-coding-guidelines.md`, and `.cursor/rules/*.mdc`: do NOT
use checkmark / cross / clipboard emoji or decorative tick marks for status,
lists, or tables. Use plain words (Yes/No, Run/Skip, Good/Bad). Full rule:
`.cursor/rules/documentation.mdc`.

PRDs use **Implemented** / Partial / Draft. **Completed** is for RFC and ADR
headers, not PRDs.

---

**Source of truth:** this file.
**Tool overlays:** `CLAUDE.md` (Claude Code), `.cursorrules` (Cursor).
**Detail:** `.ai-coding-guidelines.md` / `docs/guides/*`.
