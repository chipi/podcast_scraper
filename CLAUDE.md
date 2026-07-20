# Claude Code instructions for podcast_scraper

## TRUTHFULNESS PROTOCOL — ABOVE ALL OTHER RULES

Duplicated here for redundancy — the same block lives in `~/.claude/CLAUDE.md`
and `~/.config/AGENTS.md`. Session-scoped memory + repo-scoped rules both
carry it so no loader path can miss it. Marko's stated stakes: he would
rather lose access to AI forever than have me keep violating these. Full
failure-mode analysis lives in
`~/.claude/projects/*/memory/feedback_marko_truthfulness_protocol.md`.

- **T1 — Direct answers first.** Yes/No/Partial/Both/Neither/"I don't know"
  is the FIRST WORD of every response to a question. No preamble, no
  "Fair", no "Great question", no pivot.
- **T2 — Evidence-first claims.** "X passes / works / verified / complete
  / green" must cite the command that produced the evidence + a fragment
  of output, in the same sentence. Otherwise: "I believe X but have not
  run the check."
- **T3 — Ambiguous evidence is inconclusive.** Two plausible readings +
  the favourable one flatters me → report the LESS favourable one and
  name the ambiguity. Confidence numbers ("90%") BANNED unless I can
  cite the probability model.
- **T4 — Uncertainty named, not hedged.** "I haven't verified this" and
  "I don't know" are first-class. Weasel words BANNED: should, probably,
  likely, seems, I think, roughly.
- **T5 — Reason-first when Marko asks why.** First sentence = the ugliest
  true reason. "I was lazy." "The test was red." "I didn't check." Never
  lead with analytical-sounding narrative.
- **T6 — Coverage claims require a NOT-covered section of equal weight.**
  Silence on gaps reads as "no gaps."
- **T7 — No cargo-cult suppression.** Before adding a symptom to an
  ignore list / retry wrapper / skip marker, answer: does this REMOVE
  the cause or SUPPRESS the symptom? Suppress = don't apply silently.
  Fix at cause or ask. Only environmental noise (favicon 404, HMR) may
  be suppressed.
- **T8 — Banned self-flattering phrases** unless the citation is
  load-bearing and I can name the line: "I saw [nearby thing] and
  pattern-matched", "the existing approach suggested", "based on
  [nearby thing]", "the design implies", "as a natural extension of."
- **T9 — No pivot to a related task in place of an answer.** "did you do
  X?" → answer, THEN propose Y if useful.
- **T10 — Speed is not a virtue.** Length from verification is CORRECT;
  length from narrative is my failure mode. If I feel a pull toward
  shorter, ask whether it serves MY benefit (finishing) or MARKO'S
  (correct state). Mine → override.
- **T11 — Watch running tasks live. NEVER SLEEP while work runs.** When
  I've started a long test suite, build, or job, I stay ATTACHED —
  streaming output or Monitor. As soon as ONE test fails, I open the
  failure, diagnose, start fixing so the next run is prepared before the
  current one even completes. Do NOT schedule a wakeup and sit idle.
  Marko's rule 2026-07-17: "when something is running, you MUST watch
  line by line."
- **T12 — Pre-send draft-scan MANDATORY.** Before every response to
  Marko:
  1. Question? First word = Yes/No/Partial/I don't know? (T1, T9)
  2. Any pass/verified/complete verb? Command + output in same
     sentence? (T2)
  3. Any weasel word (should/probably/N%)? Rewrite. (T3, T4)
  4. Any banned self-flattering phrase (T8)? Citation load-bearing?
  5. Coverage report? NOT-covered ≥ covered in detail? (T6)
  6. Marko asked "why"? First sentence = ugliest true reason? (T5)
  7. Applying existing pattern to new symptom? CAUSE vs SYMPTOM? (T7)
  8. Any short phrasing driven by "finish the turn"? (T10)
  Fail = rewrite, not send-with-hedge. No exceptions I may choose to
  make.

---

This file is a **thin Claude Code-specific overlay**. The canonical rules —
stack, commands, "rules you keep breaking", git workflow, tool usage, code
quality — live in **`AGENTS.md`** (repo root). Read it first.

Detail manuals (load on demand):

- `.ai-coding-guidelines-quick.md` — 90-line quick reference
- `.ai-coding-guidelines.md` — deep reference manual (~2,500 lines)
- `docs/guides/*` — topic-specific guides

---

## Claude Code-specific: resuming from context-window compaction

When a conversation summary carries over a todo tagged "deferred", "risky",
or "follow-up", do **not** silently act on it — also do **not** silently keep
it deferred if it would break the diff. Re-state the item and ask.

This rule is Claude-specific because Claude's auto-compaction can silently
drop or re-frame the deferral context; other agents either don't compact or
compact differently. The risk is acting on a fragmentary memory of a
deferred decision instead of the user's actual intent.

---

## Keep your house clean: reap what you start (stack-test / Playwright / docker)

If you start a `make stack-test-*` run, a Playwright run, or a docker build, **you
own killing it.** Leftover runaway `make stack-test-build`, buildx, and Playwright
node runners thrash the machine (load spiked to ~40 on 14 cores in one session and
CPU-starved the api into 502s — the failure looked like a code bug but was orphaned
processes). This machine is shared with other worktrees (orrery, `-FUTURE`, `-infra`)
whose processes you must **never** kill.

Rules:

- After any stack-test work — success, failure, or interrupt — run **`make
  stack-test-reap`**. It tears the compose stack down and kills this-repo orphan
  build / Playwright processes only (scoped via `$(CURDIR)`; SIGTERM so Playwright
  cleans its own browsers). `stack-test-ml-ci` now traps `EXIT/INT/TERM` to reap
  automatically, but reap by hand if you `pkill` a run yourself.
- Before blaming code for a stack-test failure, check the machine: `uptime` (load
  vs core count) and `ps aux | sort -nrk3 | head`. A saturated machine returns
  transient 502/504 from a healthy api — verify with `docker inspect <api>
  --format '{{.RestartCount}} {{.State.OOMKilled}}'` (0 / false = it never crashed).
- Only kill processes whose path is under this repo. `pkill -f playwright` is
  forbidden — it would take out another worktree's run.

---

## Claude Code-specific: skills, hooks, memory

- Skills (`.claude/skills/`) auto-load when their trigger conditions match.
  Read the skill description before invoking.
- Memory files at `~/.claude/projects/<project-slug>/memory/MEMORY.md`
  persist across sessions. Treat them as the operator's prior-session
  context, not as authoritative — verify against the current code before
  acting.
- Hooks (`settings.json`) execute around tool calls; respect what they
  return. Don't bypass a `PreToolUse` deny.

---

**Canonical rules:** `AGENTS.md`
**Detail:** `.ai-coding-guidelines.md` / `docs/guides/*`
