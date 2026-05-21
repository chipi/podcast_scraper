# Claude Code instructions for podcast_scraper

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
