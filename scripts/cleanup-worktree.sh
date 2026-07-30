#!/usr/bin/env bash
#
# cleanup-worktree.sh — clean the house after an agent finishes work.
#
# Kills dev servers, test runners, and automation browsers that were started
# while working in THIS git worktree, and nothing else. The moment you're done
# deploying / testing / comparing, run this so the next session starts from a
# quiet machine (see AGENTS.md "Clean up after yourself").
#
# ISOLATION BOUNDARY — the worktree path (git rev-parse --show-toplevel).
# One checkout = one agent = one shell, so scoping to the worktree scopes to
# this agent. Attribution is by process CWD (not by name), so parallel worktrees
# / agents / shells are never touched — even another podcast_scraper checkout
# (e.g. podcast_scraper-FUTURE). The ONLY thing that could span worktrees is an
# orphaned automation browser (see Path B) — that is opt-in and OFF by default
# precisely to keep this strictly single-worktree.
#
# SCOPE — what the DEFAULT run kills (and only this):
#   A. Any dev/test/pipeline process whose CWD is inside this worktree — plus every
#      descendant it spawned (browsers, renderers, workers). The podcast_scraper
#      surface, from the Makefile:
#        * python        — the catch-all: `python -m podcast_scraper.cli serve`
#                          (serve-api), the e2e mock server, the :8777 metrics HTTP
#                          server, `python -m pytest`, `python -m mkdocs`, and the
#                          ML/pipeline jobs (transcription/diarization/summarization,
#                          torch/whisper/spacy) + long sweeps.
#        * uvicorn/gunicorn — named server workers the CLI serve may spawn.
#        * pytest        — the bare `.venv/bin/pytest` entrypoint.
#        * playwright    — viewer / learning-player / stack-test E2E (node).
#        * vitest        — viewer unit tests.
#        * vite/esbuild  — viewer + learning-player dev server & build (serve-ui/app).
#        * ffmpeg        — audio pipeline (transcode / segment).
#        * caffeinate    — the backgrounded keep-awake for long sweeps (AGENTS.md
#                          warns it can outlive the session — this reaps it).
#        * npm run dev/serve/preview — the node dev servers (parent of vite).
#      NOTE — Docker is deliberately NOT here: the compose stack + stack-test's own
#      Playwright/buildx orphans are owned by `make stack-test-reap` ($(CURDIR)-scoped);
#      killing a docker build mid-flight from here would be wrong. Any overlap on
#      Playwright is benign (SIGTERM to an already-dead pid is a no-op).
#
# OPT-IN (--orphans) — additionally kills:
#   B. Orphaned automation browsers: chrome-headless-shell / chromium on a temp
#      `playwright_*profile`, reparented to launchd (ppid 1). These are NOT
#      attributable to any worktree (cwd is `/`), so by default they are only
#      REPORTED, never killed. Pass --orphans only when you know the orphan is
#      yours and no other agent is mid-playwright-run.
#
# OPT-IN (--stale-loops) — additionally sweeps THIS PROJECT'S FAMILY of worktrees:
#   C. Runaway agent poll-loops: a `gh run …` call stuck in an unbounded
#      while/until loop, older than MAX_LOOP_AGE (default 2h, CLEANUP_MAX_LOOP_AGE),
#      whose CWD is in the same project family as this worktree (project-name prefix,
#      e.g. run from ~/Projects/podcast_scraper-infra -> ~/Projects/podcast_scraper*,
#      so podcast_scraper-FUTURE is swept but orrery is NOT). These leak when a
#      session dies before its SessionEnd reaper fires, orphaned in a sibling worktree
#      nothing else targets (seen: loops surviving 8-14 days). It reaches sibling
#      worktrees but never another project — the family + shape + age + CWD stack
#      can't match a live watch or an MCP server. Mirrors the auto sweep in
#      ~/.claude/hooks/session-reap.sh.
#
# NEVER killed — hard guarantees, even with --orphans / --stale-loops:
#   * Anything whose CWD is ANOTHER PROJECT'S worktree — EXCEPT a --stale-loops match,
#     which reaches sibling worktrees in THIS project's family only, gated to
#     gh-run + unbounded-loop + age.
#   * Your real browser: any Chrome/Chromium using the default
#     "Library/Application Support/Google/Chrome" profile, or Claude Desktop.
#   * lean-ctx / MCP servers / language servers that live in the tree (a lean-ctx
#     process is only matched by --stale-loops if it IS a runaway gh-run loop).
#   * A LIVE playwright run anywhere, or a gh-run watch younger than MAX_LOOP_AGE.
#   * This script itself and its parent shell.
#
# Usage:
#   bash scripts/cleanup-worktree.sh                # kill Path A, report Path B
#   bash scripts/cleanup-worktree.sh --dry-run      # report only, kill nothing
#   bash scripts/cleanup-worktree.sh --orphans      # also kill unattributable orphans
#   bash scripts/cleanup-worktree.sh --stale-loops  # also sweep global stale gh-run loops
#
set -uo pipefail

DRY=0
ORPHANS=0
STALE=0
for a in "$@"; do
  case "$a" in
    -n|--dry-run)   DRY=1 ;;
    --orphans)      ORPHANS=1 ;;
    --stale-loops)  STALE=1 ;;
    -h|--help)      sed -n '2,75p' "$0"; exit 0 ;;
    *) echo "usage: $0 [--dry-run] [--orphans] [--stale-loops]" >&2; exit 2 ;;
  esac
done

MAX_LOOP_AGE="${CLEANUP_MAX_LOOP_AGE:-7200}"   # 2h — stale-loop age floor (--stale-loops)

WT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "cleanup-worktree: not inside a git worktree — refusing to run" >&2
  exit 1
}
SELF=$$

# ps etime ([[dd-]hh:]mm:ss) -> elapsed seconds. `10#` guards ps zero-pad (08→octal).
etime_secs() {
  local e="$1" d=0 hms
  [ -z "$e" ] && { echo 0; return; }
  case "$e" in *-*) d="${e%%-*}"; hms="${e#*-}" ;; *) hms="$e" ;; esac
  local IFS=:
  # shellcheck disable=SC2086  # deliberate: split $hms on IFS=: into h/m/s positionals
  set -- $hms
  case $# in
    3) echo $(( 10#$d*86400 + 10#$1*3600 + 10#$2*60 + 10#$3 )) ;;
    2) echo $(( 10#$d*86400 + 10#$1*60 + 10#$2 )) ;;
    *) echo $(( 10#$d*86400 + 10#${1:-0} )) ;;
  esac
}

pid_cwd()  { lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1; }
pid_cmd()  { ps -p "$1" -o command= 2>/dev/null; }
pid_ppid() { ps -p "$1" -o ppid= 2>/dev/null | tr -d ' '; }

# recursively echo all descendant pids of $1
descendants() {
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null); do
    echo "$c"
    descendants "$c"
  done
}

is_protected_cmd() {
  case "$1" in
    *"Library/Application Support/Google/Chrome"*) return 0 ;;  # your real Chrome
    *"Library/Application Support/Claude"*)        return 0 ;;  # Claude Desktop
    *lean-ctx*|*mcp*|*language-server*|*tsserver*|*copilot*) return 0 ;;
  esac
  return 1
}

# is $1 exactly this worktree, or strictly inside it?  (trailing-slash discipline
# prevents a sibling like ".../podcast_scraper-FUTURE" from matching ".../podcast_scraper")
in_worktree() {
  case "$1" in
    "$WT_ROOT"|"$WT_ROOT"/*) return 0 ;;
    *) return 1 ;;
  esac
}

# Family prefix for a project path: $HOME/<container>/<base>, <base> = instance dir
# name up to its first '-'.  ~/Projects/podcast_scraper-infra & ~/Projects/podcast_
# scraper-FUTURE both -> ~/Projects/podcast_scraper (one family); ~/.treehouse/orrery-
# 311982/... -> ~/.treehouse/orrery. Generic, not layout-specific.
family_prefix() {
  local p="${1%/}" rel container instance
  case "$p" in "$HOME"/*/*) : ;; *) echo ""; return ;; esac
  rel="${p#"$HOME"/}"; container="${rel%%/*}"; rel="${rel#*/}"; instance="${rel%%/*}"
  echo "$HOME/$container/${instance%%-*}"
}

# ── Path C (--stale-loops): family-scoped stale poll-loop sweep ──────────────
# Companion to the SessionEnd reaper's automatic sweep (~/.claude/hooks/session-
# reap.sh). Opt-in because it reaches into SIBLING worktrees — but scoped to THIS
# worktree's project FAMILY (project-name prefix of WT_ROOT — see family_prefix),
# never other projects: run from podcast it reaps podcast-family loops only, never an
# orrery one. Reaps a `gh run …` call stuck in an unbounded while/until loop older
# than MAX_LOOP_AGE (the runaway CI-watchers agents leak when a session dies before
# its reaper fires — seen: loops surviving 8-14 days in a sibling worktree). The
# stack (same-family CWD + gh-run + unbounded loop shape + age floor) can't match a
# live watch or an MCP server; a runaway loop's lean-ctx wrapper does (its cmdline
# carries the loop body), so wrapper + shell both die.
sweep_stale_loops() {
  local pid cmd age cwd n=0 fam
  fam="$(family_prefix "$WT_ROOT")"
  echo "── stale poll-loops (family=${fam:-none}, > ${MAX_LOOP_AGE}s) ───────────"
  [ -z "$fam" ] && { echo "  no family derivable from $WT_ROOT — skipping"; return; }
  for pid in $(pgrep -f 'gh run ' 2>/dev/null | sort -un); do
    [ "$pid" = "$SELF" ] && continue
    cmd="$(pid_cmd "$pid")"; [ -z "$cmd" ] && continue
    case "$cmd" in *"while true"*|*"until "*) : ;; *) continue ;; esac  # unbounded loop only
    age="$(etime_secs "$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')")"
    [ "${age:-0}" -ge "$MAX_LOOP_AGE" ] || continue                     # young = maybe live watch
    cwd="$(pid_cwd "$pid")"
    case "$cwd" in "$fam"|"$fam"*) : ;; *) continue ;; esac             # same project family only
    n=$((n+1))
    printf '  %-7s age=%-8s %s\n' "$pid" "${age}s" "$(printf '%s' "$cmd" | cut -c1-70)"
    if [ "$DRY" = "0" ]; then
      kill -TERM "$pid" 2>/dev/null || true
      ( sleep 2; kill -KILL "$pid" 2>/dev/null ) >/dev/null 2>&1 &
    fi
  done
  if [ "$n" = 0 ]; then echo "  none found"
  elif [ "$DRY" = "1" ]; then echo "  ↳ $n loop(s) matched — dry-run, killed nothing"
  else echo "  ↳ $n loop(s) swept"; fi
}
[ "$STALE" = "1" ] && sweep_stale_loops

owned=()        # Path A: cwd inside this worktree, + descendants
orphan_roots=() # Path B: unattributable orphaned automation browsers

# ── A. runners whose cwd is inside this worktree, + their descendants ─────────
# podcast_scraper stack: python backend/pipeline + viewer & learning-player (node) +
# audio (ffmpeg) + long-sweep caffeinate. The cwd guard is what makes a broad match
# (e.g. bare `python`, which covers serve-api / pytest / mkdocs / ML jobs) safe — only
# processes rooted in THIS worktree are touched; MCP / lean-ctx / language-servers are
# protected below, and Docker builds are left to `make stack-test-reap`.
for pid in $(pgrep -f 'uvicorn|gunicorn|pytest|playwright|vitest|vite|esbuild|ffmpeg|caffeinate|npm run dev|npm run serve|npm run preview|python' 2>/dev/null | sort -un); do
  [ "$pid" = "$SELF" ] && continue
  cmd="$(pid_cmd "$pid")"
  is_protected_cmd "$cmd" && continue
  if in_worktree "$(pid_cwd "$pid")"; then
    owned+=("$pid")
    for d in $(descendants "$pid"); do owned+=("$d"); done
  fi
done

# ── B. orphaned automation browsers (ppid 1) — UNATTRIBUTABLE, report-only ────
for pid in $(pgrep -f 'chrome-headless-shell|playwright_.*profile|chromium.*--headless' 2>/dev/null | sort -un); do
  cmd="$(pid_cmd "$pid")"
  is_protected_cmd "$cmd" && continue
  case "$cmd" in
    *chrome-headless-shell*|*playwright_*profile*) ;;
    *) continue ;;
  esac
  [ "$(pid_ppid "$pid")" = "1" ] || continue   # only truly abandoned ones
  orphan_roots+=("$pid")
done

# ── build the kill list (Path A always; Path B only with --orphans) ──────────
declare -a candidates=()
[ "${#owned[@]}" -gt 0 ] && candidates+=("${owned[@]}")
if [ "$ORPHANS" = "1" ] && [ "${#orphan_roots[@]}" -gt 0 ]; then
  for pid in "${orphan_roots[@]}"; do
    candidates+=("$pid")
    for d in $(descendants "$pid"); do candidates+=("$d"); done
  done
fi

kill_list=()
if [ "${#candidates[@]}" -gt 0 ]; then
  while IFS= read -r pid; do
    [ -n "$pid" ] && [ "$pid" != "$SELF" ] && kill_list+=("$pid")
  done < <(printf '%s\n' "${candidates[@]}" | sort -un)
fi

echo "cleanup-worktree: $WT_ROOT"

# report unattributable orphans that we are deliberately NOT killing by default
if [ "$ORPHANS" = "0" ] && [ "${#orphan_roots[@]}" -gt 0 ]; then
  echo "── orphaned automation browsers (NOT this worktree's to claim) ─────────"
  for pid in "${orphan_roots[@]}"; do
    printf '  %-7s %s\n' "$pid" "$(pid_cmd "$pid" | cut -c1-84)"
  done
  echo "  ↳ unattributable to any worktree — left alone. Pass --orphans to reap."
fi

if [ "${#kill_list[@]}" -eq 0 ]; then
  echo "── nothing running for this worktree — house is already clean"
  exit 0
fi

echo "── targets ─────────────────────────────────────────────────────────────"
for pid in "${kill_list[@]}"; do
  printf '  %-7s %s\n' "$pid" "$(pid_cmd "$pid" | cut -c1-90)"
done

if [ "$DRY" = "1" ]; then
  echo "── dry-run: nothing killed ────────────────────────────────────────────"
  exit 0
fi

echo "── killing (SIGTERM → SIGKILL) ────────────────────────────────────────"
kill -TERM "${kill_list[@]}" 2>/dev/null
sleep 2
survivors=()
for pid in "${kill_list[@]}"; do
  kill -0 "$pid" 2>/dev/null && survivors+=("$pid")
done
if [ "${#survivors[@]}" -gt 0 ]; then
  kill -KILL "${survivors[@]}" 2>/dev/null
  sleep 1
fi

remaining=0
for pid in "${kill_list[@]}"; do
  kill -0 "$pid" 2>/dev/null && { remaining=1; printf '  STILL ALIVE: %-7s %s\n' "$pid" "$(pid_cmd "$pid")"; }
done
if [ "$remaining" = "0" ]; then
  echo "cleanup-worktree: done — ${#kill_list[@]} process(es) cleared, house is clean"
else
  echo "cleanup-worktree: some processes survived SIGKILL (see above)" >&2
  exit 1
fi
