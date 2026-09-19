#!/bin/zsh
# Relabel harness for #2075, with a PROGRESS-BASED watchdog.
#
# Runs `--pipeline-stage relabel_only` over a set of feeds and keeps going when one wedges.
#
# WHY A WATCHDOG AT ALL (#1323): a vLLM chat call can stop receiving response headers while the
# server is demonstrably healthy, and the request is bounded only by the per-call transport
# timeout. Every feed in the 2026-09-19 pass wedged this way after its last episode.
#
# WHY PROGRESS-BASED, NOT FRESHNESS: the wedged processing loop logs "Processing loop cannot
# finish" once a minute, so an mtime/liveness watchdog is defeated by the stall's own logging.
# Progress here is the count of episodes whose speaker record has been rewritten.
#
# WHY THE FAST PATH: waiting out a blind timeout on every feed cost ~12 min x 48 feeds. The stall
# announces itself, so three of its diagnostics with no new episode is a confirmed stall (~3 min).
#
# No attribution data is lost to a kill: the speaker record is written BEFORE the wedge, which
# the measured relabelled=48/48 and 29/29 on killed feeds confirm. Downstream GI/KG for the last
# episode of a killed feed IS truncated.
#
# Usage: 2075_relabel_harness.sh <set-dir> <label>
#   <set-dir>/cmds.txt  lines of  slug|feed-url|corpus-dir|ids-file|feed-title
#   Copy each corpus first — relabel rewrites in place, and you need the BEFORE state to compare.
set -u
SETDIR=$1
LABEL=$2
STALL_S=${STALL_S:-720}     # 12 min with no NEW relabelled episode => kill this feed, move on
POLL_S=20

cd "${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
echo "HARNESS_START $LABEL $(date -u +%FT%TZ)"

# NOTE the 4th field is an ids.txt PATH, not a count. The *_run.sh scripts on disk pass it to
# --max-episodes, which is not what the 2026-09-16 pass actually ran: its logs say "restricting
# this run to 13 of 68 on-disk episodes", i.e. --reprocess-episode-ids. Matching the real run.
while IFS='|' read -r slug url corpus ids feed; do
  [ -z "$slug" ] && continue
  LOG="$SETDIR/$slug.log"
  NEPS=$(wc -w < "$ids" 2>/dev/null || echo "?")
  echo "=== $slug  ${NEPS} episodes  $feed"
  if [ -n "${WD_TEST_CMD:-}" ]; then
    # Test hook: exercise the REAL watchdog below against a scripted producer, so the probe
    # tests this loop rather than a copy of it. Never set in a real run.
    SLUG=$slug zsh -c "$WD_TEST_CMD" > "$LOG" 2>&1 &
  else
    .venv/bin/python -m podcast_scraper.cli "$url" \
      --profile dev_dgx_full --pipeline-stage relabel_only \
      --output-dir "$corpus" --single-feed-uses-corpus-layout \
      --reprocess-existing-only --reprocess-episode-ids "$ids" \
      > "$LOG" 2>&1 &
  fi
  PID=$!

  last_progress=0
  last_change=$(date +%s)
  killed=0
  while kill -0 $PID 2>/dev/null; do
    sleep $POLL_S
    # `grep -c` PRINTS 0 and EXITS 1 when there are no matches, so `|| echo 0` yields the
    # two-line string "0\n0" and every integer test below fails — silently disabling the very
    # watchdog this exists to provide. grep -c already always prints a count; only guard the
    # missing-file case.
    prog=$(grep -c "speaker record rewritten" "$LOG" 2>/dev/null)
    prog=${prog:-0}

    # FAST PATH: #1323 announces itself. The wedged processing loop logs
    # "Processing loop cannot finish: N job(s) enqueued ... {'running': N}" once a minute, so a
    # few of those with no new episode is a CONFIRMED stall, not slow work. Measured on this
    # corpus: every feed wedges after its last episode (f00 logged 13 of them, f01 logged 10),
    # which at the 12-minute generic timeout costs ~12 min x 48 feeds ~= 9.6 h of pure waiting.
    # Three diagnostics (~3 min) is enough to be certain, and the record is already written.
    stalls=$(grep -c "cannot finish" "$LOG" 2>/dev/null)
    stalls=${stalls:-0}
    if [ "$stalls" -ge 3 ] && [ "$prog" -le "$last_progress" ]; then
      echo "    WATCHDOG: #1323 signature ($stalls stall diagnostics, no new episode) at $prog relabelled — killing $slug"
      pkill -TERM -P $PID 2>/dev/null
      kill -TERM $PID 2>/dev/null
      sleep 5
      kill -9 $PID 2>/dev/null
      killed=1
      break
    fi
    now=$(date +%s)
    if [ "$prog" -gt "$last_progress" ]; then
      last_progress=$prog
      last_change=$now
    elif [ $((now - last_change)) -ge $STALL_S ]; then
      echo "    WATCHDOG: no new episode in $((now - last_change))s at $prog relabelled — killing $slug (#1323)"
      pkill -TERM -P $PID 2>/dev/null
      kill -TERM $PID 2>/dev/null
      sleep 5
      kill -9 $PID 2>/dev/null
      killed=1
      break
    fi
  done
  wait $PID 2>/dev/null
  rc=$?
  echo "    $slug exit=$rc relabelled=$last_progress watchdog_killed=$killed"
done < "$SETDIR/cmds.txt"

echo "HARNESS_DONE $LABEL $(date -u +%FT%TZ)"
