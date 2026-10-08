#!/bin/bash
# The iOS half of the device performance scan — `make perf-ios` (2026-10-08).
#
# Runs MemoryWalkTests (ios/uitests) against the app already installed on the booted simulator; at
# each `=====MEM_STEP <name>=====` it samples, with `footprint`, the simulator's WebKit content
# process (where the page's images, layers and JS live) and the app process. Writes a Markdown
# report. iOS has no DevTools protocol into a WKWebView, so this is the device-side number.
#
# usage: ios-memory-walk.sh <out-dir> <derived-data> [token] [walk]
#   token  a session token to sign in with (optional; without it the app's current session is used)
#   walk   comma-separated deep-link paths, e.g. "episode/<slug>,episode/<slug>?panel=notes,topic/topic:x"
set -u
OUT=$1; DD=$2; TOKEN=${3:-}; WALK=${4:-}
SIM=${IOS_SIM:-iPhone 17}
mkdir -p "$OUT"
REPORT="$OUT/report.md"
youngest() { ps -axo pid=,etime=,command= | grep -E "$1" | grep -v -E "grep|ios-memory-walk" | awk '{print $2, $1}' | sort | head -1 | awk '{print $2}'; }
mb() { footprint -p "$1" 2>/dev/null | grep -o 'Footprint: [0-9.]* [KMG]B' | head -1 | sed 's/Footprint: //'; }
{
  echo "# iOS memory walk — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "Simulator: $SIM · footprint = phys_footprint (what jetsam counts)"
  echo
  echo "| step | WebKit content process | app process |"
  echo "|---|---|---|"
} > "$REPORT"
cd "$(dirname "$0")/../../ios/uitests" || exit 1
xcodegen generate >/dev/null
TEST_RUNNER_LP_TOKEN="$TOKEN" TEST_RUNNER_LP_WALK="$WALK" \
  xcodebuild test -project OfflineSpike.xcodeproj -scheme OfflineSpikeUITests \
    -destination "platform=iOS Simulator,name=$SIM" \
    -only-testing:OfflineSpikeUITests/MemoryWalkTests \
    -derivedDataPath "$DD" CODE_SIGNING_ALLOWED=NO 2>&1 |
  while IFS= read -r line; do
    case "$line" in
      *"=====MEM_STEP "*)
        step=${line#*MEM_STEP }; step=${step%=====*}
        sleep 5  # let the screen paint; the test holds it longer than this
        wc=$(youngest "simruntime.*WebKit\.WebContent"); app=$(youngest "/App\.app/App$")
        row="| \`${step}\` | $(mb "$wc") | $(mb "$app") |"
        echo "$row" | tee -a "$REPORT"
        ;;
      *"error:"*|*"TEST FAILED"*|*"TEST SUCCEEDED"*) echo "$line" ;;
    esac
  done
echo "✓ report: $REPORT"
