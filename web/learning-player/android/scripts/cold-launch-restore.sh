#!/bin/bash
# Android: after the OS ends the app in the background, does the next launch reopen where the
# listener left off? (#2278) — the twin of iOS ColdLaunchRestoreTests.
#
# Usage: cold-launch-restore.sh [origin-port] [out-dir]
#
# Driven from the SHELL for the same reason as magic-link-cold-launch.sh: instrumentation runs
# inside the app's own process, so a test that kills the app kills itself.
#
# Steps: sign in through the mock provider's native callback → open a fixture episode (loads the
# player) → tap Discover → HOME → `am kill` (what Android does to a cached background process;
# the app's stored data survives, exactly as after a real eviction) → launch → read the screen.
# Passes when the relaunch shows Discover's own "Browse" heading AND the episode title in the
# bottom third of the screen (the mini-player) — a Discover list row could also carry the title.
#
# Prereqs: `make android-app-install` (emulator up, origin on :4174 reversed, debug app installed).
set -u
PORT=${1:-4174}
OUT=${2:-/tmp/lp-android-restore}
ADB=${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}
PKG=app.closelistening.player
IDENTITY=coldlaunchrestore
SLUG=p09-a4bbb5dde3
TITLE="Risk Is a Systems Property"
mkdir -p "$OUT"

raw_dump() { $ADB shell uiautomator dump /sdcard/lp-ui.xml >/dev/null 2>&1; $ADB shell cat /sdcard/lp-ui.xml 2>/dev/null; }
# A freshly booted emulator often raises "System UI isn't responding" over the app (2026-10-04, seen
# on the first run). It is the emulator settling, not the app: answer "Wait" and read again.
dump() {
  local ui b x y
  ui=$(raw_dump)
  if printf '%s' "$ui" | grep -q "isn't responding"; then
    b=$(printf '%s' "$ui" | grep -o 'text="Wait"[^>]*bounds="\[[0-9]*,[0-9]*\]\[[0-9]*,[0-9]*\]"' | head -1 | grep -o '\[[0-9]*,[0-9]*\]\[[0-9]*,[0-9]*\]')
    if [ -n "$b" ]; then
      x=$(printf '%s' "$b" | tr '][' ' ' | tr ',' ' ' | awk '{print int(($1+$3)/2)}')
      y=$(printf '%s' "$b" | tr '][' ' ' | tr ',' ' ' | awk '{print int(($2+$4)/2)}')
      $ADB shell input tap "$x" "$y"; sleep 2; ui=$(raw_dump)
    fi
  fi
  printf '%s' "$ui"
}
# Bounds "[x1,y1][x2,y2]" of the first node whose text or content-desc is exactly $1.
bounds_of() {
  printf '%s' "$2" | grep -o "\(text\|content-desc\)=\"$1\"[^>]*bounds=\"\[[0-9]*,[0-9]*\]\[[0-9]*,[0-9]*\]\"" \
    | head -1 | grep -o '\[[0-9]*,[0-9]*\]\[[0-9]*,[0-9]*\]'
}
tap_label() {
  local ui b x y
  for i in $(seq 1 20); do
    ui=$(dump); b=$(bounds_of "$1" "$ui")
    if [ -n "$b" ]; then
      x=$(printf '%s' "$b" | tr '][' ' ' | tr ',' ' ' | awk '{print int(($1+$3)/2)}')
      y=$(printf '%s' "$b" | tr '][' ' ' | tr ',' ' ' | awk '{print int(($2+$4)/2)}')
      $ADB shell input tap "$x" "$y"; return 0
    fi
    sleep 1
  done
  return 1
}
# Is the episode title on screen, and in the bottom third (the mini-player)?
mini_player_shows() {
  local ui h
  ui=$1
  h=$($ADB shell wm size | grep -o '[0-9]*x[0-9]*' | tail -1 | cut -dx -f2)
  printf '%s' "$ui" | grep -o "\(text\|content-desc\)=\"[^\"]*$TITLE[^\"]*\"[^>]*bounds=\"\[[0-9]*,[0-9]*\]\[[0-9]*,[0-9]*\]\"" \
    | grep -o '\]\[[0-9]*,[0-9]*\]' | tr -d '[]' | cut -d, -f2 \
    | awk -v h="$h" 'BEGIN{hit=0} { if ($1 > h * 0.7) hit=1 } END{ exit hit ? 0 : 1 }'
}
on_discover() { printf '%s' "$1" | grep -q '\(text\|content-desc\)="Browse"'; }

echo "--> minting a native session for '$IDENTITY'"
tok=""; url="http://127.0.0.1:$PORT/api/app/auth/login?as=$IDENTITY&platform=native"
for i in 1 2 3 4 5; do
  loc=$(curl -s -o /dev/null -D - "$url" | awk 'tolower($1)=="location:"{print $2}' | tr -d '\r')
  case "$loc" in
    closelistening://*) tok=${loc#closelistening://auth\#token=}; break ;;
    http*) url="$loc" ;;
    /*) url="http://127.0.0.1:$PORT$loc" ;;
    *) break ;;
  esac
done
[ -n "$tok" ] || { echo "FAIL: could not mint a native session (is the origin up on :$PORT?)"; exit 1; }

# Clean start: a previous run's session and saved place must not decide this one.
$ADB shell pm clear "$PKG" >/dev/null
$ADB shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
# Signed out: the masthead's "Sign in" link is the proof the web layer has painted, without which
# the callback link is not heard (same precondition as AppSession.java's callback sign-in).
ready=0
for i in $(seq 1 40); do dump | grep -q '"Sign in"' && { ready=1; break; }; sleep 1; done
[ "$ready" = 1 ] || { $ADB exec-out screencap -p > "$OUT/restore-boot.png"; echo "FAIL: app did not paint a signed-out screen"; exit 1; }
$ADB shell am start -a android.intent.action.VIEW -d "'closelistening://auth#token=$tok'" "$PKG" >/dev/null
signed=0
for i in $(seq 1 30); do dump | grep -q 'Notifications' && { signed=1; break; }; sleep 1; done
[ "$signed" = 1 ] || { $ADB exec-out screencap -p > "$OUT/restore-signin.png"; echo "FAIL: sign-in did not complete"; exit 1; }
echo "--> signed in; opening the episode"

$ADB shell am start -a android.intent.action.VIEW -d "'closelistening://episode/$SLUG'" "$PKG" >/dev/null
seen=0
for i in $(seq 1 30); do dump | grep -q "$TITLE" && { seen=1; break; }; sleep 1; done
[ "$seen" = 1 ] || { $ADB exec-out screencap -p > "$OUT/restore-episode.png"; echo "FAIL: episode page did not render"; exit 1; }
sleep 3

tap_label "Discover" || { echo "FAIL: no Discover tab"; exit 1; }
before=""
for i in $(seq 1 20); do before=$(dump); on_discover "$before" && mini_player_shows "$before" && break; sleep 1; done
$ADB exec-out screencap -p > "$OUT/restore-01-before-kill.png"
on_discover "$before" || { echo "FAIL: not on Discover before the kill"; exit 1; }
mini_player_shows "$before" || { echo "FAIL: episode not in the mini-player before the kill"; exit 1; }
echo "--> on Discover with the episode loaded; backgrounding and killing the process"

$ADB shell input keyevent KEYCODE_HOME
sleep 4
$ADB shell am kill "$PKG"
sleep 2
if $ADB shell pidof "$PKG" >/dev/null 2>&1; then echo "FAIL: $PKG survived am kill"; exit 1; fi

echo "--> process gone; cold launch"
$ADB shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
after=""
for i in $(seq 1 40); do after=$(dump); on_discover "$after" && mini_player_shows "$after" && break; sleep 1; done
$ADB exec-out screencap -p > "$OUT/restore-02-after-relaunch.png"
d=0; on_discover "$after" && d=1
m=0; mini_player_shows "$after" && m=1
echo "on_discover=$d mini_player=$m screenshots=$OUT"
[ "$d" = 1 ] && [ "$m" = 1 ] && { echo "COLD_LAUNCH_RESTORE=PASS"; exit 0; }
echo "COLD_LAUNCH_RESTORE=FAIL"; exit 1
