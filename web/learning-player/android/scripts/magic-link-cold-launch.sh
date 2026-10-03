#!/bin/bash
# Android: does a sign-in link that LAUNCHES the app sign the person in? (#2272)
#
# Usage: magic-link-cold-launch.sh '<magic link>' [out-dir]
#
# Driven from the SHELL on purpose. Instrumentation runs inside the app's own process, so a test
# that force-stops the app kills itself (MagicLinkJourneyTests M3 did exactly that, 2026-10-03),
# and `am instrument` starts the app anyway — the cold case cannot be reached from inside it.
#
# Steps: force-stop the app → hand the link to the browser the way Mail does (ACTION_VIEW on the
# http(s) URL) → the browser follows the verify redirect to closelistening://auth#token=… → the
# intent filter LAUNCHES the app → read the screen. Passes when the app is foreground, signed in
# (the masthead's "Notifications" bell), and on neither Profile (a RETURNING account) nor the
# signed-out landing ("Create your free account").
#
# Prereqs: an emulator/device with the debug app installed, and the API reachable at the link's
# host from the device (e.g. `adb reverse tcp:8000 tcp:8000`). The link must be unused.
set -u
LINK=${1:?usage: $0 '<magic link>' [out-dir]}
OUT=${2:-/tmp/lp-android-cold}
ADB=${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}
PKG=app.closelistening.player
mkdir -p "$OUT"

$ADB shell am force-stop "$PKG"
if $ADB shell pidof "$PKG" >/dev/null 2>&1; then echo "FAIL: $PKG is still running"; exit 1; fi
echo "--> app stopped; opening the link in the browser"
$ADB shell am start -a android.intent.action.VIEW -d "'$LINK'" >/dev/null

dump() { $ADB shell uiautomator dump /sdcard/lp-ui.xml >/dev/null 2>&1; $ADB shell cat /sdcard/lp-ui.xml 2>/dev/null; }
foreground=""
for i in $(seq 1 45); do
  foreground=$($ADB shell dumpsys activity activities 2>/dev/null | grep -m1 "mResumedActivity\|topResumedActivity" || true)
  case "$foreground" in *"$PKG"*) break ;; esac
  # A browser "open in app?" prompt, if this browser shows one.
  ui=$(dump)
  for label in "Open" "Continue" "Open app" "Always" "Just once"; do
    if printf '%s' "$ui" | grep -q "text=\"$label\""; then
      echo "--> tapping '$label' in the browser"
      b=$(printf '%s' "$ui" | grep -o "text=\"$label\"[^>]*bounds=\"\[[0-9]*,[0-9]*\]\[[0-9]*,[0-9]*\]\"" | head -1 | grep -o '\[[0-9]*,[0-9]*\]\[[0-9]*,[0-9]*\]')
      x=$(printf '%s' "$b" | tr -d '[]' | tr ',' ' ' | awk '{print int(($1+$3)/2)}'); y=$(printf '%s' "$b" | tr '][' ' ' | tr ',' ' ' | awk '{print int(($2+$4)/2)}')
      [ -n "$x" ] && $ADB shell input tap "$x" "$y"
      break
    fi
  done
  sleep 1
done
case "$foreground" in *"$PKG"*) echo "--> app is foreground (launched by the link)" ;; *)
  $ADB exec-out screencap -p > "$OUT/cold-stuck.png"
  echo "FAIL: the app never came to the foreground: $foreground"; exit 1 ;; esac

# Let the web layer boot, take the token from the launch URL, and refresh /me.
signed=0
for i in $(seq 1 30); do
  ui=$(dump)
  printf '%s' "$ui" | grep -q 'Notifications' && { signed=1; break; }
  sleep 1
done
sleep 2; ui=$(dump)
$ADB exec-out screencap -p > "$OUT/cold-landed.png"
landing=0; printf '%s' "$ui" | grep -q 'Create your free account' && landing=1
profile=0; printf '%s' "$ui" | grep -q 'text="Account"' && printf '%s' "$ui" | grep -q 'text="Stats"' && profile=1
echo "signed_in=$signed on_landing=$landing on_profile=$profile screenshot=$OUT/cold-landed.png"
[ "$signed" = 1 ] && [ "$landing" = 0 ] && [ "$profile" = 0 ] && { echo "COLD_LAUNCH=PASS"; exit 0; }
echo "COLD_LAUNCH=FAIL"; exit 1
