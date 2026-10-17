#!/bin/bash
# Android: resuming playback while the app is in the background must not crash it (PLAYER-1I,
# 2026-10-17: Android 16, ForegroundServiceStartNotAllowedException out of the BackgroundAudio
# plugin's start(), the process killed).
#
# Usage: background-resume.sh [out-dir]
#
# Android 12+ refuses a foreground start from the background. The paused playback service is
# stopped by the OS about a minute after the app goes to the background ("Stopping service due to
# app idle"), so the resume has to start it again, from the background. This drives exactly that
# through the plugin, over the WebView's DevTools socket (debug builds are inspectable):
#   start + pause in the foreground -> HOME -> wait WAIT_S -> start (refused) -> pause
# and passes when the process survived both background calls, start answered foreground:false,
# and logcat has no FATAL EXCEPTION. WAIT_S=20 exercises the other branch: the paused service
# still running, promoted in place (also refused from the background, also must not crash).
#
# Prereqs: the emulator up and a DEBUG build installed (`make android-app-install` or
# `make android-build` + `adb install -r -d`), API 31+ image. No server needed: it calls the
# plugin directly.
set -u
OUT=${1:-/tmp/lp-android-background-resume}
ADB=${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}
PKG=app.closelistening.player
WAIT_S=${WAIT_S:-75}
PORT=${CDP_PORT:-9233}
mkdir -p "$OUT"
unset NODE_OPTIONS  # see the top of the Makefile

fail() { echo "BACKGROUND_RESUME=FAIL: $*"; exit 1; }

sdk=$($ADB shell getprop ro.build.version.sdk | tr -d '\r')
[ "${sdk:-0}" -ge 31 ] || fail "needs API 31+ (the restriction does not exist below); device is API $sdk"

# Evaluate one JS expression in the app's WebView page; prints the JSON value. Node 22+ has
# fetch and WebSocket built in.
cdp() {
  # shellcheck disable=SC2016  # the ${...} are JavaScript template literals, not shell
  node --input-type=module -e '
    const [port, expr] = process.argv.slice(1)
    const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json()
    const page = pages.find((p) => p.type === "page") ?? pages[0]
    const ws = new WebSocket(page.webSocketDebuggerUrl)
    await new Promise((r) => ws.addEventListener("open", r))
    ws.send(JSON.stringify({ id: 1, method: "Runtime.evaluate",
      params: { expression: expr, awaitPromise: true, returnByValue: true } }))
    const m = await new Promise((r) => ws.addEventListener("message", (e) => r(JSON.parse(e.data))))
    console.log(JSON.stringify(m.result?.result?.value ?? m.result?.exceptionDetails?.text ?? m))
    ws.close()' "$PORT" "$1"
}
attach() {
  local sock
  sock=$($ADB shell cat /proc/net/unix | grep -a webview_devtools_remote | awk '{print $NF}' | head -1 | tr -d '\r')
  [ -n "$sock" ] || fail "no WebView DevTools socket (is this a debug build?)"
  $ADB forward tcp:"$PORT" "localabstract:${sock#@}" >/dev/null
}
P='window.Capacitor.Plugins.BackgroundAudio'

$ADB shell am start -n "$PKG/.MainActivity" >/dev/null 2>&1
sleep 6
attach
pid=$($ADB shell pidof "$PKG" | tr -d '\r')
[ -n "$pid" ] || fail "app not running after launch"
$ADB logcat -c

echo "--> start + pause in the foreground"
fg_start=$(cdp "$P.start()") ; fg_pause=$(cdp "$P.pause()")
echo "    start $fg_start, pause $fg_pause"
echo "--> HOME, then ${WAIT_S}s in the background"
$ADB shell input keyevent KEYCODE_HOME
sleep "$WAIT_S"

# The WebView is paused in the background: the answer arrives when the app is in front again, so
# the call runs in the background and is read after bringing the app back.
echo "--> start from the background"
cdp "$P.start()" >"$OUT/bg_start.json" 2>&1 &
cdp_pid=$!
sleep 5
alive=$($ADB shell pidof "$PKG" | tr -d '\r')
cdp "$P.pause()" >"$OUT/bg_pause.json" 2>&1 &
sleep 3
$ADB shell am start -n "$PKG/.MainActivity" >/dev/null 2>&1
wait "$cdp_pid" 2>/dev/null
sleep 3
$ADB logcat -d >"$OUT/logcat.txt"

bg_start=$(cat "$OUT/bg_start.json")
echo "    background start answered: $bg_start"
grep -E "PlaybackService|Background started FGS|Stopping service due to app idle" "$OUT/logcat.txt" | cut -c1-200 | sed 's/^/    /'
[ "$alive" = "$pid" ] || fail "the process died on the background start (pid $pid -> '${alive:-none}')"
grep -q "FATAL EXCEPTION" "$OUT/logcat.txt" && fail "FATAL EXCEPTION in logcat ($OUT/logcat.txt)"
[ "$($ADB shell pidof "$PKG" | tr -d '\r')" = "$pid" ] || fail "the process died after the background calls"
case "$bg_start" in
  *'"foreground":false'*|*'"foreground":true'*) ;;
  *) fail "the background start did not answer {foreground} (got: $bg_start)" ;;
esac
echo "BACKGROUND_RESUME=PASS (process $pid survived; start answered $bg_start)"
