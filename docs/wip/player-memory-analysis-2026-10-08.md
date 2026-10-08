# Player memory — where it goes and how to use less (2026-10-08)

Trigger: a beta tester on Android (Pixel 8, app 1.0.2) saw regions of the episode notes render
black — after toggling Key points several times, and while scrolling the related-episodes rail
(only the blurred action buttons stayed visible). The operator's framing: the fix is to use less
memory, not to diagnose low-memory phones.

## Evidence

**Exit reasons (VictoriaLogs `app_exit`, real users only, last 48 h).** Anonymous by design (no
account or device id), so not provably the tester's phone:

| at (UTC) | Amsterdam | platform / version | reason |
|---|---|---|---|
| 2026-10-07 13:24:53 | 15:24 | android 1.0.2 | `low_memory` (+ `other`) |
| 2026-10-07 13:45:13 | 15:45 | android 1.0.2 | `low_memory` |
| 2026-10-07 21:39:45 | 23:39 | android 1.0.2 | `low_memory` |
| 2026-10-08 06:11:04 | 08:11 | android 1.0.2 | `low_memory` |
| 2026-10-08 07:35:51 | 09:35 | ios 1.0.2 | `bg_memory_pressure` (MetricKit) |

A Pixel 8 has 8 GB RAM. Android killing the app for `low_memory` on it points at the app's own
footprint, not a weak phone. GlitchTip has nothing for the black regions — a raster failure throws
no error — and nothing in the player project for 11:00–13:00 UTC on 2026-10-07.

**Artwork on prod (read-only, `deploy@prod-podcast`).**

- 1,318 stored originals, 957 MB; **all 1,318 have a 320px thumbnail**.
- Longest edge: ≤600px 10 · 601–1400 383 · 1401–2000 244 · **2001–3000 677** · >3000 2.
- **Average decoded size of an original: ~19 MB** (w × h × 4). A 320px thumb: ~0.4 MB.
- Episodes: 2,853 — 1,826 with their own stored art, 1,022 using stored feed art, **5 remote-only**.
  Missing thumbnails and remote fallbacks are not the problem.

## Where the originals are used

The server hands out `size=thumb` everywhere except two places, and the client then reuses one of
them in small slots:

| Source | Size | Shown where |
|---|---|---|
| `GET /episodes/{slug}` (`app_episodes.py:293`) | **large (original)** | the player hero, AND — via `summaryFromDetail` / `it.detail` — Home's Continue hero + Jump back in (up to 7), Queue, Recently played, Saved headings, Revisit, board covers, Up next |
| recap (`app_recap_view.py:106`) | **large** | recap card |
| offline download (`downloads.ts` → `episodeArtwork(detail)`) | **large** | the downloads list, offline player, lock screen |
| everything else (lists, rails, search, your week) | thumb | — |

Six tabs are kept alive (`KEEP_ALIVE_TABS`: Home, Search, Library, Profile, Catalog, Browse), so
their images stay decoded while you are elsewhere. Home alone can hold ~8 originals (~150 MB).

## Other contributors

- **Blur layers.** Every artwork tile's action buttons use `backdrop-blur-sm` (`EpisodeActions`
  overlay), and the player's zone D uses `backdrop-blur-md` over the artwork. Each is its own
  composited layer that reads back the pixels beneath it. Photo 1 (only the blurred buttons
  survived) points straight at this on Android.
- **Downloads are fine for audio**: played by `convertFileSrc` from disk, never read into RAM.
  `correctImageExtension` reads a whole image file to check 18 bytes — brief, but wasteful.

## iOS and Android are analysed separately

They do not fail the same way, and one being fine does not clear the other.

- **Android (WebView / Chromium).** Rendering runs in a separate renderer process with GPU raster
  and a tile-memory budget. Under pressure, tiles are not rastered → black regions; the system's
  low-memory killer ends the app (`ApplicationExitInfo REASON_LOW_MEMORY`). Measure with the
  `Pixel_8` AVD: `adb shell dumpsys meminfo app.closelistening.player` plus the WebView sandboxed
  renderer process, per screen; Chrome DevTools (chrome://inspect) for layers and images.
- **iOS (WKWebView).** The WebContent process is killed by jetsam; the page reloads ("starts from
  scratch", #2277) rather than going black. The 1.0.2 `bg_memory_pressure` exit and the 2026-09-18
  watchdog show iOS is not immune — it fails differently. Measure in the simulator with
  `footprint`/`vmmap --summary` on the WebContent process and Safari's Web Inspector Timelines.

## Plan (biggest saving first) — to be measured before and after on BOTH platforms

1. **Episode detail stops shipping the original to cards.** Add a thumb URL to the detail and use it
   wherever detail art fills a card (`summaryFromDetail`, Jump back in, Queue, Recent, Saved,
   Revisit, boards, Up next). Expected: ~19 MB → ~0.4 MB per card.
2. **A medium size for the player hero** (e.g. 800px; ~2.4 MB decoded instead of ~19–36 MB).
   Generated where thumbs are (the writer), plus a one-off deterministic backfill of medium copies
   for the 1,318 existing originals (prod change — needs the operator's OK when we get there).
3. **Offline downloads save the medium, not the original.**
4. **Recap uses thumb/medium.**
5. **No blur on the action buttons over artwork**; the dark plate (`bg-black/55`) keeps contrast.
6. **Fewer kept-alive tabs** (or a cap), so screens you left release their images.
7. **`correctImageExtension` reads only the first bytes.**

Each step lands with a before/after measurement on the Android Pixel_8 emulator and the iOS
simulator, and the debug-info button gives real numbers from testers' phones.

## Result (2026-10-08, same day)

**Reproduced on demand.** Build 1.0.2 from `43fafa450`, prod backend, `Pixel_8` AVD at 8 GB
(`-memory 8192 -gpu host`), signed in as the prod smoke listener. The repro scripts drive the
WebView over CDP and touch through `adb`.

| Run | Episode notes → Key points ×10 → "More like this" opened and scrolled | Rail region |
|---|---|---|
| 1.0.2 | 3 of 3 | **blank** — only the blurred buttons drawn, still blank 10 s later |
| this branch (no blur) | 3 of 3 | drawn |

Same screen, same state, 1.0.2 vs no blur (CDP `LayerTree` + memory-infra dump):

| | 1.0.2 | no blur |
|---|---|---|
| layers drawing content | 73–74 | 18–20 |
| `cc/tile_memory` | 200 MB | 81–99 MB |
| `gpu/shared_images` | 195–220 MB | 91–108 MB |

**Not reproduced on demand:** the whole sheet body going blank (the tester's Key points photo). I
captured it twice by hand on 1.0.2 earlier the same day, with the same symptom — only blurred
buttons drawn — but nine scripted Key points runs on 1.0.2 drew correctly. The fix removes what was
the only thing left drawn in both photos; it is not separately proven for that case.

**Images:** cards built from an episode detail now take `artwork_thumb_url` (320px); the player,
lock screen and offline copy take `size=medium` (≤1024px). On prod the medium copies exist only
after `m0026` runs; until then `medium` falls back to the original, so the player is unchanged and
the cards are fixed by the thumbs that already exist.
