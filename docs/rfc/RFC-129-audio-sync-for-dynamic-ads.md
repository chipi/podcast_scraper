# RFC-129: Audio Sync for Dynamic Ads — a reference from the archive, a map per listener

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 5.5)
- **Stakeholders**: Consumer App (player, iOS, Android, web), Server API, Ops (audio archive)
- **Related RFCs**:
  - `docs/rfc/RFC-126-discover-rails-reasons-paging-and-reels.md` (short listens and Learn stitch
    clips across episodes; they need clip times that are right for the listener)
- **Related Documents**:
  - `docs/recipes/prod-audio-archive.md` (the archive this design reads its reference from)
- **Issue**: #2359

## Abstract

Hosts that insert ads when an episode is fetched shift the audio a listener gets away from the audio
we transcribed. Every time we serve (transcript, insights, Moments, short listens, links with `?t=`)
can then be seconds or minutes off. Measured: the shift differs between two European countries
fetching in the same minute, so no single map made on the server is right for every listener. This
RFC builds the map **per listener, on the device**: the server keeps a fingerprint of the
**original audio from the archive** (the exact bytes we transcribed), and the app lines up small
byte ranges of its own stream against it at the few places where ads can change.

## Problem Statement

Our clip and transcript times are right for the audio we transcribed. Hosts using dynamic ad
insertion (podtrac, megaphone, acast, BBC via Triton, cohst, others) stitch ads into the file per
request. When the ads differ from those in our copy, everything after a changed break plays early or
late.

What a listener sees: a Moment that starts mid-sentence or in the wrong place, a transcript
highlight that runs ahead of or behind the voice, "Play from" landing on an ad.

**Measured, 2026-10-15/16:**

- A length probe over prod episodes found 64 of 402 more than 20 s longer or shorter than what we
  transcribed. Concentrated on BBC (10 of 15 probed), podtrac (9 of 15), acast (8 of 15),
  cohst (8 of 15), megaphone (4 of 15).
- Eleven episodes on six of those hosts, downloaded in the same minute from Germany (the prod
  server) and the Netherlands (a laptop), and lined up against the archived original:

  | Outcome | Episodes |
  |---|---|
  | Same audio as the original (offset 0.00 s throughout) | 5 |
  | Changed since ingest, same in both countries | 1 |
  | Changed, and **different between the two countries** | 3 (both BBC, one podtrac) |
  | No original in the archive to compare with | 2 |

  Example, one podtrac episode: after the mid-roll, the talk is 70.5 s earlier than in our copy in
  the Netherlands and 76.0 s earlier in Germany.
- Two requests a few minutes apart from the same place returned byte-identical files (4 of 4).
- The break points are fixed per episode. Only what fills them changes: every version breaks at
  the same seconds of the original.

**Why a server-side map is not enough:** commit 2b5e96c2a maps each drifted episode from samples of
the stream as the server receives it. On two episodes it moved 18 Moments from 0 to 16 that play what
they say (median offset 13.3 s to 0.6 s). The 2 left are not sync errors: they miss in the same way
on the original audio. But that map is right only for listeners who get the server's ads, and the
measurement above shows they often don't.

## Goals

1. **Right for every listener**, assuming the worst case: each listener can get their own ads.
2. **Never wrong without saying so.** When the app cannot measure, it says so and plays with less,
   rather than silently using a map that may be wrong.
3. **No DGX, no LLM, no re-transcription.** The reference comes from audio we already archived.
4. **Small on the device:** a few hundred KB of extra download per episode, no full second copy.

## Constraints & Assumptions

**Constraints:**

- Audio is never rehosted or proxied (`docs/recipes/prod-audio-archive.md`: the archive is for our
  own reprocessing). A proxy would also send the hosts' ad counting our server's address instead of
  the listener's.
- The browser's cross-origin rules apply on the web. Measured on all 11 test episodes:
  - every server that delivers the audio sends `Access-Control-Allow-Origin: *`;
  - every one except one acast episode advertises `Accept-Ranges: bytes`;
  - the first redirect from BBC (`open.live.bbc.co.uk`) and from cohst (`cohst.app`) sends no
    `Access-Control-Allow-Origin` header, which blocks a read in a browser;
  - BBC delivers over plain http, which an https page cannot read at all.
- The iOS and Android apps download through native code, where those browser rules do not apply.

**Assumptions (to verify in phase 1, see Open Questions):**

- A range request the app makes gets the same ads as its audio element's own request. Supported by
  "same place, minutes apart, byte-identical", not yet shown on a device.
- An episode's breaks sit where the original had ads, or at its start and end.

## Design & Implementation

### 1. Reference, once per episode, on the server

- **Source:** the archived original, only where its provenance says
  `byte_identical_to_transcribed_audio: true`. Measured on prod, 2026-10-16:
  - 2,591 archived files, 156 GiB;
  - about 2,900 provenance records say byte-identical;
  - 120 are later re-downloads (`backfill_refetch`) with different ads, which cannot serve as a
    reference.
- **Fingerprint:** per 20 ms frame, the log energy in 16 frequency bands, made level-free. Plain
  loudness was tried first. It could not place break edges: three thresholds each fixed one case and
  broke another. With 16 bands, the edges in original time agree between Germany and the Netherlands
  to within 0.05 s on 6 of 7 edges measured (0.44 s on the seventh).
- **Checkpoints, not the whole file.** The app does not need the fingerprint of the whole episode,
  only short windows where the offset can change:
  - the start of talk;
  - right after each break: from the server's own comparison of original and current stream, and
    from the ad regions the transcript already marks;
  - one checkpoint every few minutes inside long stretches, to catch a break we did not know about.

  At 16 bands × 50 frames per second × 1 byte, a 10 s window is 8 KB. A one-hour episode with 3
  breaks and checkpoints every 5 minutes is about 15 windows, roughly 120 KB, served with the episode
  on request.
- **Stored** next to the episode's other artifacts. It is a deterministic text/binary artifact, built
  from the archive by a CPU job: no DGX, no model.

### 2. Map, per playback, on the device

1. **Start of talk.** When an episode starts, the app fetches a byte range of its own stream around
   the expected start of talk. It decodes it (Web Audio's `decodeAudioData` takes a chunk of MP3
   frames), builds the same fingerprint, and slides it along the checkpoint to find the offset to
   one frame.
2. **Breaks.** Ahead of the playhead, before each break checkpoint, the same check runs for the next
   stretch. A seek beyond an unchecked break checks that break first. Each check is a range of about
   250 KB at 128 kbit/s (15 s).
3. **Result:** the same structure the player already uses (`audio_time_map`, pairs of transcript
   second and offset, `web/learning-player/src/player/audioTimeMap.ts`). Seeks, `?t=` links,
   transcript follow, insight steps and scrubber marks already go through it.
4. **Reels across episodes** (short listens, Learn): before each clip, one range around the clip's
   start confirms its window in this listener's stream. A clip that falls across a break that
   changed for this listener is skipped.
5. **Byte to time:** a range request asks for bytes, not seconds. For constant-bitrate MP3, bytes =
   seconds × bitrate / 8 after the ID3 header. For variable bitrate, the Xing/VBRI table of contents.
   A first fetch lands near the target, and the fingerprint gives the exact time.

### 3. What the app does when it cannot measure

| State | When | Behaviour |
|---|---|---|
| **Verified** | the device map is built | everything plays, including reels |
| **Estimated** | no device measurement (web on BBC or cohst, a range refused); the server map exists and the stream length matches it | the episode plays with the map; it stays out of stitched reels; the re-sync nudge is visible |
| **Unknown** | neither | as today: transcript time; out of reels; the re-sync nudge is visible |
| **Not needed** | the stream is the original (same length, checkpoint matches at 0) | as today |

### 4. Platforms

- **iOS and Android:** range requests go through native HTTP, so the browser rules don't apply, and
  every host works, BBC included.
- **Web:**
  - Works on hosts whose whole redirect chain allows cross-origin reads: 9 of the 11 tested.
  - cohst's first redirect only forwards to megaphone, which inserts the ads. The server could hand
    the web app the URL after that first redirect, if that redirect turns out not to depend on the
    listener (Open Questions).
  - BBC stays Unknown on the web: the stream is http-only.

### 5. Existing work this builds on

- Commit 2b5e96c2a, not pushed:
  - server `audio_time_map` and `play_start_ms` / `play_end_ms`;
  - the player's map conversion;
  - the Whisper-based server mapper `scripts/ops/audio_retime.py`;
  - `scripts/eval/clip_alignment_check.py --mapped`.

  The server map becomes the **Estimated** fallback. The player's conversion takes the device map
  unchanged.
- The alignment tools from the measurement (coarse alignment by loudness, edges by 16-band spectra)
  move from scratch into `scripts/eval/`, as the reference builder's first version.

## Key Decisions

1. **The device measures; the server only provides the reference.**
   - The worst case is per-listener ads, measured as real (3 of 9 differ between two countries).
2. **The reference is the archived original, not today's stream.**
   - It is exactly the audio our transcript times are measured on, so the map has no error of its
     own. Whisper word times and differences between encodes are no longer part of the result.
3. **Fingerprints, not speech recognition, on the device.**
   - Speech recognition on a phone is heavy and uncertain. Matching numbers is cheap and gave
     one-frame offsets in every talk stretch measured.
4. **A matching window is not enough to call it talk.**
   - One German download's opening ad was the same recording as the original's mid-roll, and matched
     at 0.99. Only long stretches, in order, at the checkpoints count; the transcript's ad regions
     label which stretches are ads.

## Alternatives Considered

1. **Server-side map only (2b5e96c2a).** Simple, and works when the listener gets the server's ads.
   Rejected as the main path: measured location-dependent ads. Kept as the Estimated fallback.
2. **Proxy the audio through our server and measure what passes.** Exact for every listener.
   Rejected: it rehosts audio, and it breaks the hosts' ad counting.
3. **Hide Moments in drifted episodes.** Rejected by the operator.
4. **Speech recognition on the device.** Rejected: model size, battery, and less exact than
   fingerprints.
5. **A fingerprint library (Chromaprint).** Possible later, but it would be a new dependency. The
   16-band fingerprint built here placed the edges consistently, so it is not needed now.

## Testing Strategy

- **No real episodes in fixtures** (real audio is not committed). Instead, build stitched test audio
  from the fixture MP3s already in `tests/fixtures/audio/v3/`: insert synthetic "ads" of different
  lengths at fixed break points. That produces an "original" and several "listener" versions with
  known offsets.
- **Unit tests:**
  - fingerprint;
  - alignment to one frame on the stitched audio;
  - edge placement;
  - rejecting a matching ad;
  - byte-to-time for constant and variable bitrate.
- **Player:** the device sync module against canned range responses; the state table above.
- **Real-data check before rollout:**
  - the 11-episode set from Germany and the Netherlands, run through the reference builder and a
    simulated device;
  - `clip_alignment_check.py` on every Moment of the changed episodes must match the original's
    result.
- **On a device:** iOS and Android, one episode per host, the transcript highlight against the voice
  across a break.

## Rollout & Monitoring

- **Phase 1:**
  - reference builder over the archive (CPU job on the server, read-only on the archive);
  - checkpoints served with the episode;
  - answer the Open Questions on a real device.
- **Phase 2:** device sync in the iOS and Android apps.
- **Phase 3:**
  - web, for hosts that allow it;
  - per-clip confirmation in reels.

**Monitoring:**
- share of playbacks per state (Verified / Estimated / Unknown / Not needed), per host;
- range requests refused;
- the offsets found.

**Success:** on the changed episodes, every Moment that plays what it says on the original also does
on a listener's stream (checked from a second country).

## Open Questions

1. Does a range request from the app get the same ads as its audio element's own request, on iOS,
   Android and web? Hosts tag each request (`request_event_id`, `session_id`).
2. Do the hosts count an extra range request as a download or a listen? If so, how small must it be
   not to?
3. Can a break appear where the original had none? This decides how dense the checkpoints have to
   be.
4. cohst's first redirect: does it depend on the listener? If not, the web app can start after it.
5. Byte-to-time accuracy on the variable-bitrate files we actually see.
6. Episodes without an exact original in the archive (re-downloads, never archived): Estimated only,
   or build a reference from the transcript's word times?
7. One edge differed by 0.44 s between the two countries (e02, start of talk). Not explained.

## References

- Issue #2359 (clip timing on hosts with dynamic ads)
- Commit 2b5e96c2a (server map, player conversion, `scripts/ops/audio_retime.py`)
- `scripts/ops/audio_drift_check.py`, `src/podcast_scraper/server/app_audio_drift.py`
- `web/learning-player/src/player/audioTimeMap.ts`
- `docs/recipes/prod-audio-archive.md`
