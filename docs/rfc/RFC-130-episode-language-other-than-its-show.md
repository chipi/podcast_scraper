# RFC-130: An episode in another language than its show

- **Status**: Completed (2026-10-16, not yet deployed)
- **Authors**: Marko, Claude (Opus 5.5)
- **Stakeholders**: Ingestion (transcription), Translation, Naming
- **Related RFCs**:
  - `docs/rfc/RFC-124-multilingual-transcription-and-translation.md` (the translation stage this
    routes into)
- **Issue**: #2375 (the build); measured in #2369

## Abstract

A show declares one language, and every episode inherits it. Some shows publish the odd episode in
another language. Today such an episode is caught after ASR and refused, so it never appears. This
RFC adds one step for the flagged episode only: the ASR identifies the language on three short
clips. If that language is enabled, the episode carries it from there and is translated like any
non-English show. Otherwise it is refused and skipped exactly as today. Episodes in their show's
language pay nothing extra.

## Problem Statement

The show keeps its language setting; that is the operator's direction and stays. But a show's
language is a default, not a guarantee. Ottoman History Podcast declares `en` and publishes a
Turkish-language series, hosted by Can Gümüş. On the production snapshot of 2026-10-15, 2 of 2,684
served episodes are not in their show's language, and both are from this series:
"Soykırımın Bürokrasisi" (2026-05-16) and "Cümle Fener Burada" (2026-09-30). Their titles,
descriptions and transcripts are Turkish. The live feed labels 5 items `Türkçe`.

Nothing in the metadata says so:
- RSS has no standard per-episode language tag, and this feed's items carry none. The publisher's
  `<category>Türkçe</category>` is its own convention.
- Our episode metadata inherits the show's language (`metadata_generation.py`).
- The ASR's reported language is no help: the service echoes the language it is asked for (measured
  2026-10-07: Spanish audio requested as `en` came back as correct Spanish text reporting `en`).

The transcript is the only witness. Since 2026-10-07 the post-ASR guard
(`languages_guard.transcript_language_contradiction`) reads it, and on a contradiction the episode
is **refused and skipped**:
- the refusal is logged at WARNING;
- a `policy` incident (`TranscriptLanguageMismatch`) and a `skipped` ledger row are recorded, and a
  work-list that named the episode counts it as met ("refused and not served"): refusing it is the
  guard working (#2375; until 2026-10-17 it was a hard incident and a failed episode, so a reprocess
  of only refused episodes ran red);
- `<corpus>/.language_refusals/<sha1(episode id)>.json` stores `{requested, detail, title}`;
- nothing else is written, so the episode is on no serving surface;
- later runs skip it before download until the declared language changes.

Both Turkish episodes predate the guard. Today's guard flags both: 0.0% and 0.7% English function
words, against at least 5% for a correctly declared episode. So on current code they would be
refused. An episode we could translate is silently lost.

**Use cases:**

1. **A supported language in another show:** a Spanish episode in an English show is transcribed,
   identified as Spanish, translated and served.
2. **A language we do not ingest:** a Turkish episode in an English show is identified as Turkish
   and refused as today, with the language recorded for later.
3. **The ordinary episode:** an English episode in an English show is untouched and pays nothing.

## Goals

1. **Cheap for everyone else:** no extra work for an episode in its show's language.
2. **Use what exists:** the existing guard for detection, the existing refusal for anything we
   cannot handle, the existing translation stage for what we can.
3. **Generalise beyond one publisher:** no feed-specific labels or conventions.

## Constraints & Assumptions

- Every episode is in one language. An episode split between two languages is out of scope.
- "Supported" means `enabled` in `config/languages.yaml`: en, es, it, pt, de, fr.
- Identification runs on the DGX ASR, the same model that transcribes; no new dependency.

## Design

### 1. Detect (every episode, no extra cost)

The existing post-ASR guard, unchanged. It reads the whole transcript (at least 200 words) and
flags when the declared language's function words are under 2% of it, or when the text confidently
reads as another language. Measured: a correctly declared language never fell below 5.1%
(Portuguese, the thinnest list); on the snapshot, English episodes run 32-40% on a broader list,
the two Turkish ones 1-2%.

### 2. Identify (flagged episodes only)

- Split the transcript into stretches of about 30 seconds, and keep those that clearly do not read
  as the show's language (under the same 2% floor, at least 30 words). An ad, a quote or a cameo in
  the show's language is never chosen: "Cümle Fener Burada" opens with an English sponsor line.
- Take three of them, spread early, middle and late.
- Cut each clip and send it to the ASR with **no** language, so the service detects one. Whisper
  decides the language from the first 30 seconds of what it is given.
- Accept a language only when all three clips agree.

Cost: three short ASR calls per flagged episode (2 in 2,684 on the snapshot).

Why not the function words alone: they name the language correctly on the 15 v3 fixtures
(es, it, fr, de, pt), but the production lists cover only seven languages (en, de, es, pt, fr, it,
nl), so they cannot name Turkish or any language outside them. The ASR knows about a hundred.

### 3. Route

- **Identified and enabled:** the episode carries that language from here on. In the transcription
  step this is a per-episode copy of the run's config (`language_override`), so diarization and
  naming read it. `asr.json` records it as `language.detected`. Metadata generation reads it, so the
  episode's language is recorded with `language_source: detected`, and translation and the analysis
  gate use it. The transcript in hand is kept: the ASR wrote the spoken language.
- **Not enabled, the clips disagree, or identification fails:** refuse and skip as today. The
  refusal record gains `detected` (the language, when identified), so a later mechanism that admits
  such an episode knows what it is. Admitting it later means paying for ASR again, as the transcript
  is not kept; only the rare refused episode pays that.

### 4. A refused episode is not served, whatever ran before it

A refusal writes nothing new, and it used to leave earlier runs of the same episode in place. The
two Turkish episodes were transcribed before the guard existed, so a refusing reprocess would have
left them served as English. Now:
- **On a language refusal**, every run of that episode in the corpus (not only the newest) is marked
  `unusable` (`language_refused: the audio is 'tr'`). The marker is what the catalog, the player API,
  digests, rails and MCP catalog tools already honour. Every run, because the cumulative catalog
  drops unusable runs before it picks the newest, so an unmarked older run would be served instead.
- **The search indexer withholds unusable runs.** It no longer indexes them, it forgets their
  fingerprint (so a run is embedded afresh if it ever becomes servable again), and at the end of a
  build it deletes every row of an episode that had no servable run in that build. An episode with
  an older unusable run and a newer good one keeps its rows. On the 2026-10-15 snapshot no episode
  was marked unusable, so this changes nothing for the corpus as it is.

## Testing

- **Unit:** clip selection (ads in the show's language skipped, spread, length, too few stretches,
  a show language with no word list); agreement; `transcribe_clip` returning the detected language;
  the guard's two branches through `transcribe_media_to_text` (continues in the detected language;
  refused with `detected` in the record); metadata reading the detected language.
- **DGX probe (operator's go):** the three-clip identification on the 15 v3 fixtures and the 2
  Turkish episodes, every answer read.
- **Replay over the snapshot:** only the 2 episodes change outcome.

## Results (2026-10-16)

- **Identification on the DGX:** 51 of 51 clips named their language correctly, and all 17
  episodes agreed: the 15 v3 fixtures (es, it, fr, de, pt, three each) and both Turkish episodes.
  About 1.3 s per clip. The Turkish audio was the publisher's current file (prod keeps no audio for
  those runs), so the stored timestamps may be slightly off; the clips still landed on Turkish speech.
- **Replay over the snapshot:** the guard, unchanged, judged all 2,685 served transcripts and
  flagged exactly the 2 Turkish episodes. Only a flagged episode can take the new path, so no other
  episode changes. Turkish is not enabled, so both are refused as before, with `detected: tr`.
- **Withdrawal (dry run on the snapshot):** for the 2 Turkish episodes it would mark exactly their 2 runs, nothing else.
- **Tests:** clip selection, agreement, the provider's detected language, both guard branches
  through `transcribe_media_to_text`, the refusal record, the `asr.json` record, metadata and the
  translation config; 14 mutants killed.

## Not covered

- An episode split between two languages.
- A language outside the guard's word lists as the show's language: the guard then has no opinion
  and passes the episode.
- Publisher transcripts in another language: the guard runs on our ASR output.
- An operator surface for refused episodes, and the mechanism that admits one later.
