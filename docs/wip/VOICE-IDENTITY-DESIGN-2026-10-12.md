# Cross-episode voice identity — design and cost (2026-10-12, #2304 Track B step 6)

Design only. Nothing here is built. Every step that costs DGX time or changes prod is the
operator's decision.

## What it would fix

The naming ladder decides each episode on its own text. Several of the known issues that remain
can only be solved by recognising a voice that recurs across episodes of the same feed:

| Problem (NAMING_KNOWN_ISSUES / plan) | Size measured | Why text cannot fix it |
| --- | --- | --- |
| A host's second diarization cluster | ~27 of 172 stated-but-unplaced voices (dev+val) | the name is already on the host's main voice; nothing in the second cluster's text says who it is |
| F3: feeds whose host never says their name | Dwarkesh, Lenny, ChinaTalk, BRAVE, Tyler: no evidence-named host on any served episode | no per-feed text history to lean on |
| Host/guest swaps by the model | 19 swapped of 430 named pairs (dev+val); no text rule fixes more than it breaks (address rule measured and reverted) | both voices say both names |
| Two-host shows where one voice holds both self-introductions (Hard Fork, Curious Cases, Rest Is Politics, Empire in val_v2) | recurring in val_v2 labels | the merged voice is `unknown` by design; the other host is named only by elimination |

None of these counts was measured against voice identity itself. They are the size of the
problems, not the size of the fix.

## What exists already (verified 2026-10-12)

- `pyannote.audio` 4.0.7 (the repo venv). `SpeakerDiarization` returns `DiarizeOutput` with
  `speaker_embeddings`: one centroid per speaker, in `speaker_diarization.labels()` order
  (`pipelines/speaker_diarization.py`, lines 64-75 and 751).
- `pyannote_provider.py` reads only `speaker_diarization` (line 241) and drops the embeddings.
- Nothing per voice is stored beyond labels, segments and the naming diagnostics.

Not verified: the embedding dimension of the community-1 model as we run it; whether prod keeps
episode audio after transcription (memory: audio is bridge-only, never rehosted, so assume it is
not kept); how long diarization alone takes per audio-hour on the DGX.

## Design (three stages, each gated)

1. **Capture (no extra compute).** Store each voice's centroid in the speakers diagnostics
   sidecar when an episode is diarized. Cost: a few KB per episode; no extra model call. Every
   episode the backfill chain reprocesses would carry embeddings from then on.
2. **A per-feed voice bank (no model).** For each feed, the centroids of voices named with
   strong evidence (own self-introduction, publisher label), grouped per person. Built offline from
   the sidecars; read-only.
3. **A new rung, last in the ladder (gated like any rule).** An unnamed voice whose centroid is
   within a threshold of exactly one banked person of the same feed, and clearly further from
   every other banked person, takes that name with `source=voice_identity`. Never across feeds
   (EPIC-HOST-IDENTIFICATION non-goal: no corpus-wide voiceprint). It never overrides a name the
   text gives.

The threshold is chosen on the gold sets (dev to tune, val_v2 scored once), and the rule ships only
with 0 worse on the gates and every changed voice read, as for any rung.

## Cost

| Option | DGX | Coverage | Notes |
| --- | --- | --- | --- |
| A. Capture going forward only | none extra | new and reprocessed episodes only | the backfill chain is already reprocessing; the bank grows as it runs |
| B. Re-diarize the served corpus for embeddings | one diarization pass over every served episode's audio (hours not measured) | all served episodes | needs the audio again (re-download), DGX time, and a migration to write sidecars |
| C. Embed only the episodes the rung would act on | as B, limited to feeds with unnamed or swapped voices | those feeds | smaller, but still a DGX job and a re-download |

My recommendation is A first: it costs nothing extra, and it shows whether the bank separates
voices well before anything is spent on B or C. I have not measured that separation; it is the
first thing stage 2 would show.

## Risks

- **Privacy and scope:** a voiceprint is biometric data. Keep it per feed and inside the sidecar,
  never exposed through the API, and never compared across feeds.
- **Wrong name is worse than none:** a bank seeded from a wrong name spreads it to every
  episode. Seed only from own-voice evidence, and re-check the bank when a name is repaired.
- **Diarizer drift:** a model or setting change alters the embedding space; the bank is valid per
  model version and must be rebuilt after a change.

## Decisions for the operator

1. Capture embeddings going forward (option A): yes or no.
2. Whether B or C is worth DGX time, decided after A shows how well voices separate.
