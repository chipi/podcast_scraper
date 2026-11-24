---
id: e04_provider_methods_seam
class: light
status: landed
issue: none
touches:
  - src/podcast_scraper/speaker_detectors/factory.py
---

# e04_provider_methods_seam — the speaker detector factory casts the providers it builds

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md), light class.

## 0. Why light

The E04 provider-methods seam (`providers/extensible.py`) stops the provider classes inheriting
their GI and naming methods; they are registered and added when a provider is built, so mypy no
longer sees `detect_speakers` on `OpenAIProvider` and the rest statically. The factory assigns each
provider it builds to a `SpeakerDetector` variable, which mypy then refuses (21 sites).

The change wraps each construction in `cast(SpeakerDetector, ...)`, the same cast the factory
already uses for the preloaded provider (line 169). A cast does nothing at run time: the same class
is built with the same arguments, and each branch's `verify_protocol_compliance` check is
unchanged. No corpus artifact, no file's bytes and no shown value can change, so no scoreboard
number can move.

Evidence: the provider-methods seam's suites ran before this cast with the same runtime code
(unit 16505, integration 3981, non-ML e2e 469 passed; `verify_seams.sh`, `EXIT=0`); the cast is
type-only and `mypy src/podcast_scraper` is the check it satisfies.

Effect on landed outcomes: none beyond any code change (every migration's measured closure covers
the whole package, so its digest moves with any commit).

## Landed

On prod at the latest with the core release (public d12e9acb0, release d4033b3, 2026-11-17); the
news release (085bdc27b, release d8bb835, 2026-11-23) carries it too. Code only: no corpus
change, no prod step.
Status set to landed 2026-11-24 in the record cleanup after the news release (the change gate's
records are open or landed; this one shipped).
