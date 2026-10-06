# Translation test assets

Not corpus fixtures. `transcripts/v3/` carries the app-validation corpus and is governed by
coherence rules — every episode there needs a transcript, audio, an RTTM and a built corpus entry,
enforced by `tests/unit/test_fixture_docs_match_disk.py` and
`tests/integration/fixtures/test_rttm_groundtruth.py`. Those rules exist because episodes were
once silently dropped from the built corpus.

These two files are inputs to translation tests only, so none of that applies.

| file | what it is |
| --- | --- |
| `p10_e01.es.txt` | The Spanish episode, with anonymous `SPEAKER_NN` labels — the exact state the pipeline sees after diarization and before naming (D-34). 47 turns, 103 sentences, 1,109 words, two sponsor reads. |
| `p01_e01.en.reference.txt` | The English original of the same episode, named. The Spanish above is its translation, so this is the reference a round-trip can be judged against. |

**Why the Spanish one matters beyond these tests.** Every measurement in the multilingual arc was
taken on it — V.6a's three hazards (§5.2), S2.10's 174 timed requests, ADR-157's 2.2 chars/token
floor. It lived only in a session scratchpad until now, so those numbers rested on a file that was
not in the repo.

It will ALSO become `transcripts/v3/p10_e01.txt` — the Spanish counterpart of `p01` — once its
audio can be rendered. That is blocked on a Spanish **male** voice: the two Spanish voices macOS
installs by default (`Monica` es_ES, `Paulina` es_MX) are both female and measured **11 Hz** apart
in median F0, against the English control pair's **98 Hz**, so a diarizer would very likely merge
them into one speaker. See `../FIXTURES_SPEC.md`.
