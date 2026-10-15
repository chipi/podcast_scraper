# Punctuation repair refusals: what the refused windows really hold (#2368)

Measured 2026-10-15. Input: `scripts/measure/punctuation_refusals.py` run by the prod session at
the deployed sha `f35a446b8` (pipeline-llm image), over every refused window in the newest records:
62 rows, 61 distinct windows, 0 errors. The output holds real text and stays off the repo.

## Verdicts now (same audio, same clip cut, same checks)

| Recorded refusal | Now repaired | Still refused |
| --- | --- | --- |
| `lost_words` (30) | 7 | 22 `lost_words`, 1 `too_many_new_words` |
| `too_many_new_words` (14) | 11 | 3 |
| `repetitive` (11) | 10 | 1 `lost_words` |
| `word_count_changed` (6) | 0 | 6 |
| `still_unpunctuated` (1) | 1 | 0 |

29 of 62 windows repair now, so a re-decode is not deterministic. 8.3 (`repunctuate_only`,
run 38700512700) repaired 9 of 38.

## The hypothesis is killed

"The clip decode drops the end of the audio on the last window" is false. In 20 of the 22
`lost_words` windows the re-decoded text runs to within 0.3 s of the window's end. The two
exceptions stop about 30 s short, and that 30 s is a German ad (below), not missing audio.

## What the refusals are

1. **Mis-timed original text (#2370), 6 episodes: 13 `lost_words` and all 6 `word_count_changed`.**
   These are the episodes whose stored timestamps drifted (Mustafa Suleyman, Taking down the
   cultural elite, China's AI Ecosystem, Understanding equity, Elon Musk, Hebrew Orientalism). The
   words missing from the new text are spread evenly through the window (about 20% in its last
   fifth), because the "old text" the guard compares against is the text of another stretch of
   audio. The refusal is right, and no repair can fix it. These episodes need re-transcribing once
   the #2370 fix (`e7234aaec`) is deployed.
2. **A German ad inserted at the episode's end: 8 windows in 6 episodes** (Speech Recognition,
   Spice Wars, Bizarre Watches x2, Ezra Klein, Who Checks a Proof, Memory Swap). The original pass
   wrote the ad in German (REWE, IONOS, Aldi, Whatnot, Tibber, Ford). The English-forced re-decode
   writes a few words instead ("you.", "See you later.", "Thank you much."), so 50-90 "lost" words
   are ad text. Two tails also hold invented words ("you you you", "6 6 6 7 8 8"). Here the
   refusal blocks a repair whose English content is right, except one Memory Swap decode that is
   gibberish ("Now weiusc Three are theiatbe под ..."), which is refused correctly.
3. **Real speech lost: 1 window** (Fei Fei Li): the a16z disclosures outro, 87 English words, is
   missing from the re-decode. The refusal is right.

## What would change it (not built)

- Class 1 goes away with #2370 and a re-transcription; the guard is not the problem.
- Class 2 is the only evidence that a guard refuses correct repairs. A general rule would be:
  words inside the episode's detected ad intervals do not count as lost. Before building it,
  measure whether those German ads are inside `ad_time_intervals` (geo-inserted ads at the very
  end are often not), and keep the gibberish case refused (the `retained` share alone did not
  separate it: 0.73 vs 0.82-0.87).
- `too_many_new_words` (3 still refused) and the 1 `repetitive` -> `lost_words` window are not
  read yet.
