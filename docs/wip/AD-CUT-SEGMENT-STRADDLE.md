# The ad cut drops whole segments: talk sharing a segment with an ad goes with it

Status: accepted loss after the read-coverage fixes (2026-10-16). Trimming the segment was built,
measured and rejected. Measured on the prod snapshot of 2026-10-15.

## What happens

`build_adfree_artifacts` (diarized branch) drops every segment that overlaps an excised range and
re-renders the survivors. The range is sentence-aligned; the segment is an ASR unit. When one
segment holds the end of an ad and the start of real talk, the talk is cut with the ad. New
episodes get this as well as old ones.

## Why trimming the segment did not fix it, and what did

Built: a segment the cut covers in part keeps its text outside the range. Replayed over all 2,684
served episodes and every kept piece read (351 pieces, 31,708 chars, two readers, spot-checked):
~12,500 chars of talk, ~19,200 of sponsor pitch and show promo. Trimming exposed that the RANGES
under-covered a read: mid-episode the cut held only the cue sentences, and dropping the whole
segment was what removed the pitch around them.

The coverage itself is what got fixed (`gi/ad_regions.py`, independent review on the same evidence):

- An opener and its call to action are one read (`_bracket_reads`): they clustered apart whenever
  the pitch between them ran past `CLUSTER_GAP`, the opener alone was skipped as a lone cue, and
  the pitch stayed. Replay: 84 more stretches cut, all read, all sponsor reads (WorkOS, Mercury,
  Plaud, Meter, Sonar, Buildkite, GiveWell…). Cost: about 13 of them take the end of the sentence
  spoken before the opener ("Exactly.", "pays for itself."), one ~250 chars of talk — the straddle
  again.
- A reach back for a pitch stops at the sentence naming the call to action's domain
  (`_cta_brand`): before a show's own plug there is no pitch, and the reach took Lenny's Podcast
  guest introductions and Colombia Calling's talk. Replay: 8 stretches of talk kept, 6 pitches let
  back in where ASR misheard the brand ("Tire" for entire.io, "Alphacets").

With both, segment trimming is still wrong for what remains: the residue below is talk inside the
same ASR segment as a cue sentence, and only a judge of the prose itself can split those.

## The losses m0030 accepts (test cases for that judge)

- Thuan Pham's account of arriving with no English ("We were sponsored by a church", Pragmatic
  Engineer, "Scaling Uber"): the insight is removed with its only quote.
- The Mr Beast chocolate-bar point (Complex Systems, "YouTube economics now").
- Joel Lowenstein, named only in a Lenny's Podcast guest introduction (KG person removed).
- "ENCA Mark Data Survey" (BizNews Daybreak): the news intro shares a segment with the
  biznews.com plug (GI and KG object removed).

The labelled pieces (talk vs pitch, per piece) are the reference set for that judge; they are real
transcript text, so they stay out of the repo.
