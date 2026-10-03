"""A DERIVED transcript must never be counted as an episode's transcript.

THE BUG THIS CLOSES, found by review on 2026-10-03. The multilingual branch introduced
``<base>.anon.txt`` — written unconditionally for every diarized episode by
``episode_processor._write_anon_transcript`` — and did not add it to the exclusion lists that
already skip ``.adfree`` and ``.cleaned``. It had **zero** references on ``main``, so every site
that globs ``*.txt`` and filters the known variants started seeing two near-identical texts per
episode.

THE DAMAGE WAS ON THE ENGLISH PATH, which is what made it worth a test rather than a patch.
``speaker_detectors.boilerplate.shingles_from_transcript_files`` abstains below three transcripts
and counts a passage as recurring at ``max(3, len(transcripts) // 2)``:

* the three-transcript abstention was defeated at TWO episodes;
* a passage shared by two new episodes counted four times and cleared the bar, where it previously
  needed three episodes;
* on a mature feed, new episodes' passages carried double weight against older ones.

The effect is bounded — it can only ADD voices to the recurring-script set, and the ``share < 3%``
gate still applies — so it shows up as roster naming for small-share voices on new episodes.
Bounded is not the same as absent, and nothing in the repo would have reported it: the S0.10
English-artifact allow-list pins metadata and manifest key paths, not files on disk.

WHY THIS TEST IS SHAPED AROUND THE SUFFIX REGISTRY rather than the string ``.anon``. Pinning that
one literal would catch the bug that already happened and miss the next one. Every derived suffix
``transcript_resolution`` declares is checked against every site that filters them, so a SIXTH
suffix added tomorrow fails here until each site is taught about it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from podcast_scraper.languages import language_registry
from podcast_scraper.workflow.transcript_resolution import (
    ADFREE_SUFFIX,
    ANON_SUFFIX,
    CLEANED_SUFFIX,
)

pytestmark = pytest.mark.unit

REPO = pathlib.Path(__file__).resolve().parents[3]

#: Every suffix that marks a transcript as DERIVED from another one. Read from the module that
#: declares them, so this cannot drift from the source of truth.
DERIVED_SUFFIXES = (ADFREE_SUFFIX, CLEANED_SUFFIX, ANON_SUFFIX)

#: The places that enumerate transcripts, mapped to the suffixes each one must exclude and why.
#:
#: PER-SITE, NOT ONE GLOBAL RULE, and the first version of this test got that wrong: it demanded
#: every site exclude all three and failed on the acceptance harness, which has never excluded
#: `.adfree` — on this branch or on `main`. That is the harness's own choice about what it counts,
#: and asserting otherwise would have been me changing semantics I had not investigated to satisfy
#: a test I had just written.
FILTER_SITES = {
    "src/podcast_scraper/providers/ml/diarization/pipeline.py": (
        DERIVED_SUFFIXES,
        "the #1188 recurring-script detector — double-counting defeats its three-transcript "
        "abstention and doubles new episodes' weight in the shingle tally",
    ),
    "scripts/audit/transcript_pairing_audit.py": (
        DERIVED_SUFFIXES,
        "Step 0a of the post-deploy runbook (#2082), which exists to produce trustworthy counts "
        "before every other measurement is taken",
    ),
    "scripts/acceptance/run_acceptance_tests.py": (
        (CLEANED_SUFFIX, ANON_SUFFIX),
        "the acceptance harness's transcript tally — it deliberately counts `.adfree`, so only "
        "the near-duplicate variants are asserted here",
    ),
}


class TestEverySiteThatFiltersDerivedTranscriptsKnowsAllOfThem:
    @pytest.mark.parametrize("relpath", sorted(FILTER_SITES))
    def test_the_site_excludes_every_derived_suffix(self, relpath: str) -> None:
        """Asserted against the SOURCE because these are glob filters, not functions to call.

        A site that filters two of the three suffixes is the exact state this bug was in, and it
        reads as deliberate — the list looks complete until you compare it to the registry.
        """
        required, cost = FILTER_SITES[relpath]
        source = (REPO / relpath).read_text(encoding="utf-8")
        missing = [s for s in required if s not in source]
        assert not missing, (
            f"{relpath} filters some derived transcript suffixes but not {missing}. "
            f"Cost of over-counting here: {cost}."
        )

    def test_the_registry_has_not_grown_without_this_test_noticing(self) -> None:
        """Guards the guard.

        ``DERIVED_SUFFIXES`` is imported, so it tracks the module — but only for names this file
        lists. If ``transcript_resolution`` declares a fourth ``*_SUFFIX``, every assertion above
        keeps passing while the new suffix leaks into all three sites. Counting the declarations
        is what makes that fail here instead of in production.
        """
        source = (REPO / "src/podcast_scraper/workflow/transcript_resolution.py").read_text(
            encoding="utf-8"
        )
        declared = set(re.findall(r"^([A-Z_]+_SUFFIX)\s*=", source, re.M))
        known = {"ADFREE_SUFFIX", "CLEANED_SUFFIX", "ANON_SUFFIX"}
        assert declared == known, (
            f"transcript_resolution declares {sorted(declared)}; this test knows {sorted(known)}. "
            "A new derived suffix must be added to DERIVED_SUFFIXES here AND to every site in "
            "FILTER_SITES, or it will be counted as a base transcript."
        )


class TestTheDerivedSuffixesAreActuallyDistinct:
    def test_no_suffix_is_a_substring_of_another(self) -> None:
        """The filters are substring tests, so an overlapping pair would make one of them dead.

        Not hypothetical in shape: ``.adfree.turns`` and ``.adfree`` coexist in this tree already,
        and the rule that keeps substring filtering honest is that no derived MARKER contains
        another.
        """
        for a in DERIVED_SUFFIXES:
            others = [b for b in DERIVED_SUFFIXES if b != a]
            assert not any(a in b for b in others), f"{a} is a substring of another suffix"


class TestTheFixtureCorpusWouldHaveCaughtIt:
    """The regression, reproduced against real filenames rather than constructed ones.

    `tests/fixtures/app-validation-corpus/v3` holds derived transcripts beside their bases, so the
    filter can be exercised on the same shapes the pipeline writes.
    """

    CORPUS = REPO / "tests/fixtures/app-validation-corpus/v3"

    def test_filtering_the_corpus_leaves_one_text_per_episode(self) -> None:
        """The whole rule, applied to real filenames: derived variants AND source-language bodies.

        THE SECOND EXCLUSION WAS FOUND BY THIS TEST FAILING. `p10_e01.es.txt` — the Spanish source
        a translated episode keeps beside its English canonical body (D-44) — carries none of the
        three derived markers, so the detector counted one episode twice, in two languages. Same
        double-counting as `.anon`, reached by a different route.

        Language tags come from the registry rather than a two-letter regex, because only the
        registry knows which dotted tokens are languages: a base transcript whose name ended in one
        would otherwise be dropped from the detector entirely.
        """
        if not self.CORPUS.is_dir():
            pytest.skip(f"fixture corpus missing: {self.CORPUS}")
        paths = sorted(self.CORPUS.glob("feeds/*/**/transcripts/*.txt"))
        assert paths, "no fixture transcripts found; the filter below would be vacuous"

        tags = tuple(f".{code}" for code in language_registry())

        def counted(name: str) -> bool:
            if any(s in name for s in DERIVED_SUFFIXES):
                return False
            stem = name[: -len(".txt")] if name.endswith(".txt") else name
            return not any(stem.endswith(tag) for tag in tags)

        kept = [p.name for p in paths if counted(p.name)]
        assert kept, "the filter removed every transcript"
        duplicates = sorted({k for k in kept if kept.count(k) > 1})
        assert not duplicates, f"the filter kept the same name twice: {duplicates}"

        # One counted text per episode id — the property that actually matters. A variant slipping
        # through shows up as an episode with two, which is precisely how the counts doubled.
        by_episode: dict[str, list[str]] = {}
        for name in kept:
            by_episode.setdefault(name[: -len(".txt")], []).append(name)
        doubled = {ep: names for ep, names in by_episode.items() if len(names) > 1}
        assert not doubled, (
            f"more than one counted transcript per episode: {doubled} — a derived or "
            "source-language variant is being counted as a base transcript"
        )
