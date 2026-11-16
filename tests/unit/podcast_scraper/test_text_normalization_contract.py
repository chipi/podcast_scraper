"""ADR-139 — text-normalization contract: per-fix isolation + the seam consistency invariant.

Three fixes for narrated-desk / lowercase-turbo speaker naming, each tested in isolation at the
function level, plus the case-invariance invariant: the same content, truecased and lowercased,
produces the same output for the metadata-anchored seams.
"""

from __future__ import annotations

from podcast_scraper.text_normalization import (
    first_names_match,
    normalize_for_match,
    normalize_name_for_match,
)

# --- Foundation ----------------------------------------------------------------------------------


def test_normalize_for_match_folds_case_and_whitespace():
    assert normalize_for_match("  I'm  Rich   GELFOND ") == "i'm rich gelfond"
    # idempotent
    once = normalize_for_match("The  DAILY")
    assert normalize_for_match(once) == once


def test_first_names_match_nickname_initial_and_rejects():
    assert first_names_match("rich", "Richard")
    assert first_names_match("Bob", "Robert")
    assert first_names_match("R", "Robert")  # initial
    assert first_names_match("Katie", "Katherine")
    assert not first_names_match("Eric", "Erica")  # different people
    assert not first_names_match("rich", "Michael")


def test_first_names_match_distinct_names_sharing_a_short_form_do_not_match():
    # review: the nickname table must NOT merge two DIFFERENT formal names that merely share an
    # ambiguous short form. Alexander (m) / Alexandra (f) and Jonathan / John are distinct people;
    # each keeps its own short form but they never match EACH OTHER. (Eric/Erica above is a trivial
    # true-negative because neither is in the table — these exercise a real in-table collision.)
    assert not first_names_match("Alexander", "Alexandra")
    assert not first_names_match("Jonathan", "John")
    # each still matches its own shared short form (ambiguity is acceptable there)
    assert first_names_match("Alexander", "Alex")
    assert first_names_match("Alexandra", "Alex")
    assert first_names_match("Jonathan", "Jon")
    assert first_names_match("John", "Jon")
    # spelling variants of the SAME name are correctly kept matching
    assert first_names_match("Stephen", "Steven")
    assert first_names_match("Katherine", "Catherine")
    # the surname path is safe for the ambiguous shared tokens: the FORMAL names never cross-match
    assert not first_names_match("Patrick", "Patricia")
    assert not first_names_match("Edward", "Theodore")


# --- Fix 2: nickname feeds the ADR-130 canonicalizer ---------------------------------------------


# --- Fix 3: case-blind, metadata-anchored self-introduction --------------------------------------


# --- Fix 1: narrated-desk cue vocabulary ---------------------------------------------------------


# --- The seam consistency invariant (ADR-139) ----------------------------------------------------


def test_name_helpers_are_case_and_quote_blind():
    assert normalize_name_for_match("O'Shaughnessy") == normalize_name_for_match("oshaughnessy")
    assert normalize_name_for_match("Gómez-Bombarelli") == "gomez-bombarelli"


# --- Org wart + comma tolerance + "my name is" (today's smaller fixes) --------------------------


# --- Integration: full labeling flow (diarize -> name -> classify -> census) ---------------------


# --- ADR-140: versioned labeling profile (registry + A/B switch) ---------------------------------
