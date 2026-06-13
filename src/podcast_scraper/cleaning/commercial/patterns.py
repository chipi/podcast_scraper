"""Categorized sponsor/ad patterns for commercial detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

DEFAULT_CONFIDENCE_THRESHOLD = 0.65


@dataclass(frozen=True)
class SponsorPattern:
    """One detectable sponsor phrase with metadata."""

    pattern: re.Pattern[str]
    category: str
    confidence: float
    boundary_hint: str


SPONSOR_PATTERNS: List[SponsorPattern] = [
    SponsorPattern(
        re.compile(r"this episode is (?:brought to you|sponsored|powered) by", re.I),
        "intro",
        0.9,
        "block_start",
    ),
    SponsorPattern(
        re.compile(r"today'?s (?:episode|show|podcast) is (?:brought|sponsored|supported)", re.I),
        "intro",
        0.9,
        "block_start",
    ),
    SponsorPattern(
        re.compile(
            r"(?:a )?(?:quick )?(?:word|message|shout.?out) from (?:our|today'?s) sponsor",
            re.I,
        ),
        "intro",
        0.85,
        "block_start",
    ),
    SponsorPattern(
        re.compile(r"(?:let me|I want to) tell you about", re.I),
        "intro",
        0.6,
        "block_start",
    ),
    SponsorPattern(
        re.compile(r"before we (?:continue|get back|go on|dive in)", re.I),
        "intro",
        0.5,
        "block_start",
    ),
    SponsorPattern(
        re.compile(r"our sponsors today are", re.I),
        "intro",
        0.85,
        "block_start",
    ),
    SponsorPattern(
        re.compile(r"(?:visit|go to|head to|check out) \S+\.com(?:/\S+)?", re.I),
        "cta",
        0.5,
        "inline",
    ),
    SponsorPattern(
        re.compile(r"use (?:code|promo|coupon) [A-Z0-9]+", re.I),
        "cta",
        0.7,
        "inline",
    ),
    SponsorPattern(
        re.compile(r"(?:free trial|sign up|get started) (?:at|today)", re.I),
        "cta",
        0.5,
        "inline",
    ),
    SponsorPattern(
        re.compile(r"(?:now )?(?:back to|let'?s get back to|returning to) (?:the|our)", re.I),
        "outro",
        0.7,
        "block_end",
    ),
    SponsorPattern(
        re.compile(
            r"(?:welcome back to (?:the|our) (?:show|episode|podcast)"
            r"|we'?re back(?: to the show)?)",
            re.I,
        ),
        "outro",
        0.5,
        "block_end",
    ),
    SponsorPattern(
        re.compile(
            r"(?:thanks(?: again)?|thank you) to (?:our )?(?:friends|partners|sponsor)",
            re.I,
        ),
        "outro",
        0.85,
        "block_end",
    ),
    # --- #986 expansion: native-ad / outro patterns surfaced from the manual
    # run-10 real-prod sample (54 episodes, FT-Unhedged-dominant). The original
    # SPONSOR_PATTERNS set above is template-heavy ("brought to you by"). Real
    # podcasts use host-read native ads + production-credit outros that don't
    # follow that template — #904 Tier 1 measured the gap at 2-6% content
    # recall on real prod. The patterns below close some of that gap without
    # over-fitting to a single show's voice.
    SponsorPattern(
        # "Unhedged is produced by Jake Harper" — production-credit outro
        re.compile(r"\bis produced by\s+[A-Z][\w\-' ]+", re.I),
        "outro",
        0.8,
        "block_end",
    ),
    SponsorPattern(
        # "Our executive producer is Jacob Goldstein"
        re.compile(r"\b(?:our )?executive producer (?:is|are)\b", re.I),
        "outro",
        0.85,
        "block_end",
    ),
    SponsorPattern(
        # "Special thanks to Laura Clark, Alistair Mackey, ..."
        re.compile(r"\bspecial thanks to\b", re.I),
        "outro",
        0.7,
        "block_end",
    ),
    SponsorPattern(
        # "FT Premium subscribers can get the Unhedged newsletter for free"
        re.compile(
            r"\b(?:premium )?subscribers can (?:get|receive|access|read)\b",
            re.I,
        ),
        "cta",
        0.8,
        "inline",
    ),
    SponsorPattern(
        # "a 30-day free trial is available", "30 day free trial"
        re.compile(r"\b(?:\d+[- ]?day(?:s)? )?free trial(?: is available)?\b", re.I),
        "cta",
        0.7,
        "inline",
    ),
    SponsorPattern(
        # Spoken URL: "go to ft.com slash unhedged" — the existing visit/go to
        # pattern only catches the bare URL form ("go to ft.com"), not the
        # full spoken-slash construction that follows.
        re.compile(r"\b\w+\.(?:com|net|org|io) slash\s+\S+", re.I),
        "cta",
        0.7,
        "inline",
    ),
]

BLOCK_START_PATTERNS = [p for p in SPONSOR_PATTERNS if p.boundary_hint == "block_start"]
BLOCK_END_PATTERNS = [p for p in SPONSOR_PATTERNS if p.boundary_hint == "block_end"]

BRAND_NAMES: Set[str] = {
    "stripe",
    "figma",
    "notion",
    "linear",
    "vanta",
    "miro",
    "zapier",
    "hubspot",
    "squarespace",
    "shopify",
    "mailchimp",
    "convertkit",
    "airtable",
    "wix",
    "justworks",
    # NOTE: do not add host/person names here (e.g. "lenny" — the flagship example
    # podcast's host) — a name that appears every episode would give a spurious
    # brand boost to ordinary speech. Keep this list to actual sponsor brands.
}
