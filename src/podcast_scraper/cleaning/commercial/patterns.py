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
            r"|we'?re back(?: to the show)?|okay,? so)",
            re.I,
        ),
        "outro",
        0.5,
        "block_end",
    ),
    SponsorPattern(
        re.compile(r"thanks (?:again )?to (?:our )?(?:friends|partners|sponsor)", re.I),
        "outro",
        0.85,
        "block_end",
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
    "lenny",
}
