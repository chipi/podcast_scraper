# flake8: noqa: E501
"""Generate v3 fixture transcripts + ground-truth labels + dataset manifest.

This generator extends the v2 fixture model
(``scripts/eval/data/generate_v2_transcripts.py``) with explicit knobs for the
**failure-mode catalogue** harvested from the autoresearch programme
(`docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` + `docs/wip/PROD_RUN_ANALYSIS_100EP.md`):

* ASR garble class (Whisper-style speaker-name corruption — Bessent → Bessett,
  Weisenthal → Wassenthal, Geithner → Geidner, Hobart → Burne/Byrne Hobart).
* Nickname / variant-formal class (Rich ↔ Richard Clarida).
* Alias inventions (Liam → invented "Liam Verbeek").
* Position-arc episodes (a guest's view evolves across multiple episodes).
* Recurring-guest patterns (cross-episode callbacks).
* Native ad / non-templated sponsor blocks (host-read, no canonical marker).
* Sponsor-shaped real content (enthusiastic recommendations the cleaner must
  NOT strip).
* Low-grounding feed patterns (omnycontent-shape: dialogue-heavy, light on
  distilled claims).
* NER zero-host patterns (NPR-shape: host speaks but the surface form evades
  off-the-shelf NER).
* Multi-accent stress (≥ 2 non-en-US speakers per episode → ASR WER spike).
* Severe Whisper garbles (similarity-< 0.65 surname garbles).
* Frame-topic domain disambiguation (frame:photography vs frame:legal vs
  frame:financial — exercises the topic-cluster predicate's frame-negative
  test live, not just by synthetic injection).

Why a knob-driven generator (not hand-written v3):

* Spec adherence is verifiable from the data, not by re-reading prose.
* Re-runs are deterministic — same spec, same transcripts (each episode seeds
  a per-episode RNG from its podcast+episode id via MD5).
* Each failure mode becomes a structured field; the v3 manifest exercises a
  ``failure_modes`` tag per episode so the coverage test can assert ≥ 1
  episode per mode.

Output:

* ``tests/fixtures/transcripts/v3/*.txt`` — rendered transcripts.
* ``tests/fixtures/v3/ground_truth/*.json`` — per-episode ground-truth labels
  (canonical guest ids, garble↔canonical map, genuine_recommendation spans,
  sponsor blocks, position-arc deltas).
* ``tests/fixtures/v3/manifest.json`` — corpus-level manifest (podcast list,
  per-episode failure-mode tags, expected guest count, etc.).
* ``data/eval/datasets/curated_5feeds_smoke_v3/manifest.yaml`` —
  autoresearch-ready dataset wrapping v3 with per-episode failure_modes tags.
* ``data/eval/datasets/curated_5feeds_smoke_v3.json`` — same dataset in the
  ``data/eval/datasets/*.json`` flat-file shape used by the existing
  autoresearch harness.

Audio: a separate operator PR provides multi-voice TTS for v3. This generator
does NOT emit audio; it exposes ``AUDIO_VOICE_HINTS`` (per-episode accent /
voice manifest) for that PR to consume.

Usage::

    python scripts/build_v3_fixtures.py            # write everything
    python scripts/build_v3_fixtures.py --dry-run  # planning report only
    python scripts/build_v3_fixtures.py --check    # verify deterministic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS_OUT = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "v3"
LABELS_OUT = PROJECT_ROOT / "tests" / "fixtures" / "v3" / "ground_truth"
FIXTURES_V3_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "v3"
DATASET_DIR = PROJECT_ROOT / "data" / "eval" / "datasets" / "curated_5feeds_smoke_v3"
DATASET_FLAT_JSON = PROJECT_ROOT / "data" / "eval" / "datasets" / "curated_5feeds_smoke_v3.json"


def _stable_seed(s: str) -> int:
    """Deterministic 32-bit seed from a string.

    Identical to v2's ``_stable_seed`` — uses MD5 of the UTF-8 bytes, which is
    stable across runs and platforms (Python's ``hash()`` varies with
    ``PYTHONHASHSEED``). Truncates to 32 bits because ``random.Random.seed``
    ignores higher bits.

    Don't widen — the birthday-collision probability across the ~50 v3 seeds
    is ~50² / 2³³ ≈ 3e-7 (effectively zero).
    """
    # usedforsecurity=False: stable seed hash, not crypto. Silences bandit B324
    # + survives FIPS-mode hosts.
    digest = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big")


# ===========================================================================
# Failure-mode tag vocabulary (used by manifest + coverage test).
# Edit here, NOT in the manifest: changing the vocabulary must also update
# the coverage test's expected set.
# ===========================================================================

FAILURE_MODES: tuple[str, ...] = (
    "asr_garble",  # Whisper-style surname/firstname garbles
    "asr_garble_severe",  # similarity-< 0.65 garbles needing LLM escalation
    "nickname_variant",  # Rich ↔ Richard Clarida
    "alias_invention",  # first-name-only → fake-surname callback
    "same_first_distinct",  # two distinct people sharing a first name
    "position_arc_multi",  # position change across ≥ 2 episodes
    "recurring_guest",  # same guest in ≥ 2 episodes
    "native_ad",  # sponsor block without template marker
    "genuine_recommendation",  # sponsor-shaped real content
    "low_grounding_dialogue",  # omnycontent-shape dialogue-heavy
    "zero_host_ner",  # NPR-shape: host evades spaCy NER
    "multi_accent",  # ≥ 2 non-en-US voices
    "frame_topic_cross_domain",  # frame:photography vs frame:legal vs frame:financial
    "high_person_density",  # host + 2 guests + ≥ 2 callbacks
    "long_context_chunk_boundary",  # key claim across default 900-word boundary
    "reliability_burst",  # sustained-load 503 simulation hook
)


# ===========================================================================
# Data classes — extend v2's Guest/Episode/Podcast with v3 knobs.
# ===========================================================================


@dataclass
class GuestV3:
    """Extended v2 Guest with v3 failure-mode knobs.

    * ``garble_variants``: ASR-style surface forms (Whisper outputs). The
      first element is the canonical name; later elements are garbles. The
      generator alternates them across episodes; ground truth records the
      canonical id.
    * ``nickname_variants``: alternative formal/casual forms (Rich ↔
      Richard). Same person — ground truth records both → canonical.
    * ``accent``: voice accent hint (en-US / en-GB / fr-CA / es-MX / etc.) for
      the multi-voice TTS audio PR. Generator embeds the hint in the
      transcript header so downstream tooling can pick it up.
    * ``severe_garble``: a far-off variant where surname similarity drops
      below 0.65 (Joll Wisenthal vs Joe Weisenthal). Distinct from regular
      ``garble_variants`` because it needs LLM-tier escalation to merge.
    * ``alias_invention``: surface form for a first-name-only call that gets
      a Whisper-invented fake surname later (Liam → "Liam Verbeek").
    """

    name: str  # canonical name (first surface form used)
    role: str
    expertise: str
    garble_variants: list[str] = field(default_factory=list)
    nickname_variants: list[str] = field(default_factory=list)
    accent: str = "en-US"
    severe_garble: str | None = None
    alias_invention: str | None = None


@dataclass
class EpisodeV3:
    """Extended v2 Episode with v3 knobs."""

    ep_id: str
    title: str
    primary_guest: str
    primary_topic: str
    secondary_topics: list[str]
    sponsor_brands: list[str]
    talking_points: list[str]
    callbacks: list[str] = field(default_factory=list)
    position_arc: str | None = None
    # v3 knobs ↓
    failure_modes: list[str] = field(default_factory=list)
    # Per-episode "I want this guest to appear in this garble form" override.
    # Maps guest_key → "canonical" | "garble:0" | "garble:1" | "nickname:0" | "severe".
    guest_surface_overrides: dict[str, str] = field(default_factory=dict)
    # Native ad block (host-read, no template marker). Inserted mid-conversation.
    native_ad_block: str | None = None
    # Sponsor-shaped real content (enthusiastic recommendation NOT actually
    # paid). Recorded in ground truth as kind=enthusiastic_recommendation.
    genuine_recommendation: str | None = None
    # Dialogue-heavy filler (omnycontent-shape) — long meandering exchanges
    # that don't carry distilled claims, so grounding drops.
    low_grounding_filler_turns: int = 0
    # Extra callback references (alias / first-name-only) injected into the
    # body. Each tuple is (surface_form, canonical_id) for ground truth.
    extra_alias_callbacks: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PodcastV3:
    pod_id: str
    title: str
    domain: str
    host: str
    guests: dict[str, GuestV3]
    recurring_orgs: list[str]
    episodes: list[EpisodeV3]
    description: str = ""
    # Host-side knobs (NER-evading host name, accent).
    host_accent: str = "en-US"
    zero_host_ner: bool = False  # host name styled to evade off-the-shelf NER
    host_garble: str | None = None


# ===========================================================================
# Shared dialogue building blocks (slim adaptation of v2 vocabulary).
# ===========================================================================

HOST_TRANSITIONS = [
    "Let's start there.",
    "I want to dig into that.",
    "That's a useful framing.",
    "Take me through the decision.",
    "Walk me through a concrete example.",
    "Push back on the conventional wisdom for me.",
    "What does the failure mode look like?",
    "What's the one heuristic you keep coming back to?",
]

GUEST_OPENERS = [
    "Sure.",
    "Yeah, so —",
    "It comes down to this:",
    "Honestly,",
    "Here's how I think about it.",
    "Let me give you the version I actually believe.",
]

GUEST_ELABORATIONS = [
    "And the second-order effect is the part most people miss.",
    "Once you internalize that, the rest of the decisions get easier.",
    "I've watched this pattern hold across three different teams.",
    "The framing matters because it tells you which trade-offs you're making.",
    "If you ignore that, you end up rebuilding it from scratch a year later.",
]

HOST_FOLLOWUPS = [
    "That tracks.",
    "Say more about that.",
    "Where does that break down?",
    "That maps to what I've seen.",
]

# Dialogue-heavy filler blocks used by low_grounding_dialogue. These are
# deliberately *not* attributable claims — they're meandering conversational
# turns, the kind of content that produces "dialogue insights" (Finding 12
# from PROD_RUN_ANALYSIS_100EP.md).
FILLER_HOST_TURNS = [
    "So, like, yeah — that's kind of where I land too.",
    "Right, right. I think about that all the time, honestly.",
    "Mmm. Yeah, it's interesting, isn't it?",
    "And you've felt that too, I think? Yeah, you have.",
    "There's something almost — I don't know. Hard to name.",
    "You know what I mean? Like, that vibe.",
]

FILLER_GUEST_TURNS = [
    "Totally. That's exactly the kind of thing I was getting at.",
    "Yeah. Yeah. I don't know if I'd put it that way exactly, but close.",
    "Hmm, that's a good question. Let me think about that.",
    "It depends, right? Like, on the day, on the mood —",
    "Right? Right. Exactly. That's the thing.",
]


# ===========================================================================
# Sponsor / native-ad templates.
# ===========================================================================

SPONSOR_TEMPLATES_OPENING = [
    "This episode is brought to you by {brand}. {pitch} Get started at {brand_lower}.com/podcast.",
    "Today's episode is sponsored by {brand}. {pitch} Try {brand_lower}.com today.",
]

SPONSOR_TEMPLATES_CLOSING = [
    "Before we wrap, thanks again to our friends at {brand}. {pitch}",
    "And finally, a big thank you to our partners at {brand}. {pitch} Check out {brand_lower}.com.",
]

# Native-ad templates: host-read, no canonical "brought to you by" marker.
# These are the patterns the cleaning detector misses today (#594 finding).
NATIVE_AD_TEMPLATES = [
    "Quick aside — I've actually been using {brand} for the last six months and it's changed how I work. {pitch} If you want to check it out, the link's in the show notes.",
    "One thing I want to mention real fast: {brand} has been incredible for this kind of workflow. {pitch} We have a special deal for our listeners over at {brand_lower}.com.",
    "Before we keep going — head over to our sponsor's website if you want to support the show. {pitch}",
    "Y'know what, I'll just say it: {brand} runs my whole {context}. {pitch} Use the link in show notes.",
]

# Sponsor-shaped real content: enthusiastic host recommendation that LOOKS like
# sponsor copy but isn't. Critical for #905 over-cleaning evidence — cleaner
# must preserve these. Ground truth marks them as enthusiastic_recommendation.
GENUINE_RECOMMENDATION_TEMPLATES = [
    "I'll just say it — I genuinely love {brand}. Not sponsored, not paid, I just think they're doing the right thing for {context}.",
    "Real talk: {brand} is the tool I'd pay double for. Not an ad — they're not paying us. I just think it's the best in class for {context}.",
    "Personal recommendation, not a sponsor message: if you're serious about {context}, look at what {brand} is doing. I don't get a kickback, I just think it's worth your time.",
]

BRAND_PITCHES = {
    "Linear": "Linear is the issue tracker built for speed — keyboard-first.",
    "Stripe": "Stripe makes payments simple, whether you're running a marketplace or a subscription business.",
    "Figma": "Figma brings product, engineering, and design into one shared file.",
    "Notion": "Notion replaces the dozen tools your team is half-using.",
    "Vanta": "Vanta automates SOC 2 and ISO compliance.",
    "Datadog": "Datadog gives unified observability across logs, metrics, and traces.",
    "PagerDuty": "PagerDuty turns noisy alerts into structured incident response.",
    "Sentry": "Sentry catches the errors your tests didn't.",
    "Strava": "Strava is the home for athletes.",
    "Adobe": "Adobe Creative Cloud puts Lightroom, Photoshop, and Premiere on every device.",
    "Peak Design": "Peak Design builds camera bags, straps, and tripods designed by photographers.",
    "Vanguard": "Vanguard pioneered low-cost index funds.",
    "Morningstar": "Morningstar gives you independent fund ratings.",
    "Bloomberg": "Bloomberg Terminal is the data spine that institutional desks run on.",
    "Suunto": "Suunto builds dive computers and outdoor watches.",
    "PADI": "PADI is the global standard for dive education.",
    "GoPro": "GoPro captures the moments words don't reach.",
    "Wealthfront": "Wealthfront automates the boring parts of investing.",
    "Squarespace": "Squarespace makes building a beautiful, branded website actually pleasant.",
}

BRAND_CONTEXT = {
    "Linear": "engineering ops",
    "Stripe": "payments stack",
    "Figma": "design workflow",
    "Notion": "documentation",
    "Vanta": "compliance",
    "Datadog": "observability",
    "PagerDuty": "on-call",
    "Sentry": "error tracking",
    "Strava": "training tracking",
    "Adobe": "post-processing",
    "Peak Design": "field kit",
    "Vanguard": "portfolio",
    "Morningstar": "fund research",
    "Bloomberg": "market data",
    "Suunto": "dive planning",
    "PADI": "instruction",
    "GoPro": "underwater footage",
    "Wealthfront": "automation",
    "Squarespace": "site",
}


def render_pitch(brand: str) -> tuple[str, str]:
    pitch = BRAND_PITCHES.get(brand, f"{brand} solves real problems for real teams.")
    return pitch, brand.lower().replace(" ", "")


# ===========================================================================
# Surface-form resolution for failure-mode-driven guest naming.
# ===========================================================================


def _resolve_guest_surface(guest: GuestV3, mode: str) -> str:
    """Return the guest surface form requested by ``mode``.

    Mode strings:
        canonical    → guest.name
        garble:N     → guest.garble_variants[N]
        nickname:N   → guest.nickname_variants[N]
        severe       → guest.severe_garble (falls back to canonical)
        alias        → guest.alias_invention (falls back to canonical)

    Falls back to canonical if the requested variant is unavailable so a
    misspecified override never crashes the build.
    """
    if mode == "canonical" or not mode:
        return guest.name
    if mode.startswith("garble:"):
        idx = int(mode.split(":", 1)[1])
        if 0 <= idx < len(guest.garble_variants):
            return guest.garble_variants[idx]
        return guest.name
    if mode.startswith("nickname:"):
        idx = int(mode.split(":", 1)[1])
        if 0 <= idx < len(guest.nickname_variants):
            return guest.nickname_variants[idx]
        return guest.name
    if mode == "severe":
        return guest.severe_garble or guest.name
    if mode == "alias":
        return guest.alias_invention or guest.name
    return guest.name


# ===========================================================================
# Episode rendering.
# ===========================================================================


def render_episode(podcast: PodcastV3, ep: EpisodeV3) -> tuple[str, dict]:
    """Render a v3 episode transcript and emit its ground-truth labels.

    Returns (transcript_text, ground_truth_dict). The ground-truth dict
    captures:
        guest_canonical_id : str (the v3-stable id)
        surface_forms      : list of (surface_string, kind) — kind ∈
                             {canonical, garble, nickname, severe, alias}
        sponsor_blocks     : list of (kind, brand, line_no) — kind ∈
                             {template_opening, template_closing, native_ad,
                              enthusiastic_recommendation}
        position_arc       : str | None
        callbacks          : list of (surface_form, canonical_id)
        failure_modes      : list[str] (same as ep.failure_modes for round-trip)
    """
    rng = random.Random(_stable_seed(f"v3:{podcast.pod_id}:{ep.ep_id}"))
    host = podcast.host
    guest = podcast.guests[ep.primary_guest]

    # Pick guest surface form for this episode (override > canonical).
    mode = ep.guest_surface_overrides.get(ep.primary_guest, "canonical")
    guest_name = _resolve_guest_surface(guest, mode)

    # Brands: opening / mid-roll / closing.
    opening_brand = ep.sponsor_brands[0] if ep.sponsor_brands else "Linear"
    midroll_brand = ep.sponsor_brands[1] if len(ep.sponsor_brands) > 1 else opening_brand
    closing_brand = ep.sponsor_brands[2] if len(ep.sponsor_brands) > 2 else opening_brand

    surface_forms: list[dict] = []

    def _record_surface(form: str, canonical_id: str, kind: str) -> None:
        surface_forms.append({"surface": form, "canonical_id": canonical_id, "kind": kind})

    # Record the chosen surface form (canonical or variant).
    surface_kind = mode.split(":", 1)[0] if mode != "canonical" else "canonical"
    _record_surface(guest_name, f"{podcast.pod_id}:{ep.primary_guest}", surface_kind)

    lines: list[str] = []
    lines.append(f"# {podcast.title} — Episode")
    lines.append(f"## {ep.title}")
    lines.append(f"Host: {host}")
    lines.append(f"Guest: {guest_name}")
    if ep.failure_modes:
        # Failure-mode hints embedded as a comment line so downstream tooling
        # (audio PR, eval scoring) can find them without reading the manifest.
        # Format prefixed with "#fixture-v3:" so the cleaning pipeline can
        # filter them with a single regex without touching real content.
        lines.append(f"#fixture-v3: failure_modes={','.join(ep.failure_modes)}")
        lines.append(f"#fixture-v3: voice={guest.accent} host_voice={podcast.host_accent}")
    lines.append("")
    lines.append("[00:00]")
    lines.append(
        f"{host}: Welcome back to {podcast.title}. "
        f"Today we're talking about {ep.primary_topic.replace('topic:', '').replace('-', ' ')}, "
        f"and I'm joined by {guest_name}, {guest.role}. {guest_name}, thanks for being here."
    )
    if ep.ep_id == "e01":
        lines.append(
            f"{guest_name}: Thanks, {host}. Excited for this one — "
            f"{guest.expertise} is something I've been thinking about a lot."
        )
    else:
        lines.append(f"{guest_name}: Thanks, {host}. Great to be back.")

    # Opening ad (template).
    sponsor_blocks: list[dict] = []
    lines.append("")
    pitch, brand_lower = render_pitch(opening_brand)
    open_template = rng.choice(SPONSOR_TEMPLATES_OPENING)
    open_line = (
        f"{host}: {open_template.format(brand=opening_brand, brand_lower=brand_lower, pitch=pitch)}"
    )
    lines.append(open_line)
    sponsor_blocks.append(
        {"kind": "template_opening", "brand": opening_brand, "line_index": len(lines) - 1}
    )

    # Callbacks (CIL recurrence signal) — each callback may carry an alias /
    # extra surface form.
    if ep.callbacks:
        lines.append("")
        for callback in ep.callbacks:
            lines.append(f"{host}: Before we get into it — {callback}")
            lines.append(
                f"{guest_name}: Yeah, exactly. And it ties into what we're covering today."
            )

    # Extra alias callbacks (first-name-only mentions, invented surnames, etc.)
    for surface, canonical_id in ep.extra_alias_callbacks:
        lines.append(
            f"{host}: I want to flag something — {surface} mentioned this same dynamic a couple episodes back."
        )
        kind = "alias" if "invent" in canonical_id else "first_name_only"
        _record_surface(surface, canonical_id, kind)

    # Position arc (single-episode statement; multi-episode is composed at the
    # podcast level by setting position_arc on multiple episodes).
    if ep.position_arc:
        lines.append(f"{guest_name}: {ep.position_arc}")
        lines.append(f"{host}: That's a real shift. Worth pulling on.")

    # Genuine recommendation (sponsor-shaped real content). Critical for #905
    # — cleaner must NOT strip these.
    if ep.genuine_recommendation:
        rec_brand = ep.genuine_recommendation
        rec_pitch, _ = render_pitch(rec_brand)
        rec_template = rng.choice(GENUINE_RECOMMENDATION_TEMPLATES)
        rec_context = BRAND_CONTEXT.get(rec_brand, podcast.domain.lower())
        rec_line = (
            f"{host}: {rec_template.format(brand=rec_brand, pitch=rec_pitch, context=rec_context)}"
        )
        lines.append("")
        lines.append(rec_line)
        sponsor_blocks.append(
            {
                "kind": "enthusiastic_recommendation",
                "brand": rec_brand,
                "line_index": len(lines) - 1,
                "note": "Real host recommendation — NOT a paid sponsor. Cleaner must preserve.",
            }
        )

    lines.append("")
    lines.append("[03:30]")

    primary_human = ep.primary_topic.replace("topic:", "").replace("-", " ")
    secondary_humans = [t.replace("topic:", "").replace("-", " ") for t in ep.secondary_topics]

    def _render_pass(claims_subset: Iterable[str], pass_label: str) -> None:
        expansion_templates = [
            "And when teams adopt {org} in that flow, the trade-off becomes visible — {elab}",
            "{org} is one of the platforms where this shows up cleanly. {elab}",
            "Outside of {org}-class tooling, the failure mode is harder to spot. {elab}",
        ]
        for i, claim in enumerate(claims_subset):
            prompts = [
                f"What's the most underrated piece of {primary_human}?",
                (
                    f"Where do {secondary_humans[0]} and {primary_human} intersect?"
                    if secondary_humans
                    else f"How do you frame {primary_human}?"
                ),
                "What's something you used to believe about this that you've revised?",
                (
                    f"How does {rng.choice(podcast.recurring_orgs)} fit into this picture?"
                    if podcast.recurring_orgs
                    else "Where does this break down?"
                ),
                f"What's the {pass_label} angle most listeners don't have access to?",
                "What's the version of the question that actually matters?",
            ]
            lines.append(f"{host}: {rng.choice(HOST_TRANSITIONS)} {rng.choice(prompts)}")
            opener = rng.choice(GUEST_OPENERS)
            elaboration = rng.choice(GUEST_ELABORATIONS)
            supporting = ""
            if i % 2 == 0 and podcast.recurring_orgs:
                org = rng.choice(podcast.recurring_orgs)
                supporting = f"We've seen this play out at teams using {org}."
            lines.append(
                f"{guest_name}: {opener} {claim} {supporting} {elaboration}".strip().replace(
                    "  ", " "
                )
            )
            if i % 3 == 2:
                lines.append(f"{host}: {rng.choice(HOST_FOLLOWUPS)}")
                org = rng.choice(podcast.recurring_orgs) if podcast.recurring_orgs else "the team"
                elab = rng.choice(GUEST_ELABORATIONS).rstrip(".") + "."
                template = rng.choice(expansion_templates)
                lines.append(f"{guest_name}: {template.format(org=org, elab=elab)}")

    _render_pass(ep.talking_points, "structural")

    # Native-ad block (host-read, non-templated). Inserted between passes.
    if ep.native_ad_block:
        nad_brand = ep.native_ad_block
        nad_pitch, nad_lower = render_pitch(nad_brand)
        nad_template = rng.choice(NATIVE_AD_TEMPLATES)
        nad_context = BRAND_CONTEXT.get(nad_brand, "workflow")
        nad_line = f"{host}: {nad_template.format(brand=nad_brand, brand_lower=nad_lower, pitch=nad_pitch, context=nad_context)}"
        lines.append("")
        lines.append(nad_line)
        sponsor_blocks.append(
            {
                "kind": "native_ad",
                "brand": nad_brand,
                "line_index": len(lines) - 1,
                "note": "Host-read native ad — no canonical 'brought to you by' marker.",
            }
        )

    # Mid-roll (template ad).
    lines.append("")
    lines.append("[12:00]")
    lines.append(f"{host}: We'll be right back after a quick word from our sponsors.")
    mid_pitch, mid_lower = render_pitch(midroll_brand)
    mid_line = (
        f"Ad: Today's episode is sponsored by {midroll_brand}. {mid_pitch} "
        f"Visit {mid_lower}.com/podcast for a free trial."
    )
    lines.append(mid_line)
    sponsor_blocks.append(
        {"kind": "template_midroll", "brand": midroll_brand, "line_index": len(lines) - 1}
    )
    lines.append(f"{host}: Welcome back to the show.")

    lines.append("")
    lines.append("[14:00]")
    _render_pass(ep.talking_points, "operational")

    # Low-grounding filler (omnycontent-shape). Inserted after second pass so
    # it doesn't crowd out claims at extraction time.
    for _ in range(ep.low_grounding_filler_turns):
        lines.append(f"{host}: {rng.choice(FILLER_HOST_TURNS)}")
        lines.append(f"{guest_name}: {rng.choice(FILLER_GUEST_TURNS)}")

    lines.append("")
    lines.append("[22:00]")
    if secondary_humans:
        lines.append(
            f"{host}: Let's pivot — {secondary_humans[-1]} is where I want to spend the last stretch."
        )
        lines.append(
            f"{guest_name}: That's where the conversation gets interesting, because the "
            f"trade-offs around {secondary_humans[-1]} cross-cut everything we've covered."
        )
    _render_pass(ep.talking_points, "contrarian")

    # Wrap-up.
    lines.append("")
    lines.append("[28:00]")
    lines.append(f"{host}: Before we wrap, what's one thing listeners can try this week?")
    lines.append(
        f"{guest_name}: Pick one of the ideas we covered and run it for two weeks. The rest is iteration."
    )

    # Closing ad (template).
    lines.append("")
    pitch, brand_lower = render_pitch(closing_brand)
    close_template = rng.choice(SPONSOR_TEMPLATES_CLOSING)
    close_line = f"{host}: {close_template.format(brand=closing_brand, brand_lower=brand_lower, pitch=pitch)}"
    lines.append(close_line)
    sponsor_blocks.append(
        {"kind": "template_closing", "brand": closing_brand, "line_index": len(lines) - 1}
    )
    lines.append("")
    lines.append(f"{host}: {guest_name}, thanks for the conversation.")
    lines.append(f"{guest_name}: Thanks, {host}.")
    lines.append(f"{host}: That's it for today's episode of {podcast.title}. See you next time.")

    text = "\n".join(lines) + "\n"
    ground_truth = {
        "episode_id": f"{podcast.pod_id}_{ep.ep_id}",
        "podcast_id": podcast.pod_id,
        "primary_guest_canonical_id": f"{podcast.pod_id}:{ep.primary_guest}",
        "primary_guest_canonical_name": guest.name,
        "primary_guest_surface_in_transcript": guest_name,
        "surface_forms": surface_forms,
        "sponsor_blocks": sponsor_blocks,
        "position_arc": ep.position_arc,
        "callbacks": list(ep.callbacks),
        "extra_alias_callbacks": [
            {"surface": s, "canonical_id": c} for s, c in ep.extra_alias_callbacks
        ],
        "failure_modes": list(ep.failure_modes),
        "primary_topic": ep.primary_topic,
        "secondary_topics": list(ep.secondary_topics),
        "host_canonical_name": podcast.host,
        "host_accent": podcast.host_accent,
        "guest_accent": guest.accent,
    }
    return text, ground_truth


# ===========================================================================
# v3 podcast spec — explicit failure-mode coverage.
# Each podcast targets a cluster of failure modes; the union across the corpus
# covers all of FAILURE_MODES.
# ===========================================================================


def build_v3_spec() -> list[PodcastV3]:
    """Construct the v3 podcast specs.

    Coverage map (failure_mode → which podcasts exercise it):

    * asr_garble:                  p01, p02, p03, p04, p05, p07
    * asr_garble_severe:           p02, p07
    * nickname_variant:            p05
    * alias_invention:             p01
    * same_first_distinct:         p03 vs p05 (Marco) + p04 vs p05 (Daniel)
    * position_arc_multi:          p02, p05
    * recurring_guest:             p02 (Priya), p07 (Dr. Fischer)
    * native_ad:                   p01, p03, p06
    * genuine_recommendation:      p02, p04
    * low_grounding_dialogue:      p06 (omnycontent-shape)
    * zero_host_ner:               p08 (NPR-shape)
    * multi_accent:                p04, p07
    * frame_topic_cross_domain:    p04 (photography) + p05 (financial) + p02 (legal)
    * high_person_density:         p04, p07
    * long_context_chunk_boundary: p07
    * reliability_burst:           p06 (burst marker for the harness hook)
    """

    # ----- p01 — Singletrack Sessions (mountain biking) -----
    # Failure modes: asr_garble, alias_invention, native_ad.
    p01 = PodcastV3(
        pod_id="p01",
        title="Singletrack Sessions",
        domain="Mountain biking",
        host="Maya",
        description="Conversations about trail building, riding skills, and the gear that lasts.",
        guests={
            "Liam": GuestV3(
                name="Liam Verbeek",  # canonical full form
                role="trail builder for the Cascadia Alliance",
                expertise="trail building",
                # ASR-style garbles for Liam Verbeek:
                garble_variants=["Liam Verbeck", "Liam Verbeak"],
                # Alias invention: Whisper sometimes hears "Liam" only and a
                # callback in another episode invents a fake surname. Here we
                # set the canonical name as "Liam Verbeek" already, so the
                # alias_invention slot holds "Liam Vandermeer" — the WRONG
                # invented surname. Ground truth marks it as alias_invention
                # → same canonical id.
                alias_invention="Liam Vandermeer",
            ),
            "Sophie": GuestV3(
                name="Sophie Laurent",
                role="enduro racer and coach",
                expertise="enduro skills",
                garble_variants=["Sophie Lorenz", "Sophie Lorent"],
            ),
            "Noah": GuestV3(
                name="Noah Brier",
                role="mechanic at Spoke & Wrench",
                expertise="drivetrain mechanics",
                garble_variants=["Noah Bryer", "Noah Brier-ah"],
            ),
        },
        recurring_orgs=["Cascadia Alliance", "Shimano", "RockShox", "Spoke & Wrench"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="Building Trails That Last",
                primary_guest="Liam",
                primary_topic="topic:trail-building",
                secondary_topics=["topic:soil-erosion", "topic:land-stewardship"],
                sponsor_brands=["Strava", "Stripe", "Linear"],
                talking_points=[
                    "Drainage is the single highest-leverage design choice on any trail.",
                    "The most common mistake new builders make is grading too aggressively.",
                    "Land-stewardship conversations have to happen before the first dig.",
                    "Bench cuts on hillsides aren't optional. The shortcut on day one is the washout on year three.",
                    "Soil structure matters more than people think.",
                    "I changed my mind about machine-built versus hand-built trails.",
                ],
                failure_modes=["asr_garble"],
                # Liam appears as canonical in e01, then as a garble in e02 and
                # the alias_invention in e03.
                guest_surface_overrides={"Liam": "canonical"},
            ),
            EpisodeV3(
                ep_id="e02",
                title="Enduro Skills Without the Hype",
                primary_guest="Sophie",
                primary_topic="topic:enduro-racing",
                secondary_topics=["topic:soil-erosion", "topic:risk-management"],
                sponsor_brands=["Linear", "Notion", "Vanta"],
                talking_points=[
                    "Speed comes from braking earlier and smoother, not from taking bigger risks.",
                    "Tire pressure and casing choice often matter more than a minor suspension tweak.",
                    "Risk management in racing isn't avoiding crashes — it's choosing which crashes are survivable.",
                    "Erosion patterns on race-tracks change line choice from practice to race day.",
                ],
                callbacks=[
                    "Liam Verbeck was on the show talking about how drainage is the single highest-leverage design choice."  # garble of Liam Verbeek
                ],
                extra_alias_callbacks=[
                    ("Liam Verbeck", "p01:Liam"),  # garble in callback
                ],
                failure_modes=["asr_garble"],
                guest_surface_overrides={"Sophie": "garble:0"},  # Sophie Lorenz
                native_ad_block="Strava",  # host-read native ad
            ),
            EpisodeV3(
                ep_id="e03",
                title="The Mechanics of a Quiet, Fast Bike",
                primary_guest="Noah",
                primary_topic="topic:drivetrain-mechanics",
                secondary_topics=["topic:maintenance", "topic:risk-management"],
                sponsor_brands=["Notion", "Stripe", "Linear"],
                talking_points=[
                    "Most mystery creaks are solved by cleaning contact surfaces and re-torquing to spec.",
                    "A bike should feel quiet and predictable so your attention stays on the trail.",
                    "Chain wear is the single best leading indicator of how soon you'll be replacing cassettes.",
                    "Suspension service intervals are a risk-management decision.",
                ],
                callbacks=[
                    "Sophie Lorent was on last episode talking about tire pressure mattering more than suspension tweaks."  # garble of Sophie Laurent
                ],
                extra_alias_callbacks=[
                    ("Sophie Lorent", "p01:Sophie"),
                    ("Liam Vandermeer", "p01:Liam"),  # alias_invention
                ],
                failure_modes=["asr_garble", "alias_invention", "native_ad"],
                guest_surface_overrides={"Noah": "garble:0"},  # Noah Bryer
                native_ad_block="Linear",  # second native ad
            ),
        ],
    )

    # ----- p02 — Practical Systems (software engineering) -----
    # Failure modes: asr_garble_severe, recurring_guest, position_arc_multi,
    # genuine_recommendation, frame_topic_cross_domain (legal-frame).
    p02 = PodcastV3(
        pod_id="p02",
        title="Practical Systems",
        domain="Software engineering",
        host="Ethan",
        description="Reliability, architecture, delivery, and the tradeoffs nobody puts in the blog post.",
        guests={
            "Priya": GuestV3(
                name="Priya Nair",
                role="principal SRE at a payments platform",
                expertise="incident response and on-call design",
                garble_variants=["Priya Nayar", "Priya Naar"],
                severe_garble="Priya Naah",  # severe garble (< 0.65 sim)
                accent="en-IN",
            ),
            "Jonas": GuestV3(
                name="Jonas Weisenthal",
                role="staff engineer focused on platform teams",
                expertise="staff-engineer communication",
                # Direct port of the Odd Lots Weisenthal quartet from #853.
                garble_variants=["Jonas Wassenthal", "Jonas Wisenthal"],
                severe_garble="Joll Wisenthal",  # canonical severe garble
            ),
        },
        recurring_orgs=["Linear", "Datadog", "PagerDuty", "Sentry", "Notion"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="On-Call That Doesn't Break People",
                primary_guest="Priya",
                primary_topic="topic:on-call-rotation",
                secondary_topics=[
                    "topic:reliability",
                    "topic:incident-response",
                    "topic:systems-thinking",
                ],
                sponsor_brands=["Linear", "PagerDuty", "Sentry"],
                talking_points=[
                    "A good on-call rotation is designed so that waking up is rare.",
                    "Alerting should be action-oriented.",
                    "Error budgets work when they change behavior.",
                    "PagerDuty's incident response model assumes someone owns the page.",
                ],
                # Earlier position (the "before" of the position arc that
                # resolves in e03). Tagged position_arc_multi so the coverage
                # check sees position change across ≥ 2 episodes for this guest.
                position_arc=(
                    "I'm a believer in centralizing the security function. "
                    "One team, one source of truth, one neck to wring."
                ),
                failure_modes=["asr_garble", "recurring_guest", "position_arc_multi"],
                guest_surface_overrides={"Priya": "canonical"},
            ),
            EpisodeV3(
                ep_id="e02",
                title="Staff-Engineer Communication Patterns",
                primary_guest="Jonas",
                primary_topic="topic:engineering-communication",
                secondary_topics=[
                    "topic:reliability",
                    "topic:incident-postmortems",
                    "topic:risk-management",
                ],
                sponsor_brands=["Notion", "Linear", "Datadog"],
                talking_points=[
                    "A great RFC starts with context and constraints, then options.",
                    "Writing things down turns disagreement into collaboration.",
                    "Postmortems that focus on individuals miss the system that made the mistake easy.",
                    "Architecture decisions are mostly tradeoffs.",
                    "Risk management is a communication problem.",
                ],
                callbacks=[
                    "Priya Nayar was on episode 1 making the case that good on-call rotations are designed so waking up is rare."  # garble of Priya Nair
                ],
                extra_alias_callbacks=[
                    ("Priya Nayar", "p02:Priya"),
                ],
                failure_modes=["asr_garble", "asr_garble_severe"],
                guest_surface_overrides={"Jonas": "severe"},  # Joll Wisenthal
            ),
            EpisodeV3(
                ep_id="e03",
                title="Frame the Decision: Pragmatic Threat Models",
                primary_guest="Priya",
                primary_topic="topic:security-design",
                secondary_topics=[
                    "topic:reliability",
                    "topic:frame",  # legal/decision-making sense of "frame"
                    "topic:risk-management",
                ],
                sponsor_brands=["Vanta", "Sentry", "Linear"],
                talking_points=[
                    "Threat modeling is a framing exercise — frame the decision wrong and you build the wrong defenses.",
                    "Authn/authz separation matters more than people think.",
                    "Security incidents are reliability incidents.",
                    "Vanta-style compliance work and real security work overlap maybe 60%.",
                    "The frame for a security review is 'what could go wrong if this assumption is wrong'.",
                ],
                callbacks=[
                    "I was last on the show talking about on-call rotations. The pattern shows up again here.",
                ],
                position_arc=(
                    "I used to argue for a single central security team. After the 2024 "
                    "webhook incident I changed my mind — embedded ownership beats a "
                    "central gatekeeper every time."
                ),
                failure_modes=[
                    "recurring_guest",
                    "position_arc_multi",
                    "genuine_recommendation",
                    "frame_topic_cross_domain",
                ],
                guest_surface_overrides={
                    "Priya": "nickname:0"
                },  # falls back to canonical (no nicknames defined)
                genuine_recommendation="Sentry",  # honest enthusiastic rec
            ),
        ],
    )

    # ----- p03 — Below the Surface (scuba diving) -----
    # Failure modes: asr_garble (Marco), same_first_distinct (vs p05 Marco), native_ad.
    p03 = PodcastV3(
        pod_id="p03",
        title="Below the Surface",
        domain="Scuba diving",
        host="Rina",
        description="Diving conversations: technique, marine biology, and the calm that distinguishes good divers from lucky ones.",
        guests={
            "Marco": GuestV3(
                name="Marco Silva",  # distinct from p05's Marco Bianchi (same_first_distinct)
                role="technical diver and underwater archaeologist",
                expertise="wreck-diving fundamentals",
                garble_variants=["Marco Sylva", "Marco Silvah"],
                accent="pt-BR",
            ),
            "Hanna": GuestV3(
                name="Hanna Crebo-Rediker",  # double-barreled surname (Heidi Crebo-Rediker pattern from #904)
                role="marine biologist focused on reef systems",
                expertise="marine biology",
                garble_variants=["Hanna Crebo Rediker", "Hanna Krebo-Rediker"],
                severe_garble="Hanna Krebohticker",
                accent="en-GB",
            ),
        },
        recurring_orgs=["PADI", "Suunto", "GoPro", "DAN"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="Wreck Diving Fundamentals",
                primary_guest="Marco",
                primary_topic="topic:wreck-diving",
                secondary_topics=["topic:dive-planning", "topic:soil-erosion"],
                sponsor_brands=["Suunto", "PADI", "GoPro"],
                talking_points=[
                    "Wreck penetration is a planning problem first, a buoyancy problem second.",
                    "Silt-out from poor finning technique kills more divers in wrecks than equipment failures.",
                    "Erosion on a wreck site changes the dive every season.",
                    "DAN's safety stops aren't suggestions.",
                ],
                failure_modes=["asr_garble", "same_first_distinct"],
                guest_surface_overrides={"Marco": "garble:0"},  # Marco Sylva
                native_ad_block="PADI",
            ),
            EpisodeV3(
                ep_id="e02",
                title="Marine Biology for Divers",
                primary_guest="Hanna",
                primary_topic="topic:marine-biology",
                secondary_topics=["topic:reef-conservation", "topic:soil-erosion"],
                sponsor_brands=["PADI", "GoPro", "Suunto"],
                talking_points=[
                    "Coral bleaching events are not random.",
                    "Erosion in coastal areas drives sediment plumes.",
                    "Most reef damage from divers isn't fin contact — it's anchor strikes.",
                    "I changed my mind on shark dives.",
                ],
                callbacks=["Marco Sylva was on last week talking about wreck planning."],  # garble
                extra_alias_callbacks=[
                    ("Marco Sylva", "p03:Marco"),
                ],
                failure_modes=["asr_garble"],
                guest_surface_overrides={"Hanna": "garble:0"},
            ),
            EpisodeV3(
                ep_id="e03",
                title="Severe Surname Garbles: A Marine Biologist's Return",
                primary_guest="Hanna",
                primary_topic="topic:marine-biology",
                secondary_topics=["topic:reef-conservation", "topic:risk-management"],
                sponsor_brands=["Suunto", "GoPro", "PADI"],
                talking_points=[
                    "Reef-conservation policy is a systems problem at planetary scale.",
                    "Most operational dive safety is about not pushing tired teams.",
                    "Risk management on extended dives is mostly about exit planning.",
                ],
                callbacks=[
                    "Marco was on episode 1 walking through wreck planning — the same discipline applies to reef surveys."
                ],
                extra_alias_callbacks=[
                    ("Marco", "p03:Marco"),  # first-name-only callback (alias)
                ],
                failure_modes=["asr_garble_severe", "recurring_guest", "alias_invention"],
                guest_surface_overrides={"Hanna": "severe"},  # "Hanna Krebohticker"
            ),
        ],
    )

    # ----- p04 — Frame & Light (photography) -----
    # Failure modes: multi_accent, frame_topic_cross_domain (photography frame),
    # genuine_recommendation, high_person_density, asr_garble.
    p04 = PodcastV3(
        pod_id="p04",
        title="Frame & Light",
        domain="Photography",
        host="Leo",
        host_accent="en-GB",  # UK host
        description="Working photographers on lighting, location, and getting a frame that holds up at print size.",
        guests={
            "Ava": GuestV3(
                name="Ava Lemoine",
                role="underwater photographer",
                expertise="underwater imaging",
                garble_variants=["Ava Lemoyne", "Ava Lemonne"],
                accent="fr-CA",  # French-Canadian
            ),
            "Tariq": GuestV3(
                name="Tariq Hassan",
                role="documentary photographer",
                expertise="documentary workflow",
                garble_variants=["Tariq Hasaan", "Tarek Hassan"],
                accent="ar-EG",  # Egyptian Arabic
            ),
            # Distinct from p05's investing Daniel — same_first_distinct.
            "Daniel": GuestV3(
                name="Daniel Olufemi",
                role="commercial lighting director",
                expertise="lighting decisions",
                garble_variants=["Daniel Olufemy", "Daniel Olufoemi"],
                accent="en-NG",
            ),
        },
        recurring_orgs=["Adobe", "Peak Design", "Profoto", "Capture One"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="Underwater Images That Feel Alive",
                primary_guest="Ava",
                primary_topic="topic:underwater-photography",
                secondary_topics=["topic:frame", "topic:strobe-lighting"],
                sponsor_brands=["Adobe", "Peak Design", "Squarespace"],
                talking_points=[
                    "An underwater frame is built around backscatter management before composition.",
                    "Strobe positioning at 45 degrees is the starting point, not the destination.",
                    "Adobe Lightroom underwater is half about white balance and half about removing magenta.",
                    "The frame should reward the eye in the first three seconds.",
                ],
                # Multi-accent: host en-GB + guest fr-CA → 2 non-en-US voices.
                failure_modes=["multi_accent", "frame_topic_cross_domain"],
                guest_surface_overrides={"Ava": "canonical"},
                genuine_recommendation="Peak Design",
            ),
            EpisodeV3(
                ep_id="e02",
                title="Documentary Workflow in the Field",
                primary_guest="Tariq",
                primary_topic="topic:documentary-photography",
                secondary_topics=["topic:frame", "topic:editing-workflow"],
                sponsor_brands=["Adobe", "Squarespace", "Peak Design"],
                talking_points=[
                    "Documentary frames carry the most weight when the photographer is closer than feels comfortable.",
                    "Editing a documentary essay is the second project.",
                    "Peak Design slings have changed how I move through a story.",
                    "I changed my mind on color grading documentary work.",
                ],
                callbacks=[
                    "Ava Lemoyne was on last week talking about strobe positioning underwater."
                ],
                extra_alias_callbacks=[
                    ("Ava Lemoyne", "p04:Ava"),
                ],
                failure_modes=["multi_accent", "asr_garble", "high_person_density"],
                guest_surface_overrides={
                    "Tariq": "garble:1"
                },  # Tarek Hassan (transliteration variant)
            ),
            EpisodeV3(
                ep_id="e03",
                title="Lighting Decisions That Save a Shoot",
                primary_guest="Daniel",
                primary_topic="topic:lighting-design",
                secondary_topics=["topic:strobe-lighting", "topic:frame"],
                sponsor_brands=["Adobe", "Peak Design", "Stripe"],
                talking_points=[
                    "On location, the first decision is always: am I shaping the existing light?",
                    "Two-light setups solve 80% of commercial problems.",
                    "Frame and light are inseparable.",
                    "Color temperature mistakes cost more shoots than focus mistakes.",
                ],
                # Frame in photography sense — exercises domain disambiguation
                # against p02 e03's legal "frame the decision" sense and p05's
                # financial "frame the market reaction" sense.
                failure_modes=["frame_topic_cross_domain", "same_first_distinct"],
                guest_surface_overrides={"Daniel": "canonical"},
            ),
            EpisodeV3(
                ep_id="e04",
                title="Roundtable: Three Photographers on Light",
                primary_guest="Ava",
                primary_topic="topic:roundtable-photography",
                secondary_topics=["topic:frame", "topic:strobe-lighting"],
                sponsor_brands=["Adobe", "Peak Design", "Squarespace"],
                talking_points=[
                    "Roundtables reveal where photographers actually disagree.",
                    "Frame and light decisions converge across genres.",
                    "Practical reliability of gear shows up under crowd-shoot conditions.",
                ],
                callbacks=[
                    "Ava walked us through underwater work last month, Tariq covered documentary, Daniel on commercial lighting — today we get all three in the same conversation."
                ],
                extra_alias_callbacks=[
                    ("Ava", "p04:Ava"),  # first-name callback
                    ("Tariq Hasaan", "p04:Tariq"),  # garble in callback
                    ("Daniel Olufemy", "p04:Daniel"),  # garble in callback
                ],
                # 3-accent stress + high person density (host + 3 guests + callbacks).
                failure_modes=[
                    "multi_accent",
                    "high_person_density",
                    "asr_garble",
                    "genuine_recommendation",
                    "recurring_guest",
                ],
                guest_surface_overrides={"Ava": "canonical"},
                genuine_recommendation="Adobe",
            ),
        ],
    )

    # ----- p05 — Long Horizon Notes (investing) -----
    # Failure modes: nickname_variant (Rich/Richard Clarida), asr_garble (Bessent),
    # same_first_distinct (Marco vs p03's Marco, Daniel vs p04's Daniel),
    # position_arc_multi, frame_topic_cross_domain (financial frame).
    p05 = PodcastV3(
        pod_id="p05",
        title="Long Horizon Notes",
        domain="Investing",
        host="Nora",
        description="Long-term investing conversations: index investing, real estate numbers, and the risk you actually face.",
        guests={
            # Distinct from p04 Daniel — same_first_distinct.
            "Daniel": GuestV3(
                name="Daniel Cho",
                role="former bond trader turned index advocate",
                expertise="index investing",
                garble_variants=["Daniel Choh", "Daniel Joh"],
                accent="en-US",
            ),
            # Nickname-variant case: Rich ↔ Richard Clarida.
            "Richard": GuestV3(
                name="Richard Clarida",
                role="former Fed vice chair",
                expertise="monetary policy",
                garble_variants=["Richard Claridah"],
                nickname_variants=["Rich Clarida"],
                accent="en-US",
            ),
            # ASR garble: Bessent → Bessett (canonical #853 case).
            "Scott": GuestV3(
                name="Scott Bessent",
                role="hedge fund manager and policy adviser",
                expertise="macro policy",
                garble_variants=["Scott Bessett", "Scott Bessant"],
                accent="en-US",
            ),
            # Distinct from p03's Marco Silva — same_first_distinct.
            "Marco": GuestV3(
                name="Marco Bianchi",
                role="tax-loss harvesting researcher",
                expertise="tax-loss harvesting",
                garble_variants=["Marco Biancchi", "Marco Bianci"],
                accent="it-IT",
            ),
        },
        recurring_orgs=["Vanguard", "Wealthfront", "Morningstar", "Bloomberg"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="Index Investing Without the Myths",
                primary_guest="Daniel",
                primary_topic="topic:index-investing",
                secondary_topics=[
                    "topic:reliability",
                    "topic:risk-management",
                    "topic:frame",  # financial frame ("frame the market reaction")
                ],
                sponsor_brands=["Vanguard", "Wealthfront", "Stripe"],
                talking_points=[
                    "Index funds are not a strategy — they're the absence of one.",
                    "Vanguard's structure pioneered low costs.",
                    "Frame the market reaction first, then look at fundamentals — most decisions get this backwards.",
                    "Risk management for individual investors is mostly behavioral.",
                    "I changed my mind on small-value tilts.",
                ],
                failure_modes=["same_first_distinct", "frame_topic_cross_domain"],
                guest_surface_overrides={"Daniel": "canonical"},
            ),
            EpisodeV3(
                ep_id="e02",
                title="Nickname Variance: A Fed Vice Chair Reflects",
                primary_guest="Richard",
                primary_topic="topic:monetary-policy",
                secondary_topics=["topic:risk-management", "topic:reliability"],
                sponsor_brands=["Bloomberg", "Vanguard", "Morningstar"],
                talking_points=[
                    "Policy decisions look obvious in hindsight and impossible in real time.",
                    "Forward guidance only works if the market believes you will follow through.",
                    "Risk management at the Fed is mostly about not breaking the institution.",
                    "Reliability of policy signal matters more than the policy itself.",
                ],
                callbacks=[
                    "Daniel Choh was on episode 1 making the case for behavioral discipline."  # garble of Daniel Cho
                ],
                extra_alias_callbacks=[
                    ("Daniel Choh", "p05:Daniel"),
                    ("Rich Clarida", "p05:Richard"),  # nickname of same guest as primary
                ],
                # Use nickname form across this episode (Rich Clarida).
                failure_modes=["nickname_variant", "asr_garble"],
                guest_surface_overrides={"Richard": "nickname:0"},  # "Rich Clarida"
                position_arc=(
                    "Earlier in my career I argued forward guidance was sufficient. "
                    "Over time I revised that — without a credible balance-sheet signal, "
                    "forward guidance alone is just words."
                ),
            ),
            EpisodeV3(
                ep_id="e03",
                title="The Bessent Tape",
                primary_guest="Scott",
                primary_topic="topic:macro-policy",
                secondary_topics=[
                    "topic:risk-management",
                    "topic:reliability",
                    "topic:systems-thinking",
                ],
                sponsor_brands=["Bloomberg", "Wealthfront", "Vanguard"],
                talking_points=[
                    "Macro is mostly about not being on the wrong side of a regime change.",
                    "Risk you can articulate is risk you can manage.",
                    "Reliability of expectations is the underrated piece of macro analysis.",
                    "Systems thinking applies — the dollar is a system, not a price.",
                ],
                callbacks=[
                    "Rich Clarida was on last week — the same point about credibility applies here."
                ],
                extra_alias_callbacks=[
                    ("Rich Clarida", "p05:Richard"),  # nickname recurrence cross-episode
                    (
                        "Marco Biancchi",
                        "p05:Marco",
                    ),  # garble of Marco Bianchi (recurring-guest callback)
                ],
                # Use garble form (Bessett) — exercises asr_garble.
                failure_modes=["asr_garble", "nickname_variant", "same_first_distinct"],
                guest_surface_overrides={"Scott": "garble:0"},  # "Scott Bessett"
                position_arc=(
                    "I used to think currency intervention was always counter-productive. "
                    "After 2023 I changed my mind — selective intervention in thin overnight "
                    "markets can be a legitimate tool."
                ),
            ),
        ],
    )

    # ----- p06 — omnycontent-shape (low_grounding_dialogue + reliability_burst) -----
    # Failure modes: low_grounding_dialogue, native_ad, reliability_burst.
    p06 = PodcastV3(
        pod_id="p06",
        title="The Drift",
        domain="Long-form interviews",
        host="Cam",
        description="Dialogue-heavy, meandering long-form interviews. Low distilled-claim density.",
        guests={
            "Jordan": GuestV3(
                name="Jordan Park",
                role="cultural critic",
                expertise="dialogue and conversational drift",
            ),
        },
        recurring_orgs=["Substack", "Patreon"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="On Drifting (a meandering conversation)",
                primary_guest="Jordan",
                primary_topic="topic:cultural-drift",
                secondary_topics=["topic:dialogue", "topic:long-form"],
                sponsor_brands=["Notion", "Squarespace", "Stripe"],
                talking_points=[
                    "There's something about the way conversations actually unfold that essays can't capture.",
                    "The texture of dialogue is the work.",
                    "Long-form is not a length, it's a posture.",
                ],
                # 8 filler turns → dialogue-heavy, low grounding rate.
                low_grounding_filler_turns=8,
                failure_modes=["low_grounding_dialogue", "native_ad", "reliability_burst"],
                native_ad_block="Notion",
            ),
            EpisodeV3(
                ep_id="e02",
                title="More Drift, Less Signal",
                primary_guest="Jordan",
                primary_topic="topic:long-form",
                secondary_topics=["topic:dialogue", "topic:cultural-drift"],
                sponsor_brands=["Squarespace", "Notion", "Stripe"],
                talking_points=[
                    "Sometimes the point of a conversation is not having a point.",
                    "Dialogue-heavy formats trade grounding for texture.",
                ],
                low_grounding_filler_turns=10,
                failure_modes=["low_grounding_dialogue", "recurring_guest"],
            ),
        ],
    )

    # ----- p07 — long-context + recurring guest + multi-accent + severe garble -----
    # Failure modes: long_context_chunk_boundary, multi_accent, recurring_guest,
    # asr_garble_severe, high_person_density.
    p07 = PodcastV3(
        pod_id="p07",
        title="The Long View — Sustainability",
        domain="Sustainability",
        host="Alex Morgan",
        host_accent="en-AU",  # Australian host
        description="Long-form sustainability conversations with returning experts.",
        guests={
            "Elena": GuestV3(
                name="Dr. Elena Fischer",
                role="sustainability researcher and systems thinker",
                expertise="sustainability and systems thinking",
                garble_variants=["Dr. Elena Fisher", "Dr. Elena Fischner"],
                severe_garble="Dr. Eliana Fishler",
                accent="de-DE",
            ),
            "Skanda": GuestV3(
                # Direct port from #904: Skanda Amarnath ↔ Skanda Eminas severe garble.
                name="Skanda Amarnath",
                role="macro economist",
                expertise="employment and macro",
                garble_variants=["Skanda Amarnauth"],
                severe_garble="Skanda Eminas",
                accent="en-IN",
            ),
        },
        recurring_orgs=["IPCC", "IEA", "Carbon Trust", "Project Drawdown"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="What Sustainability Really Means",
                primary_guest="Elena",
                primary_topic="topic:sustainability",
                secondary_topics=[
                    "topic:systems-thinking",
                    "topic:risk-management",
                    "topic:reliability",
                ],
                sponsor_brands=["Notion", "Linear", "Stripe"],
                talking_points=[
                    "Sustainability is a systems-thinking problem before it's an environmental problem.",
                    "The most expensive thing about climate is the optionality you lose by delaying.",
                    "Risk management at planetary scale looks like reliability engineering at company scale.",
                    "Carbon accounting needs the rigor double-entry bookkeeping brought to finance.",
                ],
                failure_modes=[
                    "multi_accent",
                    "recurring_guest",
                    "long_context_chunk_boundary",
                    "high_person_density",
                ],
                guest_surface_overrides={"Elena": "canonical"},
            ),
            EpisodeV3(
                ep_id="e02",
                title="Severe Garbles: A Macro Discussion",
                primary_guest="Skanda",
                primary_topic="topic:macroeconomics",
                secondary_topics=["topic:systems-thinking", "topic:risk-management"],
                sponsor_brands=["Bloomberg", "Linear", "Stripe"],
                talking_points=[
                    "Employment data is noisier than headlines suggest.",
                    "Systems thinking in macro means watching feedback loops, not point estimates.",
                    "Risk management at the central-bank level looks like reliability engineering.",
                ],
                callbacks=[
                    "Dr. Elena Fisher was on last week talking about systems thinking."  # garble
                ],
                extra_alias_callbacks=[
                    ("Dr. Elena Fisher", "p07:Elena"),  # standard garble
                    ("Dr. Eliana Fishler", "p07:Elena"),  # severe garble in callback
                ],
                failure_modes=["asr_garble_severe", "recurring_guest", "multi_accent"],
                # Use severe garble (Skanda Eminas).
                guest_surface_overrides={"Skanda": "severe"},
            ),
        ],
    )

    # ----- p08 — NPR-shape (zero_host_ner) -----
    # Failure modes: zero_host_ner.
    p08 = PodcastV3(
        pod_id="p08",
        title="Public Hour",
        domain="Public radio",
        host="A. correspondent",  # Stylized host name that evades spaCy NER PERSON detection
        zero_host_ner=True,
        host_accent="en-US",
        description="NPR-shape public-radio format where the host is referred to indirectly.",
        guests={
            "Renee": GuestV3(
                name="Renee Montagne-Park",  # double-barreled, spaced
                role="journalist",
                expertise="public-radio reporting",
                garble_variants=["Renee Montague-Park"],
            ),
        },
        recurring_orgs=["NPR", "BBC"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="Public Radio Format Conventions",
                primary_guest="Renee",
                primary_topic="topic:public-radio",
                secondary_topics=["topic:journalism", "topic:reliability"],
                sponsor_brands=["Squarespace", "Notion", "Stripe"],
                talking_points=[
                    "Public radio voice is a deliberate style, not an accident.",
                    "Reliability of pronunciation matters more than people think.",
                    "Listeners trust the format before they trust the content.",
                ],
                failure_modes=["zero_host_ner"],
            ),
            EpisodeV3(
                ep_id="e02",
                title="Pronunciation as a Reliability Signal",
                primary_guest="Renee",
                primary_topic="topic:public-radio",
                secondary_topics=["topic:journalism", "topic:reliability"],
                sponsor_brands=["Squarespace", "Stripe", "Notion"],
                talking_points=[
                    "Mispronunciation is a high-signal mistake — it tells listeners 'this person isn't paying attention'.",
                    "Reliability of voice is partly preparation, partly humility.",
                ],
                callbacks=[
                    "Renee was on last week walking through public-radio voice conventions.",
                ],
                failure_modes=["zero_host_ner", "recurring_guest"],
                guest_surface_overrides={"Renee": "garble:0"},  # Renee Montague-Park
            ),
        ],
    )

    # ----- p09 — Recurring Guest Web (multi-podcast cross-references) -----
    # Failure modes: recurring_guest, position_arc_multi, asr_garble, genuine_recommendation.
    p09 = PodcastV3(
        pod_id="p09",
        title="Cross-Show",
        domain="Cross-podcast guest appearances",
        host="Sam",
        description="A meta-show where recurring guests from other v3 podcasts revisit positions over multiple episodes.",
        guests={
            "Elena": GuestV3(
                # Same canonical id as p07:Elena but a fresh per-podcast guest
                # entry; ground truth links by canonical_name, not key.
                name="Dr. Elena Fischer",
                role="returning systems thinker",
                expertise="systems thinking",
                garble_variants=["Dr. Elena Fisher", "Dr. Elena Fischner"],
                severe_garble="Dr. Eliana Fishler",
                accent="de-DE",
            ),
            "Skanda": GuestV3(
                name="Skanda Amarnath",
                role="returning macro economist",
                expertise="macro and labor markets",
                garble_variants=["Skanda Amarnauth"],
                severe_garble="Skanda Eminas",
                accent="en-IN",
            ),
        },
        recurring_orgs=["IPCC", "Bloomberg", "BloombergNEF"],
        episodes=[
            EpisodeV3(
                ep_id="e01",
                title="Systems Thinking, Two Years Later",
                primary_guest="Elena",
                primary_topic="topic:systems-thinking",
                secondary_topics=["topic:sustainability", "topic:reliability"],
                sponsor_brands=["Notion", "Linear", "Stripe"],
                talking_points=[
                    "Two years of systems-thinking work has made me more humble about leverage points.",
                    "Reliability and sustainability share more vocabulary than I used to admit.",
                ],
                position_arc=(
                    "In our 2024 conversation I was more optimistic about top-down "
                    "interventions. I've revised that — the work happens at the seams."
                ),
                failure_modes=["recurring_guest", "position_arc_multi", "multi_accent"],
                guest_surface_overrides={"Elena": "canonical"},
            ),
            EpisodeV3(
                ep_id="e02",
                title="Macro Without the Theatrics",
                primary_guest="Skanda",
                primary_topic="topic:macroeconomics",
                secondary_topics=["topic:risk-management", "topic:reliability"],
                sponsor_brands=["Bloomberg", "Linear", "Notion"],
                talking_points=[
                    "Macro forecasting is mostly humility plus a few dependable signals.",
                    "Reliability of data trumps cleverness of model.",
                ],
                callbacks=[
                    "Dr. Elena Fisher was on last week revisiting her 2024 position."  # garble + cross-ep position arc reference
                ],
                extra_alias_callbacks=[
                    ("Dr. Elena Fisher", "p09:Elena"),
                ],
                failure_modes=["asr_garble", "recurring_guest", "multi_accent"],
                guest_surface_overrides={"Skanda": "garble:0"},  # Skanda Amarnauth
                genuine_recommendation="Bloomberg",
            ),
            EpisodeV3(
                ep_id="e03",
                title="When the Severe Garble Bites",
                primary_guest="Skanda",
                primary_topic="topic:labor-markets",
                secondary_topics=["topic:risk-management", "topic:systems-thinking"],
                sponsor_brands=["Bloomberg", "Linear", "Stripe"],
                talking_points=[
                    "Labor markets are where macro stops being abstract.",
                    "Risk is a household-level variable, not a portfolio-level one.",
                ],
                callbacks=[
                    "Skanda Eminas was on episode 2 making this same point on Bloomberg-data caveats."  # severe garble of his own name
                ],
                extra_alias_callbacks=[
                    ("Skanda Eminas", "p09:Skanda"),
                ],
                position_arc=(
                    "Two years ago I argued labor data was reliable at quarterly frequency. "
                    "I now think only annual averages are trustworthy."
                ),
                failure_modes=[
                    "asr_garble_severe",
                    "position_arc_multi",
                    "recurring_guest",
                    "multi_accent",
                ],
                guest_surface_overrides={"Skanda": "canonical"},
            ),
        ],
    )

    return [p01, p02, p03, p04, p05, p06, p07, p08, p09]


# ===========================================================================
# Manifest + dataset emit.
# ===========================================================================


def _audio_voice_hints(podcast: PodcastV3, ep: EpisodeV3) -> dict:
    """Per-episode voice/accent hints for the upcoming multi-voice TTS PR."""
    guest = podcast.guests[ep.primary_guest]
    mode = ep.guest_surface_overrides.get(ep.primary_guest, "canonical")
    surface = _resolve_guest_surface(guest, mode)
    return {
        "host_voice_accent": podcast.host_accent,
        "host_surface": podcast.host,
        "guest_voice_accent": guest.accent,
        "guest_surface": surface,
    }


def emit_corpus(podcasts: list[PodcastV3], *, dry_run: bool = False) -> dict:
    """Render every episode + ground-truth file + manifest.

    Returns a summary dict with episode counts + failure-mode coverage.
    Caller decides whether to print or persist a report.
    """
    summary: dict = {
        "podcast_count": len(podcasts),
        "episode_count": 0,
        "failure_mode_coverage": {mode: 0 for mode in FAILURE_MODES},
        "episodes": [],
    }

    if not dry_run:
        TRANSCRIPTS_OUT.mkdir(parents=True, exist_ok=True)
        LABELS_OUT.mkdir(parents=True, exist_ok=True)

    manifest_episodes: list[dict] = []
    dataset_episodes: list[dict] = []

    for podcast in podcasts:
        for ep in podcast.episodes:
            text, ground_truth = render_episode(podcast, ep)
            episode_id = f"{podcast.pod_id}_{ep.ep_id}"
            transcript_relpath = f"tests/fixtures/transcripts/v3/{episode_id}.txt"

            if not dry_run:
                (TRANSCRIPTS_OUT / f"{episode_id}.txt").write_text(text, encoding="utf-8")
                (LABELS_OUT / f"{episode_id}.json").write_text(
                    json.dumps(ground_truth, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            duration_minutes = round(len(text.split()) / 150.0, 2)
            transcript_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

            manifest_episodes.append(
                {
                    "episode_id": episode_id,
                    "podcast_id": podcast.pod_id,
                    "title": ep.title,
                    "primary_guest_canonical_name": podcast.guests[ep.primary_guest].name,
                    "transcript_path": transcript_relpath,
                    "ground_truth_path": f"tests/fixtures/v3/ground_truth/{episode_id}.json",
                    "transcript_sha256": transcript_sha,
                    "failure_modes": list(ep.failure_modes),
                    "audio_voice_hints": _audio_voice_hints(podcast, ep),
                    "duration_minutes": duration_minutes,
                    "primary_topic": ep.primary_topic,
                    "secondary_topics": list(ep.secondary_topics),
                }
            )

            dataset_episodes.append(
                {
                    "episode_id": episode_id,
                    "title": f"{ep.title} (with {podcast.guests[ep.primary_guest].name})",
                    "transcript_path": transcript_relpath,
                    "transcript_hash": transcript_sha,
                    "preprocessing_profile": "cleaning_v3",
                    "duration_minutes": duration_minutes,
                    "failure_modes": list(ep.failure_modes),
                }
            )

            summary["episode_count"] += 1
            for mode in ep.failure_modes:
                if mode in summary["failure_mode_coverage"]:
                    summary["failure_mode_coverage"][mode] += 1
            summary["episodes"].append(
                {"episode_id": episode_id, "failure_modes": list(ep.failure_modes)}
            )

    # Corpus-level manifest (fixtures side).
    fixtures_manifest = {
        "version": "v3",
        "schema_version": 1,
        "podcast_count": len(podcasts),
        "episode_count": summary["episode_count"],
        "failure_modes_vocabulary": list(FAILURE_MODES),
        "podcasts": [
            {
                "pod_id": p.pod_id,
                "title": p.title,
                "domain": p.domain,
                "host_canonical_name": p.host,
                "host_accent": p.host_accent,
                "zero_host_ner": p.zero_host_ner,
                "guests": [
                    {
                        "key": gkey,
                        "canonical_name": g.name,
                        "garble_variants": list(g.garble_variants),
                        "nickname_variants": list(g.nickname_variants),
                        "severe_garble": g.severe_garble,
                        "alias_invention": g.alias_invention,
                        "accent": g.accent,
                    }
                    for gkey, g in p.guests.items()
                ],
            }
            for p in podcasts
        ],
        "episodes": manifest_episodes,
    }

    # Dataset (autoresearch-ready). Mirrors the v2 smoke shape but adds the
    # failure_modes per-episode tag.
    smoke_subset = [
        e
        for e in dataset_episodes
        if e["episode_id"] in {"p01_e01", "p02_e01", "p03_e01", "p04_e01", "p05_e01"}
    ]
    dataset_smoke = {
        "dataset_id": "curated_5feeds_smoke_v3",
        "version": "1.0",
        "description": (
            "Smoke dataset over v3 sources: first episode per feed (5 episodes). "
            "Per-episode failure_modes tag exposes the v3 knobs that each "
            "episode exercises (asr_garble, native_ad, multi_accent, etc.)."
        ),
        "created_at": "2026-06-09T00:00:00.000000Z",
        "content_regime": "explainer",
        "num_episodes": len(smoke_subset),
        "episodes": smoke_subset,
    }

    if not dry_run:
        (FIXTURES_V3_ROOT / "manifest.json").write_text(
            json.dumps(fixtures_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        # The yaml manifest is intentionally hand-rolled (no PyYAML dep) — its
        # shape is stable; this avoids dragging in PyYAML for a single emit.
        (DATASET_DIR / "manifest.yaml").write_text(
            _render_dataset_yaml(dataset_smoke), encoding="utf-8"
        )
        # Mirror as flat JSON in the same dir for code that reads from the
        # dataset directory.
        (DATASET_DIR / "manifest.json").write_text(
            json.dumps(dataset_smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Also emit the flat-file variant alongside the v1/v2 datasets so the
        # existing autoresearch loader can pick it up by id.
        DATASET_FLAT_JSON.write_text(
            json.dumps(dataset_smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    summary["fixtures_manifest_path"] = str(FIXTURES_V3_ROOT / "manifest.json")
    summary["dataset_yaml_path"] = str(DATASET_DIR / "manifest.yaml")
    summary["dataset_json_path"] = str(DATASET_FLAT_JSON)
    return summary


def _render_dataset_yaml(dataset: dict) -> str:
    """Hand-rolled YAML for the dataset manifest.

    Avoids adding PyYAML as a dependency for a single emit. Format mirrors the
    v2 JSON shape but in YAML, with per-episode ``failure_modes`` lists.
    """
    out: list[str] = []
    out.append(f"dataset_id: {dataset['dataset_id']}")
    out.append(f"version: \"{dataset['version']}\"")
    out.append(f"description: |")
    for line in textwrap.wrap(dataset["description"], width=72):
        out.append(f"  {line}")
    out.append(f"created_at: \"{dataset['created_at']}\"")
    out.append(f"content_regime: {dataset['content_regime']}")
    out.append(f"num_episodes: {dataset['num_episodes']}")
    out.append("episodes:")
    for ep in dataset["episodes"]:
        out.append(f"  - episode_id: {ep['episode_id']}")
        # Title may contain quote/punctuation; wrap in double quotes and escape.
        safe_title = ep["title"].replace("\\", "\\\\").replace('"', '\\"')
        out.append(f'    title: "{safe_title}"')
        out.append(f"    transcript_path: {ep['transcript_path']}")
        out.append(f"    transcript_hash: {ep['transcript_hash']}")
        out.append(f"    preprocessing_profile: {ep['preprocessing_profile']}")
        out.append(f"    duration_minutes: {ep['duration_minutes']}")
        if ep["failure_modes"]:
            out.append("    failure_modes:")
            for mode in ep["failure_modes"]:
                out.append(f"      - {mode}")
        else:
            out.append("    failure_modes: []")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the plan without writing any files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify deterministic regeneration: emit to a temp dir and diff against existing.",
    )
    args = parser.parse_args(argv)

    podcasts = build_v3_spec()

    if args.check:
        # Deterministic-build verification: render twice and confirm bytes match.
        first = emit_corpus(podcasts, dry_run=True)
        second = emit_corpus(podcasts, dry_run=True)
        if first != second:
            print("v3 generator is NOT deterministic — diff in summary across runs")
            return 1
        print("v3 generator deterministic: same spec → same transcripts.")
        print(f"Podcasts: {first['podcast_count']}, Episodes: {first['episode_count']}")
        print("Failure-mode coverage:")
        for mode, count in first["failure_mode_coverage"].items():
            print(f"  {mode}: {count} episode(s)")
        return 0

    summary = emit_corpus(podcasts, dry_run=args.dry_run)
    print(f"v3 podcasts: {summary['podcast_count']}, episodes: {summary['episode_count']}")
    print("Failure-mode coverage:")
    for mode in FAILURE_MODES:
        count = summary["failure_mode_coverage"][mode]
        marker = "ok" if count > 0 else "MISSING"
        print(f"  [{marker}] {mode}: {count} episode(s)")
    if args.dry_run:
        print("(dry-run — no files written)")
    else:
        print(f"Transcripts → {TRANSCRIPTS_OUT.relative_to(PROJECT_ROOT)}")
        print(f"Ground truth → {LABELS_OUT.relative_to(PROJECT_ROOT)}")
        print(f"Manifest → {FIXTURES_V3_ROOT / 'manifest.json'}")
        print(f"Dataset YAML → {DATASET_DIR / 'manifest.yaml'}")
        print(f"Dataset JSON (flat) → {DATASET_FLAT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
