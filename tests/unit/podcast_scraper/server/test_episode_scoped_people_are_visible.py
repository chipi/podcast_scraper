"""A guest we identified must appear on their own episode, even without a corpus-wide identity.

THE CASE. WSJ "My Monday Morning: Twiggy". The pipeline gets this right all the way to the graph:

    person:unresolved-twiggy-bb004670-...   role=guest   name='Twiggy'

and then the episode's entity card shows the interviewer and NO TWIGGY. `entities_from_kg` drops
every `person:unresolved-*` id.

WHY THAT RULE EXISTS (#1685). A single-token name identifies someone within one episode and nobody
globally — every "Twiggy", "Carly" or "Jensen" in the corpus would merge into one person — so the
id is episode-scoped, and the scoped ids are filtered from corpus-scope surfaces so they cannot
become followable people or pollute cross-episode aggregation. That reasoning is sound and is NOT
changed here.

WHAT IS WRONG is applying it to the EPISODE's own card. On that page "Twiggy" is not
under-specified at all — she is the guest, named, and the only Twiggy in sight. The filter was
protecting a corpus-wide invariant on a surface that is episode-scoped by definition, and the cost
was hiding the guest on the one page where the guest matters most.

THE DISTINCTION THIS PINS. Two different things were being filtered by one rule:

  * ``person:speaker-{ep}-03`` / ``person:speaker-{ep}-host`` — an ANONYMOUS voice. There is no
    name. Rendering it gives the reader "SPEAKER_03" as a person, which is noise. Still dropped.
  * ``person:unresolved-{name}-{ep}`` — a NAMED person whose name is not globally unique. Shown,
    and flagged ``episode_scoped`` so the client can render the chip without offering a tap into an
    entity card that would be empty (#1685's own stated worry).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.server.app_kg_view import entities_from_kg

pytestmark = pytest.mark.unit

EP = "bb004670-ae00-11f1-8d2d-8fac93d999e1"


def kg(*persons) -> Dict[str, Any]:
    return {
        "schema_version": "2.1",
        "nodes": [
            {"id": pid, "type": "Person", "properties": {"name": name, "role": role}}
            for pid, name, role in persons
        ],
    }


def _by_name(kg_doc):
    persons, _orgs, _topics = entities_from_kg(kg_doc)
    return {p.name: p for p in persons}


class TestTheNamedEpisodeScopedGuestIsVisible:
    def test_the_mononym_guest_appears(self) -> None:
        doc = kg(
            (f"person:unresolved-twiggy-{EP}", "Twiggy", "guest"),
            ("person:lane-florsheim", "Lane Florsheim", "host"),
        )
        assert "Twiggy" in _by_name(doc), "the guest is missing from her own episode"

    def test_she_keeps_her_role(self) -> None:
        doc = kg((f"person:unresolved-twiggy-{EP}", "Twiggy", "guest"))
        assert _by_name(doc)["Twiggy"].role == "guest"

    def test_she_is_flagged_episode_scoped(self) -> None:
        # So the client can render the chip WITHOUT a tap target into an empty entity card —
        # the exact failure mode #1685 set out to avoid.
        doc = kg((f"person:unresolved-twiggy-{EP}", "Twiggy", "guest"))
        assert _by_name(doc)["Twiggy"].episode_scoped is True

    def test_a_globally_identified_person_is_not_flagged(self) -> None:
        doc = kg(("person:lane-florsheim", "Lane Florsheim", "host"))
        assert _by_name(doc)["Lane Florsheim"].episode_scoped is False

    def test_a_mentioned_mononym_is_also_shown(self) -> None:
        # Same reasoning: on this episode "Carly" is a person who was talked about.
        doc = kg((f"person:unresolved-carly-{EP}", "Carly", "mentioned"))
        assert "Carly" in _by_name(doc)


class TestAnonymousVoicesStayHidden:
    """The half of #1685 that was right, and stays."""

    def test_a_numbered_diarization_voice_is_dropped(self) -> None:
        doc = kg((f"person:speaker-{EP}-03", "SPEAKER_03", "guest"))
        assert _by_name(doc) == {}

    def test_a_role_placeholder_is_dropped(self) -> None:
        doc = kg((f"person:speaker-{EP}-host", "Host", "host"))
        assert _by_name(doc) == {}

    def test_the_legacy_global_role_id_is_dropped(self) -> None:
        # `person:host` spans 54 prod episodes until those are re-derived.
        doc = kg(("person:host", "Host", "host"))
        assert _by_name(doc) == {}

    def test_a_bare_speaker_display_name_is_dropped(self) -> None:
        doc = kg(("person:whatever", "SPEAKER_07", "guest"))
        assert _by_name(doc) == {}


class TestTheRealEpisodeEndToEnd:
    def test_both_people_reach_the_card_with_the_right_roles(self) -> None:
        doc = kg(
            ("person:lane-florsheim", "Lane Florsheim", "host"),
            (f"person:unresolved-twiggy-{EP}", "Twiggy", "guest"),
            ("person:woody-allen", "Woody Allen", "mentioned"),
            (f"person:speaker-{EP}-00", "SPEAKER_00", "unknown"),
        )
        got = {p.name: (p.role, p.episode_scoped) for p in entities_from_kg(doc)[0]}
        assert got == {
            "Lane Florsheim": ("host", False),
            "Twiggy": ("guest", True),
            "Woody Allen": ("mentioned", False),
        }
