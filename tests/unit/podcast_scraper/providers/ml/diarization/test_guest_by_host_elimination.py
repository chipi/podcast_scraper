"""The guest is the one substantial voice left when every host is accounted for (#2075).

``_name_guest_voices`` forces a name only when exactly one name and exactly one voice remain. On a
three-to-five-voice episode that never holds, so a real guest stayed a raw ``SPEAKER_NN`` even when
the arithmetic was complete. These tests pin the rule that closes that — and, more importantly,
each condition under which it must REFUSE, because a wrong name is worse than no name (#876).
"""

from __future__ import annotations

from podcast_scraper.providers.ml.diarization.roster import _guest_voice_by_host_elimination

HOST_TEXT = "Hello and welcome back to the show. I'm Russ Roberts, and today we are talking about "
HOST_TEXT += "economics and the history of markets. " * 12
GUEST_TEXT = "Thanks so much for having me. " + ("I think markets are misunderstood. " * 20)
TAPE_TEXT = "Yeah."


def _call(texts, intros, hosts, name, **kw):
    return _guest_voice_by_host_elimination(texts, intros, hosts, name, **kw)


class TestItFires:
    def test_one_pinned_host_and_one_substantial_voice(self) -> None:
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT, "S2": TAPE_TEXT}
        intros = {"S0": "Russ Roberts"}
        assert _call(texts, intros, ["Russ Roberts"], "Ada Lovelace") == "S1"

    def test_both_co_hosts_pinned(self) -> None:
        texts = {"S0": HOST_TEXT, "S1": HOST_TEXT, "S2": GUEST_TEXT, "S3": TAPE_TEXT}
        intros = {"S0": "Kevin Roose", "S1": "Casey Newton"}
        assert _call(texts, intros, ["Kevin Roose", "Casey Newton"], "Ada Lovelace") == "S2"


class TestItRefuses:
    def test_when_a_host_is_unaccounted_for(self) -> None:
        # Only one of the feed's two hosts introduced themselves, so "the voice that is not a
        # pinned host" may simply BE the other host. This is the clause the whole rule rests on.
        texts = {"S0": HOST_TEXT, "S1": HOST_TEXT, "S2": GUEST_TEXT}
        intros = {"S0": "Kevin Roose"}
        assert _call(texts, intros, ["Kevin Roose", "Casey Newton"], "Ada Lovelace") is None

    def test_when_no_host_is_pinned_at_all(self) -> None:
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT, "S2": TAPE_TEXT}
        assert _call(texts, {}, ["Russ Roberts"], "Ada Lovelace") is None

    def test_when_two_voices_remain_substantial(self) -> None:
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT, "S2": GUEST_TEXT}
        intros = {"S0": "Russ Roberts"}
        assert _call(texts, intros, ["Russ Roberts"], "Ada Lovelace") is None

    def test_on_a_two_voice_episode(self) -> None:
        # The two-voice complement in `resolution` already owns that case, at its own measured
        # precision; two rules answering one question is how they drift apart.
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT}
        assert _call(texts, {"S0": "Russ Roberts"}, ["Russ Roberts"], "Ada Lovelace") is None

    def test_on_a_produced_desk_show(self) -> None:
        # Six or more voices: reporters, field tape and vox-pops each get a cluster and nobody
        # names most of them, so "the one voice left over" is a guess, measured at 60%.
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT}
        texts.update({f"S{i}": TAPE_TEXT for i in range(2, 8)})
        assert _call(texts, {"S0": "Russ Roberts"}, ["Russ Roberts"], "Ada Lovelace") is None

    def test_when_the_voice_says_it_is_somebody_else(self) -> None:
        # A co-host the recurrence scan missed still introduces themselves on air, and a
        # self-introduction outranks an arithmetic conclusion.
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT, "S2": TAPE_TEXT}
        intros = {"S0": "Russ Roberts", "S1": "Ryan Knutson"}
        assert _call(texts, intros, ["Russ Roberts"], "Ada Lovelace") is None

    def test_when_the_voice_only_talks_about_that_person(self) -> None:
        texts = {
            "S0": HOST_TEXT,
            "S1": "As I was saying, Ada Lovelace wrote the first program. " * 20,
            "S2": TAPE_TEXT,
        }
        assert _call(texts, {"S0": "Russ Roberts"}, ["Russ Roberts"], "Ada Lovelace") is None

    def test_with_no_host_list(self) -> None:
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT, "S2": TAPE_TEXT}
        assert _call(texts, {"S0": "Russ Roberts"}, [], "Ada Lovelace") is None

    def test_with_no_guest_name(self) -> None:
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT, "S2": TAPE_TEXT}
        assert _call(texts, {"S0": "Russ Roberts"}, ["Russ Roberts"], "") is None


class TestTheSubstantialityFloor:
    def test_a_cameo_is_not_the_guest_candidate(self) -> None:
        # S2 speaks, but under 5% of the episode: it does not compete, so the conclusion stands.
        texts = {"S0": HOST_TEXT, "S1": GUEST_TEXT, "S2": "Right. Mm-hm. Sure."}
        assert _call(texts, {"S0": "Russ Roberts"}, ["Russ Roberts"], "Ada Lovelace") == "S1"
