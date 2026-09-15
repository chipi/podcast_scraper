"""Isolated unit tests for speaker_detectors.hosts (E1, RFC-059).

Covers the pure network/organisation classifiers and the transcript self-intro
extractor (#876) directly, without a spaCy model.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    detect_hosts_from_feed,
    extract_self_introduced_host,
    guests_introduced_by_the_host,
    has_org_markers,
    hosts_from_episode_description,
    is_known_network,
    is_network_or_org_author,
    is_plausible_mononym,
    is_publishable_speaker_name,
    looks_like_publisher,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Pushkin", True),  # whole name is a known network
        ("Pushkin Industries", True),  # first token is a known network
        ("Oprah", False),  # real-person mononym, not a known network
        ("Patrick O'Shaughnessy", False),
        ("", False),
    ],
)
def test_is_known_network(name: str, expected: bool) -> None:
    assert is_known_network(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Acme Media", True),  # explicit org marker word
        ("News & Friends", True),  # ampersand marker
        ("Oprah", False),  # trusted mononym person — no org markers
        ("Patrick O'Shaughnessy", False),
        ("", True),  # empty is treated as non-person
    ],
)
def test_has_org_markers(name: str, expected: bool) -> None:
    assert has_org_markers(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NPR", True),  # mononym/acronym rejected for RSS author tags
        ("Oprah", True),  # lone token → treated as network for author tags
        ("Acme Media", True),  # org marker
        ("Patrick O'Shaughnessy", False),  # First Last → a real host
        ("", True),
    ],
)
def test_is_network_or_org_author(name: str, expected: bool) -> None:
    assert is_network_or_org_author(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("The New York Times", True),  # multi-token publisher on the known list
        ("Reuters", True),  # single-token publisher on the known list
        ("The Economist", True),
        ("Bloomberg Businessweek", True),  # first token is a known network
        ("Chicago Tribune", True),  # news-outlet suffix (no known-list entry needed)
        ("The Daily Gazette", True),
        ("Rolling Stone Magazine", True),
        ("Oprah", False),  # real-person mononym is kept (unlike is_network_or_org_author)
        ("Sting", False),
        ("Patrick O'Shaughnessy", False),
        ("Emily Post", False),  # a person whose surname collides with a publisher word
        ("", True),
    ],
)
def test_looks_like_publisher(name: str, expected: bool) -> None:
    # Unlike is_network_or_org_author, a lone real-person token is NOT flagged: this guard
    # strips publishers from already-resolved person surfaces without dropping mononym people.
    assert looks_like_publisher(name) is expected


def test_extract_self_introduced_host_basic() -> None:
    text = "Hello and welcome to the show. I'm Patrick O'Shaughnessy and today we dig in."
    assert extract_self_introduced_host(text) == "Patrick O'Shaughnessy"


def test_extract_self_introduced_host_skips_network_bumper() -> None:
    # Network shows open with a publisher bumper in the same "I'm <X>" shape; the
    # known-network bumper is skipped and the real host name is returned (#876).
    text = "This is Unhedged from the FT. I'm Pushkin. I'm Katie Martin here with you."
    assert extract_self_introduced_host(text) == "Katie Martin"


def test_extract_self_introduced_host_none_when_absent() -> None:
    assert extract_self_introduced_host("No introductions in this clip.") is None
    assert extract_self_introduced_host(None) is None


def test_extract_self_introduced_host_only_scans_intro_window() -> None:
    # A self-intro past the intro window is ignored (a later guest "I'm …").
    text = "x" * 2100 + " I'm Late Guest"
    assert extract_self_introduced_host(text, intro_chars=2000) is None


def test_extract_self_introduced_host_rejects_a_single_opener_token() -> None:
    # ADR-126 opener-leak: ASR renders a disfluency as "I'm But it …"; the "I'm <Cap>" regex
    # captures a bare "But". The 1-token branch used to return it unchecked (its sibling
    # ``distinct_self_introductions`` already guarded this); now a sentence-opener is refused.
    assert extract_self_introduced_host("Well, I'm But it, so let's get started.") is None


def test_extract_self_introduced_host_keeps_scanning_past_an_opener_to_the_real_intro() -> None:
    # ``continue`` (not ``return None``) on rejection: a bogus "I'm But" must not SHADOW a real
    # later self-introduction in the same window. Community-1 "Moonlake" lost the real speaker this
    # way when the opener returned first.
    text = "Well, I'm But it. A moment later — I'm Sarah Chen, and I built this."
    assert extract_self_introduced_host(text) == "Sarah Chen"


def test_extract_self_introduced_host_still_accepts_a_real_mononym() -> None:
    # The 1-token guard rejects openers, not real single-name hosts (Brandon, Oprah, Sting).
    assert (
        extract_self_introduced_host("Hey, I'm Brandon, welcome to the AI for Science podcast.")
        == "Brandon"
    )


@pytest.mark.parametrize(
    ("name", "publishable"),
    [
        # Opener/stop-word leaks the final gate must reject (ADR-126).
        ("But Sun", False),  # community-1 "Moonlake": opener + real surname
        ("So Nick", False),  # community-1 "Quantum": opener + real first name
        ("But", False),  # bare opener from "I'm But it"
        ("Dr", False),  # bare honorific
        ("Andrew Look", False),  # accepted collateral: a surname colliding with a stop-token
        # Real names the gate must keep.
        ("Kevin Roose", True),
        ("Fan-yun Sun", True),
        ("RJ Scringe", True),
        ("Lulu Garcia Navarro", True),
        # One-token names must survive WITHOUT requiring a capital first letter — a lowercase handle
        # already vouched by a trusted source (Latent Space's "swyx") is real. This is why the gate
        # uses a stop-word reject, not the stricter ``is_plausible_mononym``.
        ("swyx", True),
        ("Rohan", True),
        ("Kalshi", True),
        ("Kevin", True),
        ("", False),
        # BUG 5: turn-opener contractions the ASR capitalises at a turn boundary, then a
        # screenplay-line reader grabs whatever precedes the colon as the "speaker"
        # ("I'm: You'll never find a harder worker..."). Both apostrophe spellings (straight
        # and the curly ’ punctuation-restoration can emit) must be rejected, in any case.
        ("I'm", False),
        ("I’m", False),  # curly apostrophe
        ("i'm", False),  # lowercase — the gate does not require a capital for single tokens
        ("I've", False),
        ("I'll", False),
        ("I'd", False),
        ("You're", False),
        ("You’re", False),
        ("You'll", False),
        ("You've", False),
        ("You'd", False),
        ("We're", False),
        ("We've", False),
        ("We'll", False),
        ("They're", False),
        ("They've", False),
        ("It's", False),
        ("It’s", False),
        ("That's", False),
        ("There's", False),
        ("Here's", False),
        ("Let's", False),
        ("Don't", False),
        ("Doesn't", False),
        ("Didn't", False),
        ("Can't", False),
        ("Won't", False),
        ("Wouldn't", False),
        ("Shouldn't", False),
        ("Couldn't", False),
        ("Isn't", False),
        ("Aren't", False),
        ("Wasn't", False),
        ("Weren't", False),
        ("Haven't", False),
        ("Hasn't", False),
        # Real apostrophe-bearing surnames must still pass — the reject list is a closed set of
        # known contractions, not a blanket "has an apostrophe" rule.
        ("O'Brien", True),
        ("D'Angelo", True),
        ("Sarah Gonzales", True),
    ],
)
def test_is_publishable_speaker_name(name: str, publishable: bool) -> None:
    assert is_publishable_speaker_name(name) is publishable


# --- Regression: the catastrophic-backtracking hang + the IGNORECASE case-sensitivity bug ---
# The guest-intro name pattern compiles with re.IGNORECASE for its lowercase cue words. Before the
# `(?-i:...)` fix, IGNORECASE made `[A-Z]` match lowercase too, so the name pattern matched every
# multi-word lowercase phrase in the transcript — crowning non-names AND backtracking for minutes
# on a 77k-char episode (found via a faulthandler stack dump).


def test_guest_intro_name_pattern_is_case_sensitive() -> None:
    """A capitalised introduction is caught; the same words as lowercase prose are not a name."""
    assert guests_introduced_by_the_host({"v": "Jia Li is with us today."}) == {"Jia Li"}
    # Under the IGNORECASE bug this lowercase clause matched as a two-word "name". It must not.
    assert guests_introduced_by_the_host({"v": "the plan is with us today, it is."}) == set()


def test_guest_intro_no_catastrophic_backtracking_on_long_prose() -> None:
    """A long CAPITALISED run — the actual O(n^2) worst case — resolves fast, not in seconds.

    The name regex is case-bound (``(?-i:[A-Z])``), so the quadratic only bites on capitalised
    text: a run of "Cap Cap and Cap Cap ..." that almost forms a name list but never reaches a
    greeting tail forces the (formerly unbounded) nested quantifiers to backtrack over every
    partition at every offset. Measured on the unbounded pattern: ~60k chars → 3.3s, ~120k → 13s;
    the bounded pattern runs both in <0.05s. The 2s budget FAILS the old code and stays clear of
    the fixed cost with wide CI headroom. A lowercase input would not exercise this path at all.
    """
    import time

    text = " and ".join(["Word Alpha"] * 8000) + " zzz"  # ~120k chars, all capitalised
    start = time.perf_counter()
    guests_introduced_by_the_host({"v": text})
    assert time.perf_counter() - start < 2.0, "guest-intro scan backtracked catastrophically"


def test_detect_hosts_from_feed_extracts_journalists_phrase() -> None:
    """The feed_hosts wiring depends on this: the show blurb's "journalists X and Y" names hosts."""
    hosts = detect_hosts_from_feed(
        "Hard Fork",
        "Each week, journalists Kevin Roose and Casey Newton explore the world of tech.",
        ["The New York Times"],
    )
    assert hosts == {"Kevin Roose", "Casey Newton"}


# --- single-name self-intro recovery + introduced-guest greeting (no-anchor feed recall) ---


@pytest.mark.parametrize(
    ("token", "ok"),
    [
        ("Brandon", True),  # real mononym self-intro (Latent Space host)
        ("Neeraj", True),
        ("Oprah", True),
        ("Sting", True),
        ("American", False),  # the "I'm American" class the guard exists for
        ("Republican", False),
        ("Christian", False),
        ("Here", False),  # ordinary word, capitalised by ASR
        ("sorry", False),
        ("", False),
        ("A", False),
    ],
)
def test_is_plausible_mononym(token, ok) -> None:
    assert is_plausible_mononym(token) is ok


def test_bare_honorific_is_not_a_name_or_a_distinct_speaker() -> None:
    # The self-intro regex `\bI'?m\s+([A-Z][\w'’\-]+…)` stops at the period in "I'm Dr. Jane Smith",
    # capturing the bare title "Dr". It must not be a plausible mononym (a voice named "Dr"), and it
    # must not count as a distinct self-introduction — else "I'm Dr. Jane … I'm Jane" would read
    # as a two-person cold-open montage and suppress a real single speaker.
    from podcast_scraper.speaker_detectors.hosts import distinct_self_introductions

    assert is_plausible_mononym("Dr") is False
    assert is_plausible_mononym("Professor") is False
    assert is_plausible_mononym("Brandon") is True  # a real mononym still passes
    text = "Hello, I'm Dr. Jane Smith. And again, I'm Jane Smith, thanks for tuning in."
    assert distinct_self_introductions(text, intro_chars=2000) == ["Jane Smith"]


def test_guest_greeting_name_then_welcome() -> None:
    """The host greets a just-introduced guest by name: 'Jody Rosen, welcome …'."""
    assert guests_introduced_by_the_host({"v": "Jody Rosen, welcome to the show."}) == {
        "Jody Rosen"
    }
    assert guests_introduced_by_the_host({"v": "Nic Harrigan, thanks so much for coming on."}) == {
        "Nic Harrigan"
    }
    # a bare greeting with no preceding name is NOT a guest introduction
    assert guests_introduced_by_the_host({"v": "welcome to Hard Fork this week"}) == set()


@pytest.mark.parametrize("opener", ["But", "Well", "Anyway", "So", "Now"])
def test_greeting_opener_is_not_swept_into_a_name(opener) -> None:
    """R1/#876 fu: a sentence-opening discourse marker the ASR capitalised at a turn boundary

    ("So Nick, welcome") was captured as a 2-word "name" ("So Nick") because the greeting paths
    lacked the ordinary-word guard the self-intro path applies. It must yield NO guest — a wrong
    label is worse than an unnamed voice. Deliberately parametrised over openers other than the two
    incident strings so the test pins the CLASS, not the literals.
    """
    assert guests_introduced_by_the_host({"v": f"{opener} Nick, welcome to the show."}) == set()
    assert (
        guests_introduced_by_the_host({"v": f"{opener} Sun, thanks so much for coming on."})
        == set()
    )
    # positive control in the same shape: a real two-word name is still captured, so the guard is
    # not merely rejecting everything.
    assert guests_introduced_by_the_host({"v": "Jody Rosen, welcome to the show."}) == {
        "Jody Rosen"
    }


def test_discourse_opener_is_not_a_plausible_mononym() -> None:
    """The shared ``_NOT_A_NAME_TOKEN`` set couples both paths: adding the openers to reject

    "So Nick" also (intentionally) makes "I'm Well"/"I'm Now" fail mononym self-intro — same ASR
    capitalisation noise class. Pinned so the side-effect is visible, not latent. Real mononyms are
    unaffected.
    """
    assert is_plausible_mononym("Nick") is True
    assert is_plausible_mononym("Sun") is True
    for opener in ("But", "Well", "Anyway", "Now", "Then", "Look"):
        assert is_plausible_mononym(opener) is False


def test_transcript_intro_does_not_capture_a_lowercase_run_as_a_host(monkeypatch) -> None:
    # N3: under a blanket re.IGNORECASE the [A-Z][a-z]+ name classes matched any letter, so
    # "I'm going to explain how this works" captured "going to explain..." as a host name. The name
    # capture is now case-SENSITIVE via (?-i:...); the cue stays case-insensitive.
    from podcast_scraper.speaker_detectors import hosts

    monkeypatch.setattr(hosts, "_extract_person_entities", lambda text, nlp: [])
    nlp = object()  # truthy so the function does not early-return; NER is patched out above
    assert (
        hosts.detect_hosts_from_transcript_intro(
            "I'm going to explain how this works today before we begin.", nlp
        )
        == set()
    )
    assert "Noah Kravitz" in hosts.detect_hosts_from_transcript_intro(
        "Welcome to the show. I'm Noah Kravitz and today we go deep.", nlp
    )


class TestTheEpisodeDescriptionNamesItsOwnHost:
    """ "X is joined by Y" names BOTH roles — the guest cue read only the second half.

    WHERE THIS IS THE ONLY SOURCE. On a show whose RSS author tag is an organisation, the feed
    cannot state a host: a16z's author is "Andreessen Horowitz", correctly discarded by the org
    filter, and the show ROTATES its host per episode so no feed-level answer would be right
    anyway. The episode's own description is the only place a host is named.

    MEASURED on the 136 production episodes that end with no named speaker: 25 yield a host here,
    22 of them a16z's 48. Extractions verified by hand — Elena Burger, Ben Horowitz, Theo Jaffee,
    Tim Scarfe, and the Alastair Campbell / Rory Stewart pair.
    """

    def test_the_name_before_the_cue_is_the_host(self) -> None:
        got = hosts_from_episode_description(
            "", "Elena Burger is joined by a16z's Andy McCall and Joe Schmidt.", "The a16z Show"
        )
        assert got == {"Elena Burger"}

    def test_two_hosts_share_the_slot(self) -> None:
        got = hosts_from_episode_description(
            "", "Yoko Li and Justine Moore speak with Ideogram's founder.", "The a16z Show"
        )
        assert got == {"Yoko Li", "Justine Moore"}

    def test_the_show_itself_is_never_a_host(self) -> None:
        """ "Planet Money is joined by..." must not mint a person — the #2064 failure."""
        assert (
            hosts_from_episode_description(
                "", "Planet Money is joined by an economist.", "Planet Money"
            )
            == set()
        )

    def test_a_publisher_is_never_a_host(self) -> None:
        assert (
            hosts_from_episode_description(
                "", "Andreessen Horowitz is joined by a founder.", "The a16z Show"
            )
            == set()
        )

    def test_a_single_token_is_refused(self) -> None:
        """A bare first name cannot be bound from here — it needs evidence this function lacks."""
        assert (
            hosts_from_episode_description(
                "", "Tyler is joined by Alison Gopnik.", "Conversations with Tyler"
            )
            == set()
        )

    def test_no_cue_means_no_host(self) -> None:
        assert (
            hosts_from_episode_description(
                "", "Alison Gopnik is a psychologist at Berkeley.", "Conversations with Tyler"
            )
            == set()
        )


class TestTheHostSaysTheirRoleBeforeTheirName:
    """ "I'm your host, Noah Kravitz" — the commonest opening in the corpus, and unmatchable.

    THE PATTERN REQUIRED THE NAME TO FOLLOW THE CUE IMMEDIATELY. `your` is lowercase, so the
    capitalised run never started and the scanner returned nothing at all. Measured over the 136
    production episodes that end with no named speaker: the old pattern matched **0**, the widened
    one matches **36** — 25 of NVIDIA's 31, 10 of The Rest Is Politics' 12, and Ottoman History.

    "The host never self-introduces on these shows" was a property of the REGEX, and was reported
    as a property of the data.

    REGRESSION CHECK, on the 1,986 episodes whose roster already names a host:

        agrees with the roster host   721 -> 862   (+141)
        disagrees                     186 -> 214   (+28)
        finds nothing               1,079 -> 910

    5:1, and the disagreements are not all errors — several are the extractor being RIGHT where
    the roster is wrong ("Kevin Rothrock" against a roster saying "Boris Goryachev" on The Naked
    Pravda; "Gustavo Ribeiro" against a roster saying "The Brazilian Report").
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Welcome to the NVIDIA AI podcast. I'm your host, Noah Kravitz.", "Noah Kravitz"),
            ("I'm the co-host, Casey Newton, and today we talk shop.", "Casey Newton"),
            ("I'm your co-host Sarah Guo and this week we cover agents.", "Sarah Guo"),
            # The idiom that names ONESELF in British broadcasting.
            ("The Rest Is Politics: Leading, with me, Alastair Campbell.", "Alastair Campbell"),
            ("...and me, Rory Stewart, back for another season.", "Rory Stewart"),
            # Still works, unchanged.
            ("Hello and welcome, I'm Patrick O'Shaughnessy.", "Patrick O'Shaughnessy"),
        ],
    )
    def test_the_role_phrase_does_not_hide_the_name(self, text: str, expected: str) -> None:
        assert extract_self_introduced_host(text, intro_chars=4000) == expected

    def test_joining_me_introduces_somebody_else(self) -> None:
        """THE GUARD ON THE NEW FORM. "joining me, X" is a GUEST — binding it to the host's voice
        would paint a guest's name on the host, the exact direction of error this module prevents.
        Only "with me," and "and me," are self-reference, and the comma is required."""
        assert (
            extract_self_introduced_host(
                "Joining me, Alison Gopnik, a professor at Berkeley.", intro_chars=4000
            )
            is None
        )

    def test_the_network_bumper_is_still_skipped(self) -> None:
        """#876 — the widened pattern must not re-open the "I'm Pushkin" leak."""
        assert (
            extract_self_introduced_host(
                "This is Unhedged. I'm Pushkin. I'm Katie Martin.", intro_chars=4000
            )
            == "Katie Martin"
        )

    def test_an_ordinary_phrase_is_still_not_a_name(self) -> None:
        assert extract_self_introduced_host("I'm Coming Out was a great song.") is None


class TestDistinctSelfIntroductionsReadsBothForms:
    """The "and with me, <Name>" form was invisible here while its sibling saw it (#2075)."""

    def test_the_with_me_form_is_read(self) -> None:
        from podcast_scraper.speaker_detectors.hosts import distinct_self_introductions

        text = "Welcome back to the show. I'm Kevin Roose, and with me, Casey Newton."
        assert distinct_self_introductions(text, intro_chars=2000) == [
            "Kevin Roose",
            "Casey Newton",
        ]

    def test_joining_me_still_introduces_somebody_else(self) -> None:
        from podcast_scraper.speaker_detectors.hosts import distinct_self_introductions

        # "joining me, X" is a GUEST introduction; admitting it here would make every week's
        # guest look like a recurring co-host.
        text = "I'm Kevin Roose. Joining me, Sam Altman."
        assert distinct_self_introductions(text, intro_chars=2000) == ["Kevin Roose"]


class TestRecurrentHostsAcrossEpisodes:
    """A guest appears once; a host appears every week (#2075)."""

    @staticmethod
    def _run(episodes, **kw):
        from podcast_scraper.speaker_detectors.hosts import recurrent_hosts_across_episodes

        return recurrent_hosts_across_episodes(episodes, **kw)

    def test_a_weekly_self_introducer_is_the_host(self) -> None:
        episodes = [["Russ Roberts"], ["Russ Roberts"], ["Russ Roberts"], ["Russ Roberts"]]
        assert self._run(episodes) == {"Russ Roberts"}

    def test_a_one_off_guest_is_not(self) -> None:
        episodes = [
            ["Russ Roberts", "Ada Lovelace"],
            ["Russ Roberts"],
            ["Russ Roberts"],
            ["Russ Roberts"],
        ]
        assert self._run(episodes) == {"Russ Roberts"}

    def test_three_episodes_is_not_enough_on_a_long_feed(self) -> None:
        # Clears min_episodes, fails min_share: a recurring guest on a twenty-episode show.
        episodes = [["Ada Lovelace"]] * 3 + [["Russ Roberts"]] * 17
        assert self._run(episodes) == {"Russ Roberts"}

    def test_asr_spelling_variants_are_one_host_not_three(self) -> None:
        # Counted separately none of these clears 25%; merged, Casey is plainly the co-host.
        episodes = (
            [["Kevin Roose", "Casey Newn"]] * 4
            + [["Kevin Roose", "Casey Noon"]] * 4
            + [["Kevin Roose", "Casey Newton"]] * 4
        )
        out = self._run(episodes)
        assert "Kevin Roose" in out
        assert len([n for n in out if n.lower().startswith("casey")]) == 1

    def test_the_show_saying_its_own_name_is_not_a_presenter(self) -> None:
        episodes = [["Trivium China"]] * 10
        assert self._run(episodes, feed_title="The Trivium China Podcast") == set()

    def test_a_recurring_mononym_is_refused(self) -> None:
        # "Brandon" cleared the thresholds on Latent Space and is not one of its hosts.
        assert self._run([["Brandon"]] * 10) == set()

    def test_a_network_bumper_is_refused(self) -> None:
        assert self._run([["Pushkin Industries"]] * 10) == set()


class TestHostsFromEpisodeDescriptionRefusesNonHosts:
    """The name before the cue is the host — except when it isn't (#2075).

    Every case here was found by running the extractor over the production corpus, not invented.
    """

    @staticmethod
    def _run(title, desc, feed):
        from podcast_scraper.speaker_detectors.hosts import hosts_from_episode_description

        return hosts_from_episode_description(title, desc, feed)

    def test_the_plain_case_still_works(self) -> None:
        out = self._run(
            "Whatnot's global marketplace",
            "David George is joined by Grant LaFontaine, co-founder of Whatnot.",
            "The a16z Show",
        )
        assert out == {"David George"}

    def test_two_hosts_share_the_slot(self) -> None:
        out = self._run(
            "",
            "Jack Farley and Max Wiethe speak with Ara Kharazian, Lead Economist at Ramp.",
            "The a16z Show",
        )
        assert out == {"Jack Farley", "Max Wiethe"}

    def test_the_title_does_not_run_into_the_description(self) -> None:
        # Joined with a bare space this produced a host called "Forecasting Theo Jaffee".
        out = self._run(
            "Robin Hanson on Prediction Markets and the Future of Forecasting",
            "Theo Jaffee speaks with economist Robin Hanson about prediction markets.",
            "The a16z Show",
        )
        assert out == {"Theo Jaffee"}

    def test_a_job_title_is_not_part_of_the_name(self) -> None:
        # "a16z General Partner David George is joined by..." — the run has no left anchor.
        out = self._run(
            "",
            "a16z General Partner David George is joined by Grant LaFontaine.",
            "The a16z Show",
        )
        assert out == {"David George"}

    def test_the_guest_first_phrasing_is_refused(self) -> None:
        # "Listen as journalist Stephen Witt speaks with EconTalk's Russ Roberts..." — the show
        # naming itself after the cue is what marks the sentence as running the other way.
        out = self._run(
            "The Man Who Built NVIDIA (with Stephen Witt)",
            "Listen as journalist Stephen Witt speaks with EconTalk's Russ Roberts about Jensen.",
            "EconTalk",
        )
        assert out == set()

    def test_the_show_can_never_be_its_own_host(self) -> None:
        out = self._run("", "Planet Money is joined by an economist.", "Planet Money")
        assert out == set()
