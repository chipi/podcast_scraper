"""A post-nominal credential is not a surname, and reading it as one invented a person (#2075).

The Peter Attia Drive states its host as "Peter Attia, MD". `_surname_token` dropped generational
suffixes but not credentials, and `_canonicalize_to_known_host` took a raw `host.split()[-1]`, so
the surname read as "MD" on one side and "attia" on the other. One human read as two, the ASR's
rendering of his self-introduction could never snap onto the stated host, and "Peter Atiyah" — a
person who does not exist — was published as that episode's GUEST.

Latent until the self-intro reader learned to match "I'm your host, <Name>"; that is what put the
mangled spelling in front of the canonicalizer.
"""

from __future__ import annotations

from podcast_scraper.providers.ml.diarization.roster import (
    _canonicalize_to_known_host,
    _core_name_tokens,
    _same_person,
    _surname_token,
)


class TestTheCredentialIsNotTheSurname:
    def test_surname_token_skips_the_credential(self) -> None:
        assert _surname_token("Peter Attia, MD") == "attia"

    def test_the_credentialled_and_plain_forms_are_one_person(self) -> None:
        assert _same_person("Peter Attia", "Peter Attia, MD") is True

    def test_the_asr_mangle_snaps_onto_the_stated_host(self) -> None:
        # Without this the mangle survives as a distinct person and publishes as a GUEST.
        assert _canonicalize_to_known_host("Peter Atiyah", ["Peter Attia, MD"]) == "Peter Attia, MD"

    def test_generational_suffixes_still_work(self) -> None:
        assert _surname_token("Robert Pape Jr.") == "pape"
        assert _same_person("Robert Pape", "Robert Pape Jr.") is True


class TestItStillRefusesDifferentPeople:
    def test_a_shared_surname_with_a_different_given_name_is_not_the_same_person(self) -> None:
        assert _same_person("Robert Pape", "Karen Pape") is False

    def test_a_guest_sharing_the_hosts_first_name_is_left_alone(self) -> None:
        assert _canonicalize_to_known_host("Peter Thiel", ["Peter Attia, MD"]) == "Peter Thiel"

    def test_a_credential_only_name_has_no_surname(self) -> None:
        # "Dr, MD" is not a person; stripping both must not leave a one-token name looking valid.
        assert _surname_token("Dr, MD") is None


class TestOneDefinitionOfTheParts:
    def test_core_tokens_drop_punctuation_suffixes_and_credentials(self) -> None:
        assert _core_name_tokens("Peter Attia, MD") == ["Peter", "Attia"]
        assert _core_name_tokens("Robert Pape Jr.") == ["Robert", "Pape"]
        assert _core_name_tokens("Adam Rodman") == ["Adam", "Rodman"]
