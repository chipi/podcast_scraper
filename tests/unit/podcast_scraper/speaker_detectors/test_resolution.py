class TestTheRefutationIsEvidenceNotJustAVeto:
    """A refused name says where the person ISN'T — on two voices that identifies them.

    MEASURED ON A REAL FAILURE. Conversations with Tyler, "Alison Gopnik on Childhood Learning":
    the model answered `SPEAKER_00 = Alison Gopnik`, but SPEAKER_00 is Tyler, the host, who says
    "this is Tyler" and "I am talking with Alison Gopnik". The third-person guard correctly refused
    it — and the episode then shipped with NO named speakers, an empty roster and zero SPOKEN_BY
    edges, while the fact that Alison must therefore be the other voice was discarded with the
    refutation.

    WHY STRICTLY TWO VOICES. Replayed against episodes whose voices are already correctly named on
    the production snapshot, "bind the one voice not refuted for this name" scores:

        any voice count    338 fires   90.5% correct
        exactly 2 voices   297 fires   98.0% correct

    With three or more voices "exactly one unrefuted" misattributes about one time in ten, which is
    the failure this module exists to prevent. The comparison for the two-voice case is not 98%
    against a perfect answer: this fires ONLY where the proposal was already discarded, so the
    alternative is no name at all.
    """

    HOST = "Hi listeners, this is Tyler. Today I am talking with Alison Gopnik about childhood."
    GUEST = "Well, when we started this research, the big puzzle was how children actually learn."

    @staticmethod
    def _llm(payload):
        import json as _json

        return lambda _prompt: _json.dumps({"voices": payload})

    def test_the_name_lands_on_the_other_voice(self) -> None:
        from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

        out = resolve_voices_and_roles(
            stated_names=["Alison Gopnik"],
            voice_texts={"SPEAKER_00": self.HOST, "SPEAKER_01": self.GUEST},
            complete=self._llm({"SPEAKER_00": {"name": "Alison Gopnik", "role": "guest"}}),
            episode_title="Alison Gopnik on Childhood Learning",
        )
        assert (
            out["SPEAKER_01"].name == "Alison Gopnik"
        ), "the refutation says she is not SPEAKER_00; on two voices that names SPEAKER_01"
        assert out["SPEAKER_00"].name is None, "the refused voice must still not carry the name"

    def test_three_voices_do_not_complement(self) -> None:
        """THE GUARD ON THE GUARD. 90.5% vs 98.0% is the whole reason this is two-voice only."""
        from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

        out = resolve_voices_and_roles(
            stated_names=["Alison Gopnik"],
            voice_texts={
                "SPEAKER_00": self.HOST,
                "SPEAKER_01": self.GUEST,
                "SPEAKER_02": "I agree with that entirely, it matches our own findings.",
            },
            complete=self._llm({"SPEAKER_00": {"name": "Alison Gopnik", "role": "guest"}}),
            episode_title="Alison Gopnik on Childhood Learning",
        )
        assert all(
            v.name is None for v in out.values()
        ), "with a third voice, 'the one not refuted' is not evidence — it misattributes ~1 in 10"

    def test_it_never_overwrites_a_voice_the_model_named_directly(self) -> None:
        from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

        out = resolve_voices_and_roles(
            stated_names=["Alison Gopnik", "Tyler Cowen"],
            voice_texts={"SPEAKER_00": self.HOST, "SPEAKER_01": self.GUEST},
            complete=self._llm(
                {
                    "SPEAKER_00": {"name": "Alison Gopnik", "role": "guest"},
                    "SPEAKER_01": {"name": "Tyler Cowen", "role": "host"},
                }
            ),
            episode_title="Alison Gopnik on Childhood Learning",
        )
        assert out["SPEAKER_01"].name == "Tyler Cowen", "a direct answer outranks the complement"

    def test_it_abstains_when_the_other_voice_also_talks_about_them(self) -> None:
        """Both voices discussing the person is no evidence either way — abstain."""
        from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

        out = resolve_voices_and_roles(
            stated_names=["Alison Gopnik"],
            voice_texts={
                "SPEAKER_00": self.HOST,
                "SPEAKER_01": "I read Alison Gopnik's book last year and she makes a good case.",
            },
            complete=self._llm({"SPEAKER_00": {"name": "Alison Gopnik", "role": "guest"}}),
            episode_title="Alison Gopnik on Childhood Learning",
        )
        assert all(v.name is None for v in out.values())

    def test_an_invented_name_still_cannot_reach_a_voice_through_the_complement(self) -> None:
        """#876 stands: the model identifies, never authors. The complement is no back door."""
        from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

        out = resolve_voices_and_roles(
            stated_names=["Alison Gopnik"],
            voice_texts={"SPEAKER_00": self.HOST, "SPEAKER_01": self.GUEST},
            complete=self._llm({"SPEAKER_00": {"name": "Someone Unstated", "role": "guest"}}),
            episode_title="Alison Gopnik on Childhood Learning",
        )
        assert all(v.name is None for v in out.values())
