"""Wikidata resolves identity; Wikipedia supplies prose (#2111).

`WikipediaProvider` guessed an article title from the display name, which conflated three
outcomes into one 404: genuinely absent, present-in-Wikidata-only, and present-in-Wikipedia-
but-at-another-title. `WikidataResolvedProvider` resolves structurally and then fetches the
article Wikidata names via `sitelinks.enwiki`.

The two properties that matter most are NOT the happy path:

1. **Old payloads keep deriving.** 940 cached payloads on prod are bare Wikipedia summaries
   with no `schema` key. A re-derive must not drop the 793 rows built from them.
2. **Ambiguity is refused — but only REAL ambiguity.** `wbsearchentities` on a common name
   always returns someone, and a confident wrong match puts the wrong person's biography on
   an episode page. The discriminator is the enwiki article, not humanness: "Balaji
   Srinivasan" really returns five humans, four of them ORCID stubs, and refusing on that
   count would drop someone we serve today.
3. **The card contract is bio + description + image.** Wikidata resolves identity; it never
   supplies content. A Wikidata-only person is a miss, not a one-line stub row.

No network: every test drives `httpx.MockTransport`.
"""

from __future__ import annotations

import httpx
import pytest

from podcast_scraper.enrichment.enrichers.person_web import (
    _FALLBACK_SCHEMA,
    _RESOLVED_SCHEMA,
    _wiki_file_title,
    TransientFetchError,
    WikidataResolvedProvider,
)

WD = "https://www.wikidata.org/w/api.php"


def _human(qid, *, desc="American researcher", enwiki=None, image=None):
    ent = {
        "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]},
        "descriptions": {"en": {"value": desc}},
        "labels": {"en": {"value": "Someone"}},
    }
    if enwiki:
        ent["sitelinks"] = {"enwiki": {"title": enwiki}}
    if image:
        ent["claims"]["P18"] = [{"mainsnak": {"datavalue": {"value": image}}}]
    return ent


def _resolve(routes, person_id, name):
    """fetch_raw + derive in one step — the pair is how the enricher actually calls it."""
    p = _provider(routes)
    raw = p.fetch_raw(person_id, name)
    return p.derive(person_id, name, raw) if raw is not None else None


def _provider(routes):
    """routes: callable(url) -> (status, json)."""

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = routes(str(request.url))
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WikidataResolvedProvider(client=client)


class TestBackwardCompatibility:
    """The 940 cached bare-summary payloads must keep working."""

    def test_legacy_wikipedia_payload_still_derives(self):
        p = _provider(lambda url: (200, {}))
        legacy = {
            "extract": "Katie Couric is an American journalist.",
            "description": "American journalist",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Katie_Couric"}},
            "thumbnail": {"source": "https://upload.wikimedia.org/x.jpg"},
        }

        info = p.derive("person:kc", "Katie Couric", legacy)

        assert info is not None, "a legacy payload must not stop deriving"
        assert info.bio.startswith("Katie Couric is an American journalist")
        assert info.description == "American journalist"
        assert info.source == "wikipedia", "legacy rows keep their original provenance"

    def test_legacy_disambiguation_payload_still_refused(self):
        p = _provider(lambda url: (200, {}))

        assert p.derive("person:x", "X", {"type": "disambiguation", "extract": "..."}) is None


class TestAmbiguityIsRefused:
    def test_two_people_who_BOTH_have_articles_and_NEITHER_is_primary(self):
        """Two notable Adam Browns, neither at the bare title — genuinely unguessable.

        Note both titles are parenthetically qualified. If one of them sat at plain
        "Adam Brown", Wikipedia would have declared it the primary topic and we would take it;
        see TestWikipediaAlreadyDisambiguated.
        """

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {
                    "search": [
                        {"id": "Q1", "label": "Adam Brown"},
                        {"id": "Q2", "label": "Adam Brown"},
                    ]
                }
            return 200, {
                "entities": {
                    "Q1": _human("Q1", enwiki="Adam Brown (politician)"),
                    "Q2": _human("Q2", enwiki="Adam Brown (musician)"),
                }
            }

        p = _provider(routes)
        raw = p.fetch_raw("person:ab", "Adam Brown")

        assert raw is not None, "a payload is still persisted so a future re-derive can revisit"
        assert raw["articled_ids"] == ["Q1", "Q2"]
        assert "wikidata_id" not in raw, "no winner was picked"
        assert p.derive("person:ab", "Adam Brown", raw) is None

    def test_non_human_match_produces_no_row(self):
        """A name matching a ship or a film is not a person.

        Zero articled candidates DOES fall back to the direct title lookup (no-regression
        rule), so the article must 404 for this to end as no-row — otherwise we would be
        asserting the fallback away.
        """

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q9", "label": "Discovery"}]}
            if "wbgetentities" in url:
                return 200, {
                    "entities": {
                        "Q9": {
                            "claims": {
                                "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q11446"}}}}]
                            }
                        }
                    }
                }
            return 404, {"detail": "no such article"}

        p = _provider(routes)
        raw = p.fetch_raw("person:d", "Discovery")

        assert raw["articled_ids"] == []
        assert p.derive("person:d", "Discovery", raw) is None

    def test_absent_from_BOTH_sources_is_an_authoritative_miss(self):
        """Anish Acharya: no Wikidata item AND no article. Only then may we cache a miss.

        Note this now requires BOTH to be empty. Wikidata returning nothing is no longer
        sufficient on its own — the fallback still asks Wikipedia directly first.
        """

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": []}
            return 404, {"detail": "no such article"}

        assert _provider(routes).fetch_raw("person:aa", "Anish Acharya") is None


class TestResolutionRemovesTheTitleGuess:
    def test_article_is_fetched_at_the_title_wikidata_gives(self):
        """The bug this exists for: the title comes from sitelinks, never from the name."""
        seen = []

        def routes(url):
            seen.append(url)
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q42", "label": "John Smith"}]}
            if "wbgetentities" in url:
                return 200, {"entities": {"Q42": _human("Q42", enwiki="John Smith (economist)")}}
            return 200, {"extract": "An economist.", "description": "British economist"}

        p = _provider(routes)
        raw = p.fetch_raw("person:js", "John Smith")

        assert raw["enwiki_title"] == "John Smith (economist)"
        assert any(
            "John_Smith_%28economist%29" in u or "John_Smith_(economist)" in u for u in seen
        ), "the summary must be fetched at the disambiguated title, not the bare name"
        info = p.derive("person:js", "John Smith", raw)
        assert info is not None and info.bio == "An economist."

    def test_the_articled_candidate_wins_over_stub_namesakes(self):
        """The real "Balaji Srinivasan": 5 humans, 4 of them ORCID stubs with no article.

        Refusing on "more than one human" would drop a person we serve today — a regression
        dressed as a safety improvement. The one with an article is the one with a biography.
        """

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": q} for q in ("Q87", "Q102", "Q62", "Q91", "Q85")]}
            if "wbgetentities" in url:
                return 200, {
                    "entities": {
                        "Q87": _human("Q87", desc="American entrepreneur", enwiki="Balaji S"),
                        "Q102": _human("Q102", desc="Ph.D. The University of Chicago 1995"),
                        "Q62": _human("Q62", desc="researcher (ORCID 0000-0001-9996-0535)"),
                        "Q91": _human("Q91", desc="researcher (ORCID 0000-0002-1430-7818)"),
                        "Q85": _human("Q85", desc="researcher (ORCID 0000-0001-9378-0525)"),
                    }
                }
            return 200, {"extract": "An entrepreneur and investor.", "description": "Investor"}

        p = _provider(routes)
        raw = p.fetch_raw("person:bs", "Balaji Srinivasan")

        assert len(raw["human_ids"]) == 5, "all five really are humans"
        assert raw["articled_ids"] == ["Q87"], "only one of them has an article"
        info = p.derive("person:bs", "Balaji Srinivasan", raw)
        assert info is not None and info.bio == "An entrepreneur and investor."


class TestTheCardContract:
    """bio + description + image. A row that cannot carry all three is a miss, not a stub."""

    def test_a_wikidata_only_person_is_a_miss_not_a_one_line_row(self):
        """Adam Mastroianni: Q125651464, no enwiki, no P18. Nothing to render but a subtitle.

        Emitting that would put a one-line card next to fully-populated neighbours. Wikidata
        resolves identity; it is not a content source.
        """

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q125651464", "label": "Adam Mastroianni"}]}
            if "wbgetentities" in url:
                return 200, {
                    "entities": {"Q125651464": _human("Q125651464", desc="American Rhodes Scholar")}
                }
            return 404, {"detail": "no such article"}

        p = _provider(routes)
        raw = p.fetch_raw("person:am", "Adam Mastroianni")

        assert p.derive("person:am", "Adam Mastroianni", raw) is None

    def test_wikidata_fills_a_description_the_article_omits(self):
        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q1"}]}
            if "wbgetentities" in url:
                ent = _human("Q1", desc="American economist", enwiki="X")
                return 200, {"entities": {"Q1": ent}}
            return 200, {"extract": "Prose.", "thumbnail": {"source": "https://upload/x.jpg"}}

        info = _resolve(routes, "person:x", "X")

        assert info.bio == "Prose."
        assert info.description == "American economist", "subtitle filled from Wikidata"
        assert info.image_url == "https://upload/x.jpg"

    def test_wikidata_fills_a_portrait_the_article_omits(self):
        """Wikipedia REST summaries often carry no thumbnail; the item may still have P18."""

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q2"}]}
            if "wbgetentities" in url:
                return 200, {"entities": {"Q2": _human("Q2", enwiki="Y", image="Y pic.jpg")}}
            return 200, {"extract": "Prose about Y.", "description": "A person"}

        info = _resolve(routes, "person:y", "Y")

        assert info.bio == "Prose about Y."
        assert info.description == "A person", "the article's own subtitle wins"
        assert info.image_url and "Special:FilePath" in info.image_url
        assert "Y_pic.jpg" in info.image_url, "spaces become underscores, then url-quoted"
        assert "width=" in info.image_url, (
            "Special:FilePath without ?width= serves the full-resolution original (16 MB for "
            "Katie Couric's portrait, 8x _IMAGE_MAX_BYTES) — every P18 photo would be cached "
            "as a PERMANENT skip. Verified live 2026-09-17: 16358503 B bare vs 177700 B at "
            "width=640."
        )
        assert _wiki_file_title(info.image_url) == "Y_pic.jpg", (
            "the query string must not leak into the File: title or imageinfo reports missing "
            "and we skip the photo for lack of a licence"
        )

    def test_the_articles_own_photo_is_never_overwritten(self):
        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q3"}]}
            if "wbgetentities" in url:
                return 200, {"entities": {"Q3": _human("Q3", enwiki="Z", image="Wrong.jpg")}}
            return 200, {"extract": "Prose.", "thumbnail": {"source": "https://upload/right.jpg"}}

        info = _resolve(routes, "person:z", "Z")

        assert info.image_url == "https://upload/right.jpg"


class TestTransportFailuresStillRaise:
    def test_blocked_search_raises_rather_than_claiming_absence(self):
        """The 2026-09-16 lesson: 'could not ask' must never become a 30-day miss."""
        p = _provider(lambda url: (403, {"detail": "blocked"}))

        with pytest.raises(TransientFetchError):
            p.fetch_raw("person:x", "Someone")


class TestNoRegressionFallback:
    """The new path must never return LESS than the Wikipedia-only provider did."""

    def test_wikidata_blind_spot_falls_back_to_direct_title(self):
        """Wikidata search misses them, but the article exists at the guessed title."""

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": []}
            return 200, {"extract": "A person.", "description": "Someone notable"}

        p = _provider(routes)
        raw = p.fetch_raw("person:x", "Obscure Person")

        assert raw is not None, "must fall back rather than declare absence"
        # What matters is that the fallback derives via the WIKIPEDIA path — its schema is not the
        # resolved one. It used to carry no ``schema`` key at all, which made it indistinguishable
        # from a provider-#1 fossil, and #2158 must tell those apart: a fossil is dropped so the
        # resolver can retry it, whereas dropping THIS would just take the same fallback again on
        # every run. Hence the explicit self-identifying marker.
        assert raw.get("schema") == _FALLBACK_SCHEMA, "the fallback must mark itself"
        assert raw.get("schema") != _RESOLVED_SCHEMA, "still derives via Wikipedia"
        info = p.derive("person:x", "Obscure Person", raw)
        assert info is not None and info.bio == "A person."

    def test_ambiguity_does_NOT_fall_back(self):
        """Two articled humans stay refused — a title guess would just pick one blindly."""
        calls = []

        def routes(url):
            calls.append(url)
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q1"}, {"id": "Q2"}]}
            if "wbgetentities" in url:
                # Neither title is the bare display name, so neither is the primary topic.
                return 200, {
                    "entities": {
                        "Q1": _human("Q1", enwiki="Adam Brown (politician)"),
                        "Q2": _human("Q2", enwiki="Adam Brown (musician)"),
                    }
                }
            return 200, {"extract": "Wrong person.", "description": "nope"}

        p = _provider(routes)
        raw = p.fetch_raw("person:ab", "Adam Brown")

        assert p.derive("person:ab", "Adam Brown", raw) is None
        assert not any(
            "page/summary" in c for c in calls
        ), "ambiguity must not trigger a title-guess fallback"


class TestApiContractIsUnchanged:
    """The UI and app depend on this envelope; the provider must not alter its shape."""

    def test_every_derived_row_has_a_non_empty_bio(self):
        """AppPersonWeb declares bio as required and non-nullable. No row may emit None."""
        from podcast_scraper.server.schemas import AppPersonWeb

        assert AppPersonWeb.model_fields["bio"].is_required(), (
            "if bio ever becomes optional, revisit the Wikidata-only branch — until then a "
            "None bio is a 500"
        )

        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": "Q1", "label": "X"}]}
            if "wbgetentities" in url:
                return 200, {"entities": {"Q1": _human("Q1", enwiki="X", image="P.jpg")}}
            return 200, {"extract": "Prose about X.", "description": "A person"}

        p = _provider(routes)
        info = p.derive("person:x", "X", p.fetch_raw("person:x", "X"))

        assert info.bio and info.bio.strip(), "a derived row must always carry prose"
        AppPersonWeb(
            bio=info.bio,
            description=info.description,
            image_url=info.image_url,
            source=info.source,
            license=info.license,
        )


class TestWikipediaAlreadyDisambiguated:
    """Several articles is not ambiguity — the bare title IS Wikipedia's answer.

    Measured on prod 2026-09-17: an earlier "more than one article means refuse" rule would
    have dropped 5 of 30 audited rows — Aaron Burr, Alex Jones, Bill Cassidy, Charlie Roberts
    and Henry VIII — every one of them a person we serve correctly today.
    """

    @staticmethod
    def _routes(entities, extract="Prose."):
        def routes(url):
            if "wbsearchentities" in url:
                return 200, {"search": [{"id": q} for q in entities]}
            if "wbgetentities" in url:
                return 200, {"entities": entities}
            return 200, {"extract": extract, "description": "A person"}

        return routes

    @pytest.mark.parametrize(
        "name,titles",
        [
            ("Alex Jones", ["Alex Jones", "Alex Jones (actor)"]),
            ("Aaron Burr", ["Aaron Burr", "Aaron Burr Sr."]),
            ("Bill Cassidy", ["Bill Cassidy (footballer, born 1917)", "Bill Cassidy"]),
            ("Henry VIII", ["Henry VIII of Waldeck", "Henry VIII", "Henry VII of Brzeg"]),
        ],
    )
    def test_the_bare_title_wins(self, name, titles):
        """Real candidate sets, straight from the prod audit."""
        ents = {f"Q{i}": _human(f"Q{i}", enwiki=t) for i, t in enumerate(titles)}

        info = _resolve(self._routes(ents), "person:x", name)

        assert info is not None, f"{name} is served today and must not become a miss"

    def test_the_match_is_case_and_whitespace_insensitive(self):
        ents = {"Q1": _human("Q1", enwiki="Alex Jones"), "Q2": _human("Q2", enwiki="A (actor)")}

        assert _resolve(self._routes(ents), "person:x", "  alex jones  ") is not None

    def test_a_qualified_title_alone_still_resolves(self):
        """One candidate needs no tie-break — that is the John Smith (economist) case."""
        ents = {"Q1": _human("Q1", enwiki="John Smith (economist)")}

        assert _resolve(self._routes(ents), "person:js", "John Smith") is not None

    def test_two_bare_title_matches_are_still_refused(self):
        """Cannot happen with real Wikipedia titles, but the rule must not pick arbitrarily."""
        ents = {"Q1": _human("Q1", enwiki="Jane Roe"), "Q2": _human("Q2", enwiki="jane roe")}

        assert _resolve(self._routes(ents), "person:jr", "Jane Roe") is None
