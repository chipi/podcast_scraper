"""A known host must be surfaced as the host, not as "mentioned" (#2060).

Prod, native app: the host of the show a listener is currently inside is rendered with the
"mentioned" relationship — the same weak tag a third party who was only name-dropped would get.

The cause is an ordering interaction between the two paths that build person nodes:

  LLM KG extraction   ->  role="mentioned"  (hardcoded, kg/pipeline.py:839)
  speaker pipeline    ->  role="host"/"guest" via _append_pipeline_entities

`_append_pipeline_entities` is handed `existing_entity_keys` and skips any name already present:

    key = _entity_dedup_key(name=n, entity_kind=kind)
    if key in existing_entity_keys:
        continue                      # <-- the host NODE is never appended
    nodes.append(v2_node)

So when the LLM has already extracted the host from the transcript — which it does whenever the
host is named aloud, i.e. almost always — the typed host node is never created at all. The
HOSTS/GUESTS_ON show edge IS emitted (it is added above the `continue`), which is why the graph
looks correct while the rendered role does not.

`_dedupe_nodes_by_id` has a role-precedence rule that upgrades "mentioned" -> host/guest, and it
never fires here: precedence needs TWO nodes to merge, and the skip guarantees there is only one.
"""

from __future__ import annotations

import pytest

from podcast_scraper.kg.pipeline import build_artifact

pytestmark = pytest.mark.unit

_HOST = "Aaron Levie"
_GUEST = "Theo Jaffee"


def _artifact(llm_entities):
    """A KG artifact where the speaker pipeline knows the host and guest."""
    return build_artifact(
        "ep1",
        "Some transcript text.",
        podcast_id="p1",
        episode_title="T",
        publish_date="2025-01-01T00:00:00Z",
        transcript_ref="t.txt",
        prefilled_partial={"topics": [], "entities": llm_entities},
        detected_hosts=[_HOST],
        detected_guests=[_GUEST],
    )


def _role_of(artifact, name: str):
    for node in artifact["nodes"]:
        props = node.get("properties") or {}
        if props.get("name") == name:
            return props.get("role")
    return None


class TestTheHostIsNotDemotedByBeingMentioned:
    def test_a_host_the_llm_also_extracted_is_still_the_host(self) -> None:
        # The transcript names the host aloud, so the LLM extracts them as a mentioned entity.
        art = _artifact([{"name": _HOST, "entity_kind": "person"}])
        assert _role_of(art, _HOST) == "host", (
            "the host was demoted to 'mentioned' because the LLM got there first — the exact "
            "defect a listener sees on the episode panel"
        )

    def test_a_guest_the_llm_also_extracted_is_still_the_guest(self) -> None:
        art = _artifact([{"name": _GUEST, "entity_kind": "person"}])
        assert _role_of(art, _GUEST) == "guest"

    def test_both_survive_when_the_llm_extracted_both(self) -> None:
        art = _artifact(
            [{"name": _HOST, "entity_kind": "person"}, {"name": _GUEST, "entity_kind": "person"}]
        )
        assert (_role_of(art, _HOST), _role_of(art, _GUEST)) == ("host", "guest")

    def test_the_host_role_wins_regardless_of_which_path_ran_first(self) -> None:
        # Whoever built the node first must not decide what the person IS.
        art = _artifact([{"name": _HOST, "entity_kind": "person"}])
        roles = [
            (n.get("properties") or {}).get("role")
            for n in art["nodes"]
            if (n.get("properties") or {}).get("name") == _HOST
        ]
        assert roles and all(r == "host" for r in roles), roles


class TestGenuineMentionsAreUnchanged:
    """Over-promoting would make every name-dropped third party look like a participant."""

    def test_someone_neither_hosting_nor_guesting_stays_mentioned(self) -> None:
        art = _artifact([{"name": "Satya Nadella", "entity_kind": "person"}])
        assert _role_of(art, "Satya Nadella") == "mentioned"

    def test_an_organization_is_untouched(self) -> None:
        art = _artifact([{"name": "Box", "entity_kind": "company"}])
        assert _role_of(art, "Box") == "mentioned"


class TestMetadataBeatsTheTranscriptOnKind:
    """The LLM mis-kinding a host must not erase them (#2060 / advisor H4).

    The #2057 prompt tells the model to "use object when unsure", which raises the rate of a real
    person being typed `event` / `object`. When that happened the entity became `object:{slug}`,
    the speaker pipeline's `person:{slug}` node was skipped as a duplicate by NAME, and the
    episode ended with NO Person node for its own host plus a dangling HOSTS edge.
    """

    @pytest.mark.parametrize("wrong_kind", ["event", "object", "podcast", "banana", None])
    def test_a_known_host_is_a_person_whatever_the_llm_called_them(self, wrong_kind) -> None:
        art = _artifact([{"name": _HOST, "entity_kind": wrong_kind}])
        people = [
            n
            for n in art["nodes"]
            if n["type"] == "Person" and (n.get("properties") or {}).get("name") == _HOST
        ]
        assert people, f"the host vanished when the LLM called them {wrong_kind!r}"
        assert (people[0].get("properties") or {}).get("role") == "host"

    def test_a_known_guest_is_a_person_whatever_the_llm_called_them(self) -> None:
        art = _artifact([{"name": _GUEST, "entity_kind": "object"}])
        assert _role_of(art, _GUEST) == "guest"

    def test_no_stray_object_node_is_left_beside_the_person(self) -> None:
        art = _artifact([{"name": _HOST, "entity_kind": "event"}])
        objects = [n for n in art["nodes"] if n["type"] == "Object"]
        assert not objects, "one human must not be both an Object and a Person"

    def test_no_dangling_host_edge(self) -> None:
        art = _artifact([{"name": _HOST, "entity_kind": "event"}])
        node_ids = {n["id"] for n in art["nodes"]}
        for edge in art["edges"]:
            if edge["type"] in ("HOSTS", "GUESTS_ON"):
                assert edge["from"] in node_ids, f"dangling {edge['type']} from {edge['from']}"

    def test_someone_NOT_a_known_participant_keeps_the_llm_kind(self) -> None:
        # The override is scoped to hosts/guests. A genuine object must stay an object.
        art = _artifact([{"name": "The Norman Conquest", "entity_kind": "event"}])
        objects = [
            n
            for n in art["nodes"]
            if n["type"] == "Object"
            and (n.get("properties") or {}).get("name") == "The Norman Conquest"
        ]
        assert objects, "metadata override must not promote every entity to a person"

    def test_an_organization_sharing_a_hosts_name_stays_an_organization(self) -> None:
        """The over-reach caught by an existing test, pinned here.

        A first version coerced EVERY entity whose name matched a detected host into a person —
        turning a company called "ACME" into a human because the show's host was also "ACME".
        The override is for entities the model could not PLACE (the `object` catch-all), never
        for a positive assertion about a different referent that happens to share a name.
        """
        art = _artifact([{"name": _HOST, "entity_kind": "organization"}])
        orgs = [
            n
            for n in art["nodes"]
            if n["type"] == "Organization" and (n.get("properties") or {}).get("name") == _HOST
        ]
        assert orgs, "a stated organization must survive a same-named host"
