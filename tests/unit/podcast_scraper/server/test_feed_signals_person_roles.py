"""Show-page people carry a role, and the list is ORDERED by it.

Operator, 2026-09-27: "can we on show in people section, as on insights panel for episode, mark ppl
as host/guest/mentioned using same style and make sure that is also order".

The ordering has to happen server-side and that is the part worth testing. `key_people` is truncated
to ``top_k``, so sorting on the client would only rearrange whichever names had already survived a
count-based cut — a show's own host could be absent from the payload entirely. Sorting before the
cut is what makes the badge meaningful.
"""

from __future__ import annotations

from podcast_scraper.server.feed_signals import _accumulate_kg_entities, aggregate_role


def _person(pid: str, name: str, role: str | None) -> dict:
    props: dict[str, object] = {"kind": "person", "name": name}
    if role is not None:
        props["role"] = role
    return {"id": pid, "type": "Entity", "properties": props}


def _kg(*nodes: dict) -> dict:
    return {"nodes": list(nodes)}


def test_roles_are_collected_per_episode_and_aggregated_to_the_strongest() -> None:
    """A person can host one episode of a show and merely be mentioned in the next."""
    person_eps: dict[str, tuple[str, set[str]]] = {}
    roles: dict[str, list[str]] = {}

    _accumulate_kg_entities(
        _kg(_person("person:ada", "Ada", "mentioned")), "e1", {}, person_eps, roles
    )
    _accumulate_kg_entities(_kg(_person("person:ada", "Ada", "host")), "e2", {}, person_eps, roles)

    assert sorted(roles["person:ada"]) == ["host", "mentioned"]
    assert aggregate_role(roles["person:ada"]) == "host"


def test_a_roleless_person_stays_in_the_payload_unbadged() -> None:
    """Older KGs carry no role. Dropping those people would be worse than not badging them —
    an unbadged chip is honest, an omitted person is a silent loss."""
    person_eps: dict[str, tuple[str, set[str]]] = {}
    roles: dict[str, list[str]] = {}

    _accumulate_kg_entities(_kg(_person("person:nora", "Nora", None)), "e1", {}, person_eps, roles)

    assert "person:nora" in person_eps
    assert roles.get("person:nora") is None
    assert aggregate_role([]) is None


def test_role_outranks_episode_count_in_the_ordering() -> None:
    """The defect this closes: the band sorted on footprint alone, so a guest with one more episode
    than the show's own host sat above them on a page titled 'what this show's about'.

    Calls `person_sort_key` — THE SHIPPED comparison — rather than `build_feed_signals`, which
    needs a corpus on disk. The first draft of this test rebuilt the sort expression inline and
    passed cheerfully when the role term was deleted from the real one; that is why the key is now
    a named function instead of a lambda body. A duplicated sort key is not a tested sort key.
    """
    from podcast_scraper.server.feed_signals import person_sort_key

    person_eps = {
        "person:guest": ("Guest", {"e1", "e2", "e3"}),  # MORE episodes
        "person:host": ("Host", {"e1", "e2"}),  # fewer, but hosts
        "person:extra": ("Extra", {"e1", "e2", "e3", "e4"}),  # most, no role at all
    }
    roles = {"person:guest": ["guest"], "person:host": ["host"]}

    ordered = [
        pid
        for pid, _ in sorted(
            person_eps.items(), key=lambda kv: person_sort_key(kv[0], kv[1], roles)
        )
    ]

    assert ordered == ["person:host", "person:guest", "person:extra"], (
        "host must lead despite having the fewest episodes, guest next, and the roleless person "
        "last despite having the most — role first, footprint only as the tiebreak within a role"
    )
