"""A bare single-token speaker must not end up in the artifact twice (#2062).

FOUND BY A FRESH DGX INGEST on 2026-09-13: 2 episodes of "The Journal." produced a gi.json with 116
SPOKEN_BY edges for 59 quotes, because the SAME human was in the graph twice —

    person:unresolved-twiggy-bb004670-...   57 edges
    person:twiggy                           57 edges

WHERE THE RULE LIVES, and where it does NOT. ``identity.bare_name_scope`` is the rule: a
single-token name identifies one person within an episode and nobody globally, so it is scoped to
the episode. It is applied as ONE PASS over the finished payloads — by the pipeline
(``workflow/metadata_generation``) and by the m0007 migration — and deliberately NOT inside any id
mint, because it needs the episode's whole roster and because there are three mint families
(``entity_node_id``, ``person_node_id``, ``identity.slugify.person_id``) that would drift apart if
one of them scoped on its own.

MY FIRST FIX WAS WRONG. I put the scoping inside the GI speaker path's ``_person_node_id``. That
made that one family disagree with the other two, which ``test_entity_identity_invariants`` caught
on ``O'Brien``, ``will.i.am``, ``3Blue1Brown`` and ``Speakman`` — and it pre-empted the pass's
HEALING, which can bind a bare name to a real person's id instead of scoping it.

THE ACTUAL CAUSE is ordering: ``enrich-edges`` runs AFTER the pipeline's scoping pass, so the ids
it minted were never scoped and sat beside the already-scoped ones. That CLI now runs the same
shared pass. These tests therefore assert the ARTIFACT's outcome — the thing a reader sees — not
the return value of one mint function.
"""

from __future__ import annotations

import pytest

from podcast_scraper.identity.bare_name_scope import rewrite_ids

pytestmark = pytest.mark.unit

EP = "bb004670-ae00-11f1-8d2d-8fac93d999e1"


def test_dropping_only_a_copied_edge_still_counts_as_a_change() -> None:
    # Callers persist only when `changes` is non-zero (`_apply_bare_name_scope`), so a pass whose
    # sole effect is dropping a copy must say so, or the copy stays on disk.
    edge = {"type": "SPOKEN_BY", "from": "quote:1", "to": f"person:unresolved-twiggy-{EP}"}
    art = {"nodes": [], "edges": [dict(edge), dict(edge)]}
    out, changes = rewrite_ids(art, {"person:absent": f"person:unresolved-absent-{EP}"})
    assert out["edges"] == [edge]
    assert changes == 1
