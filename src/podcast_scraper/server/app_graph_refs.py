"""Resolve a highlight to canonical KG graph refs (#1419, PRD-046 FR6 / RFC-111 §3).

The "carry the graph" substrate: a highlight resolves to canonical person/topic references
(mirroring the shipped ``AppEntityRef`` — ``{id: "person:… | topic:…", kind, label}``) so every
outbound surface (recall, digest, share card, the next-arc export) distributes graph nodes, not
flat clips. One resolver, shared by capture (persist-at-save) and the digest assembler.

Granularity (honest, per RFC-111 §3): resolution is **episode-level** — the highlight's episode KG
entities/topics. Char-offset-precise span→entity lift (RFC-072 KL1) is future; this uses what the
shipped per-episode bridge provides and returns ``[]`` cleanly when an episode has no KG.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_slugs import resolve_slug

#: A small cap keeps digests/cards legible and the persisted highlight compact.
DEFAULT_LIMIT = 3


def refs_for_slug(root: Path, slug: str, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, str]]:
    """Canonical person/topic refs for an episode slug (persons first, then topics), capped."""
    if not slug:
        return []
    row = resolve_slug(root, slug)
    if row is None or not row.has_kg:
        return []
    persons, _orgs, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
    refs: list[dict[str, str]] = [{"id": p.id, "kind": "person", "label": p.name} for p in persons]
    refs += [{"id": t.id, "kind": "topic", "label": t.label} for t in topics]
    return refs[:limit]


def refs_for_highlight(root: Path, highlight: dict) -> list[dict[str, str]]:
    """Graph refs for a highlight — its stored refs if present, else episode-level resolution."""
    stored = highlight.get("graph_refs")
    if isinstance(stored, list) and stored:
        return [r for r in stored if isinstance(r, dict) and r.get("id")]
    return refs_for_slug(root, str(highlight.get("episode_slug") or ""))


def carries_the_graph(root: Path, highlight: dict) -> bool:
    """Whether this highlight may be surfaced at all — the moat rule, as a predicate (#38).

    Note what this is NOT: the digest assembler does not call it, because ``_digest_item`` needs
    the refs themselves and drops the item when they come back empty. That is the same condition by
    construction (``not refs`` is exactly ``not carries_the_graph``), not a second implementation —
    and ``test_the_two_gates_agree_for_every_shape_of_highlight`` pins the two together so they
    cannot drift into disagreeing the way the surfaces already once did.

    The slug check below is belt-and-braces, and honestly labelled as such: production never needed
    it, because :func:`refs_for_slug` already returns ``[]`` for an empty slug. It exists so this
    predicate answers from its own argument rather than depending on what a callee happens to do
    three frames down — but sabotaging it changes no test, precisely because that callee still
    guards. It is redundancy, not a fix.

    (What surfaced it was a test STUB that answered a blank slug with refs — kinder than the real
    resolver — which read as the two gates disagreeing. The stub was the defect there, not the
    code; it now mirrors production. A stub more generous than the real thing manufactures failures
    as readily as it hides them.)

    Every revisit surface asks this question, and they used to answer it differently: the digest
    assembler dropped a refless highlight (``_digest_item`` returns None), while ``/resurfacing``
    had no such requirement. Same user, same capture, two answers — the Revisit tab listed moments
    that Your Week and the email silently withheld, which reads as a bug and was how an empty Your
    Week went unexplained on a corpus without per-episode KG.

    Product call (Marko, 2026-08-17): one rule for all three surfaces. The email is a REMINDER of
    the page you would see anyway, so it and Your Week must agree by construction — and the Revisit
    tab must not quietly disagree with both.

    A refless highlight means the episode has no KG, which is a PIPELINE defect rather than a
    normal state — so callers log the drop and corpus validation fails the build. Silence here is
    what let it go unnoticed.
    """
    if not str(highlight.get("episode_slug") or ""):
        return False
    return bool(refs_for_highlight(root, highlight))
