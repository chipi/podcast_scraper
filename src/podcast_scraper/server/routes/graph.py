"""The graph API's routes (ADR-183): the catalogue and the presets.

- ``GET /api/graph/catalogue``: the engine's constants (``db.graph.catalogue``).
- ``GET /api/graph/presets/{name}``: a preset with its parameters as query strings (a list repeats
  its key), served once per revision and build (``server.graph_presets``) with its ETag; a
  matching ``If-None-Match`` is 304.
- ``POST /api/graph/presets/{name}``: the same preset with a JSON body, for long id lists
  (``items``, ``exclude_items``). It carries the ETag but answers no conditional read: RFC 9110
  makes a failed ``If-None-Match`` on a POST a 412, not a 304 (database, 2026-11-28).

A bad parameter is 422 in FastAPI's ``{detail: [{loc, msg, type}]}`` with ``loc`` naming it
(decision 5); a read past the deadline is 503 with ``retry: narrower`` and no ``Retry-After``
(decision 9). Presets are cached and not counted against decision 12's limit on generic queries.

Not yet: ``POST /api/graph/query``.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

from fastapi import APIRouter, Body, Request, Response
from fastapi.responses import JSONResponse

from podcast_scraper.db import graph
from podcast_scraper.server import graph_presets

router = APIRouter(tags=["graph"])

#: Every preset's parameters besides its own (decision 7); groups.summary takes only its kind.
_COMMON = ("budget", "edge_budget", "exclude_items")
_ALLOWED: dict[str, tuple[str, ...]] = {
    name: params + (_COMMON if name != "groups.summary" else ())
    for name, params in graph.PRESETS.items()
}
_INTS = frozenset({"budget", "edge_budget", "min_items", "expand_groups"})
_LISTS = frozenset({"sources", "content_types", "items", "exclude_items", "with"})
#: A request parameter's name where the engine's keyword differs.
_KWARG = {"with": "with_", "items": "natural_ids", "node": "node_id", "group": "group_id"}


class _Bad(ValueError):
    def __init__(self, loc: list[Any], msg: str) -> None:
        super().__init__(msg)
        self.loc = loc


def _unprocessable(loc: list[Any], msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": [{"loc": loc, "msg": msg, "type": "value_error"}]}
    )


def _kwargs(name: str, raw: dict[str, Any], where: str) -> dict[str, Any]:
    """``raw`` as the preset function's keyword arguments, each parameter checked by name."""
    if name not in _ALLOWED:
        raise _Bad(["path", "name"], f"preset {name!r} is not one of {sorted(_ALLOWED)}")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in _ALLOWED[name]:
            raise _Bad([where, key], f"{name} takes no parameter {key!r}")
        if key in _INTS:
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise _Bad([where, key], f"{key} must be an integer") from None
        elif key in _LISTS:
            if not isinstance(value, list):
                value = [value]
            value = [str(v) for v in value]
        elif not isinstance(value, str):
            raise _Bad([where, key], f"{key} must be a string")
        out[_KWARG.get(key, key)] = value
    for kwarg in _required(name):
        if kwarg not in out:
            param = next((k for k, v in _KWARG.items() if v == kwarg), kwarg)
            raise _Bad([where, param], f"{name} needs {param}")
    return out


def _required(name: str) -> tuple[str, ...]:
    """The preset function's keywords with no default, read from its signature (past ``conn``)."""
    params = list(inspect.signature(graph_presets.FUNCS[name]).parameters.values())[1:]
    return tuple(p.name for p in params if p.default is inspect.Parameter.empty)


def _field_of(msg: str, sent: dict[str, Any]) -> Optional[str]:
    """The parameter an engine refusal names: its message opens with it, or with a count of it."""
    for word in msg.replace("'", " ").split()[:3]:
        if word in sent:
            return word
    return None


def _serve(name: str, raw: dict[str, Any], where: str) -> Any:
    from podcast_scraper.db.engines import corpus_ro_engine

    try:
        params = _kwargs(name, raw, where)
        return graph_presets.serve(corpus_ro_engine(), name, params)
    except _Bad as exc:
        return _unprocessable(exc.loc, str(exc))
    except graph.GraphTimeout:
        return JSONResponse(status_code=503, content={"detail": "deadline", "retry": "narrower"})
    except graph.UnknownPath as exc:
        return _unprocessable([where, "path"], str(exc))
    except graph.UnknownKind as exc:
        return _unprocessable([where, "kind"], str(exc))
    except ValueError as exc:
        field = _field_of(str(exc), raw)
        return _unprocessable([where, field] if field else [where], str(exc))


@router.get("/graph/catalogue")
def graph_catalogue(response: Response) -> dict[str, Any]:
    """What this build's graph engine serves (decision 11). The same for every revision, so it is
    cached as long as the process runs."""
    response.headers["Cache-Control"] = "public, max-age=300"
    return graph.catalogue()


@router.get("/graph/presets/{name}")
def graph_preset(name: str, request: Request) -> Any:
    """A preset over the query string; 304 when ``If-None-Match`` names this revision's answer."""
    raw: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        if key in _LISTS:
            raw.setdefault(key, []).append(value)
        elif key in raw:
            return _unprocessable(["query", key], f"{key} is given more than once")
        else:
            raw[key] = value
    served = _serve(name, raw, "query")
    if isinstance(served, Response):
        return served
    headers = {"ETag": served.etag}
    if served.etag in [t.strip() for t in request.headers.get("if-none-match", "").split(",")]:
        return Response(status_code=304, headers=headers)
    return JSONResponse(content=served.body, headers=headers)


@router.post("/graph/presets/{name}")
def graph_preset_post(name: str, body: dict[str, Any] = Body(default_factory=dict)) -> Any:
    """A preset over a JSON body, for long id lists; it carries the ETag, with no 304."""
    served = _serve(name, body, "body")
    if isinstance(served, Response):
        return served
    return JSONResponse(content=served.body, headers={"ETag": served.etag})
