#!/usr/bin/env python3
"""HTTP server for web/gi-kg-viz plus repo file API (enables ?data=… deep links).

Serves static files from --viz-dir (default repo/web/gi-kg-viz) at /. Exposes:
  GET /_api/gi-kg-list?path=RELATIVE   — JSON { "files": [ posix paths under repo-root ] }
  GET /_repo/REL/PATH.gi.json         — raw file bytes (path must stay under --repo-root)

Run from repo root: make serve-gi-kg-viz
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def _safe_under_root(candidate: Path, root: Path) -> Path | None:
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
        return resolved
    except (OSError, ValueError):
        return None


class GiKgVizRequestHandler(BaseHTTPRequestHandler):
    repo_root: Path
    viz_dir: Path

    def _send_json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path_only = unquote(parsed.path)

        if path_only == "/_api/gi-kg-list":
            self._handle_gi_kg_list(parsed)
            return
        if path_only.startswith("/_repo/"):
            rel = path_only[len("/_repo/") :].lstrip("/")
            self._handle_repo_file(rel)
            return

        self._handle_static(path_only)

    def _handle_gi_kg_list(self, parsed) -> None:
        qs = parse_qs(parsed.query or "")
        raw = (qs.get("path") or [""])[0].strip().lstrip("./")
        root = self.repo_root
        if not raw:
            self._send_json({"error": "path query parameter required", "files": []}, 400)
            return
        base = _safe_under_root((root / raw).resolve(), root)
        if base is None or not base.is_dir():
            self._send_json({"error": "invalid or missing directory", "files": []}, 400)
            return
        files: list[str] = []
        root_resolved = root.resolve()
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in filenames:
                low = name.lower()
                if low.endswith(".gi.json") or low.endswith(".kg.json"):
                    full = Path(dirpath) / name
                    try:
                        rel = full.resolve().relative_to(root_resolved)
                    except ValueError:
                        continue
                    files.append(rel.as_posix())
        files.sort()
        self._send_json({"files": files})

    def _handle_repo_file(self, rel_url_path: str) -> None:
        parts = [unquote(p) for p in rel_url_path.split("/") if p and p != ".."]
        candidate = self.repo_root.joinpath(*parts)
        safe = _safe_under_root(candidate, self.repo_root)
        if safe is None or not safe.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = safe.read_bytes()
        ct = "application/json; charset=utf-8"
        if safe.suffix.lower() != ".json":
            ct = "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_static(self, path_only: str) -> None:
        if path_only in ("", "/"):
            path_only = "/index.html"
        rel = path_only.lstrip("/")
        if ".." in rel.split("/"):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        candidate = (self.viz_dir / rel).resolve()
        try:
            candidate.relative_to(self.viz_dir.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        suffix = candidate.suffix.lower()
        ct = "text/html; charset=utf-8"
        if suffix == ".js":
            ct = "text/javascript; charset=utf-8"
        elif suffix == ".css":
            ct = "text/css; charset=utf-8"
        elif suffix == ".json":
            ct = "application/json; charset=utf-8"
        elif suffix == ".svg":
            ct = "image/svg+xml"
        elif suffix in (".png", ".ico", ".webp"):
            ct = f"image/{suffix[1:]}"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser(description="GI/KG viz static server + ?data= repo API")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Static root (default: <repo-root>/web/gi-kg-viz)",
    )
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    viz = (args.viz_dir or repo / "web" / "gi-kg-viz").resolve()
    if not viz.is_dir():
        raise SystemExit(f"viz-dir not found: {viz}")

    class Handler(GiKgVizRequestHandler):
        pass

    Handler.repo_root = repo
    Handler.viz_dir = viz

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GI/KG viz → http://{args.host}:{args.port}/  " f"(repo API ?data=… from {repo})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
