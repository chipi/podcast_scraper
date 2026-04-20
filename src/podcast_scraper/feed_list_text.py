"""Shared parsing for RSS URL list files (CLI ``--rss-file`` legacy line lists)."""

from __future__ import annotations


def parse_rss_list_file_lines(text: str) -> list[str]:
    """Return non-empty, non-comment lines (``#`` comments), preserving order and duplicates."""
    urls: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls
