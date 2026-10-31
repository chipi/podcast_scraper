"""Read one large JSON object entry by entry, in bounded memory.

The search index's sidecar (``search/metadata.json``, one object of ``row id -> row metadata``) is
490 MB on the production corpus, and a whole parse of it holds several GB. ``upgrade verify`` ran
the production host out of memory that way (2026-10-31, D2 item 6). This walks the object with the
standard library's own decoder (``json.JSONDecoder.raw_decode``), one ``key: value`` entry at a
time over a read buffer, and holds only the current entry and the buffer. No new dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection, Iterator, Optional, Tuple

_DECODER = json.JSONDecoder()
_WS = " \t\n\r"
_AFTER_VALUE = _WS + ",:}]"


def iter_json_object_items(
    path: Path | str, *, keys: Optional[Collection[str]] = None, chunk_size: int = 1 << 20
) -> Iterator[Tuple[str, Any]]:
    """``(key, value)`` for each entry of the top-level JSON object in ``path``, in file order.

    ``keys``: when given, only those entries are decoded into values; the others are still parsed
    to find their end, then dropped. Raises ``ValueError`` on a file that is not one JSON object.
    """
    with open(path, encoding="utf-8") as fh:
        reader = _Buffer(fh, chunk_size)
        reader.skip_ws()
        if reader.peek() != "{":
            raise ValueError(f"{path}: not a JSON object")
        reader.pos += 1
        reader.skip_ws()
        if reader.peek() == "}":
            return
        while True:
            key = reader.value()
            if not isinstance(key, str):
                raise ValueError(f"{path}: an object key is not a string")
            reader.skip_ws()
            if reader.peek() != ":":
                raise ValueError(f"{path}: expected ':' after key {key!r}")
            reader.pos += 1
            value = reader.value()
            if keys is None or key in keys:
                yield key, value
            reader.skip_ws()
            nxt = reader.peek()
            reader.pos += 1
            if nxt == "}":
                return
            if nxt != ",":
                raise ValueError(f"{path}: expected ',' or '}}' after key {key!r}")
            reader.skip_ws()


class _Buffer:
    """A text buffer over a file, refilled as the decoder needs, keeping only the unread part."""

    def __init__(self, fh: Any, chunk_size: int) -> None:
        self.fh, self.chunk_size = fh, max(1, chunk_size)
        self.buf, self.pos, self.eof = "", 0, False

    def _fill(self) -> bool:
        if self.eof:
            return False
        more = self.fh.read(self.chunk_size)
        if not more:
            self.eof = True
            return False
        self.buf = self.buf[self.pos :] + more  # drop what has been read
        self.pos = 0
        return True

    def peek(self) -> str:
        while self.pos >= len(self.buf):
            if not self._fill():
                raise ValueError("unexpected end of JSON")
        return self.buf[self.pos]

    def skip_ws(self) -> None:
        while True:
            while self.pos < len(self.buf) and self.buf[self.pos] in _WS:
                self.pos += 1
            if self.pos < len(self.buf) or not self._fill():
                return

    def value(self) -> Any:
        """Decode the next JSON value, reading more until it is complete.

        A value counts only when a delimiter follows it. One cut by the buffer can still decode:
        ``12`` of ``123`` at the buffer's end, or ``-250`` of ``-250.0`` with the ``.`` left over.
        So it is decoded again once more text is read, or the file ends.
        """
        self.skip_ws()
        while True:
            try:
                value, end = _DECODER.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError:
                if self._fill():
                    continue
                raise ValueError("incomplete or invalid JSON value") from None
            if (end == len(self.buf) or self.buf[end] not in _AFTER_VALUE) and self._fill():
                continue
            self.pos = end
            return value
