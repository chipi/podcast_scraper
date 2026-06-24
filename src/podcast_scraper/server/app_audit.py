"""Best-effort JSONL audit log for operator write actions (#1071).

Appends one JSON line per mutating operator request. Best-effort: failures never block the
request. Written only when an audit path is configured (``app.state.audit_path``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def append_audit(audit_path: Path | None, record: dict[str, Any]) -> None:
    """Append ``record`` (stamped with ``ts``) to the JSONL audit log; never raises."""
    if audit_path is None:
        return
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": int(time.time()), **record}, ensure_ascii=False)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
