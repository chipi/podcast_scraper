"""Models package for podcast scraper.

This package re-exports models from models.py and schemas from schemas/ for backward compatibility.
Note: The actual models (Episode, RssFeed, TranscriptionJob) are in the parent models.py file.
"""

# Import models from models.py file
# We need to import it before this package shadows it
# Use importlib to load models.py directly
import importlib.util
from pathlib import Path

# Import schemas directly (no circular dependency)
from ..schemas.summary_schema import (
    parse_summary_output,
    ParseResult,
    SummarySchema,
    validate_summary_schema,
)

_models_file_path = Path(__file__).parent.parent / "models.py"
if _models_file_path.exists():
    spec = importlib.util.spec_from_file_location("podcast_scraper._models_py", _models_file_path)
    if spec and spec.loader:
        _models_py_module = importlib.util.module_from_spec(spec)
        # Register in sys.modules so dataclass can find it
        import sys

        sys.modules["podcast_scraper._models_py"] = _models_py_module
        # Set __name__ to match what we registered
        _models_py_module.__name__ = "podcast_scraper._models_py"
        spec.loader.exec_module(_models_py_module)

        # Re-export the models
        Episode = _models_py_module.Episode
        RssFeed = _models_py_module.RssFeed
        TranscriptionJob = _models_py_module.TranscriptionJob
    else:
        # Fallback: try to import from parent
        from .. import models as _models_py

        Episode = _models_py.Episode
        RssFeed = _models_py.RssFeed
        TranscriptionJob = _models_py.TranscriptionJob
else:
    raise ImportError(f"Could not find models.py at {_models_file_path}")

__all__ = [
    "Episode",
    "RssFeed",
    "TranscriptionJob",
    "ParseResult",
    "SummarySchema",
    "parse_summary_output",
    "validate_summary_schema",
]
