"""Every model checkpoint is pinned to a SHA revision (ADR-155).

An unpinned checkpoint is a moving remote archive. Six of the summarization models ship only
``pytorch_model.bin``, so loading one runs a pickle through ``torch.load`` — and until 2026-09-24
those six were also unpinned, which is the combination that matters. The format is what makes
``transformers >= 4.56`` refuse below torch 2.6 (PYSEC-2025-41); the missing pin is what made it
dangerous.

No safetensors version of those weights exists — the Hub's copies are themselves pickle, and
``allenai/led-base-16384-ms2`` is a fine-tune — so pinning is the control, not a format migration.

If this test fails you added a model without a pin. Add one to ``get_pinned_revision_for_model``
rather than relaxing the assertion: the pin is two lines and it is the whole protection.
"""

from __future__ import annotations

import pytest

from podcast_scraper.config_constants import get_pinned_revision_for_model, is_sha_revision
from podcast_scraper.providers.ml.summarizer import DEFAULT_SUMMARY_MODELS

pytestmark = [pytest.mark.unit]

#: Checkpoints loaded outside the summarization alias map — the GIL evidence stack and the
#: embedding model. Listed explicitly so this test covers the whole model surface, not one table.
_EVIDENCE_MODELS = (
    "deepset/roberta-base-squad2",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-small",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/flan-t5-base",
    "google/flan-t5-large",
)


def _all_checkpoints() -> list[str]:
    return sorted({*DEFAULT_SUMMARY_MODELS.values(), *_EVIDENCE_MODELS})


@pytest.mark.parametrize("model_id", _all_checkpoints())
def test_every_checkpoint_has_a_pinned_revision(model_id: str) -> None:
    revision = get_pinned_revision_for_model(model_id)
    assert revision, (
        f"{model_id} has no pinned revision. ADR-155: every checkpoint is pinned, no exceptions "
        "— an unpinned model is a moving remote archive, and most of these are pickles that "
        "torch.load executes at load time."
    )
    assert is_sha_revision(revision), (
        f"{model_id} is pinned to {revision!r}, which is not a 40-hex SHA. A branch name is not "
        "a pin: it moves."
    )


def test_the_alias_map_is_actually_covered() -> None:
    """Guard the guard: a parametrised test over an empty list passes silently."""
    checkpoints = _all_checkpoints()
    assert len(checkpoints) >= 12, (
        f"only {len(checkpoints)} checkpoints discovered — DEFAULT_SUMMARY_MODELS probably moved, "
        "and this test is now asserting almost nothing"
    )
