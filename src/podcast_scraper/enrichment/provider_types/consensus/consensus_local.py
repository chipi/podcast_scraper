"""``consensus_local`` — local composite ConsensusScorer (MiniLM cosine + DeBERTa contradiction).

The production provider for ``topic_consensus`` (ADR-108). Composes a lazy
sentence-transformers encoder (embedding cosine = the shared-question gate) with a
lazy :class:`DeBERTaNliScorer` (max contradiction over both directions = the direction
gate) behind one :class:`NliEmbeddingConsensusScorer`. Both models load on first
``score()`` call, so importing this module stays cheap on ``.[dev]``-only installs.

Per AGENTS.md → "what 'no LLM in CI' means": both are LOCAL models (no paid API), fine
in the ``.[ml]`` jobs; the default ``.[dev]`` CI path uses ``fixed_consensus`` instead.

Params (all optional):

* ``embed_model`` — sentence-transformers model id. Default ``"all-MiniLM-L6-v2"``.
* ``nli_model`` — cross-encoder NLI model id. Default ``"cross-encoder/nli-deberta-v3-small"``.
* ``device`` — torch device (``"cpu"`` / ``"cuda"`` / ``"mps"``) for the encoder.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.provider_types.registry import register_provider_type
from podcast_scraper.enrichment.scorers.consensus import NliEmbeddingConsensusScorer
from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer


class _LazyEncoder:
    """Lazy sentence-transformers encoder — defers the model load to first call."""

    def __init__(self, model_id: str, device: str | None) -> None:
        self._model_id = model_id
        self._device = device
        self._model: Any = None

    def __call__(self, text: str) -> list[float]:
        if self._model is None:
            self._model = self._load()
        return [float(x) for x in self._model.encode(text)]

    def _load(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover — extras not installed
            raise RuntimeError(
                "consensus_local provider requires the [ml] / [search] extra: "
                "pip install -e '.[ml]'"
            ) from exc
        if self._device:
            return SentenceTransformer(self._model_id, device=self._device)
        return SentenceTransformer(self._model_id)


def _make_scorer(params: dict[str, Any]) -> NliEmbeddingConsensusScorer:
    embed_model = params.get("embed_model")
    embed_model = (
        embed_model if isinstance(embed_model, str) and embed_model else "all-MiniLM-L6-v2"
    )
    nli_model = params.get("nli_model")
    device_raw = params.get("device")
    device = device_raw if isinstance(device_raw, str) and device_raw else None
    nli = (
        DeBERTaNliScorer(model_id=nli_model)
        if isinstance(nli_model, str) and nli_model
        else DeBERTaNliScorer()
    )
    return NliEmbeddingConsensusScorer(embed_text=_LazyEncoder(embed_model, device), nli=nli)


register_provider_type(
    name="consensus_local",
    protocol="ConsensusScorer",
    description=(
        "Local composite consensus scorer — MiniLM embedding cosine + DeBERTa "
        "contradiction (lazy-loaded, requires [ml] / [search] extras)."
    ),
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "embed_model": {
                "type": "string",
                "default": "all-MiniLM-L6-v2",
                "description": "sentence-transformers model id for insight cosine.",
            },
            "nli_model": {
                "type": "string",
                "default": "cross-encoder/nli-deberta-v3-small",
                "description": "cross-encoder NLI model id for the contradiction gate.",
            },
            "device": {
                "type": "string",
                "enum": ["cpu", "cuda", "mps"],
                "default": "cpu",
                "description": "torch device for the encoder. Omit to let it pick.",
            },
        },
    },
    factory=_make_scorer,
)


__all__: list[str] = []
