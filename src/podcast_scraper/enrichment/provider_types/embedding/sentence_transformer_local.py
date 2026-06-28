"""``sentence_transformer_local`` — local sentence-transformers EmbeddingProvider.

Wraps a ``SentenceTransformer.encode`` callable in :class:`TopicEmbeddingProvider`.
Model load is LAZY (deferred to first ``embed_text`` call) so importing this
module doesn't trigger sentence-transformers' transitive imports — the
CLI stays importable on ``.[dev]``-only installs.

Params:

* ``model`` — sentence-transformers model id (e.g. ``"all-MiniLM-L6-v2"``,
  ``"all-mpnet-base-v2"``). Required.
* ``device`` — torch device string (``"cpu"`` / ``"cuda"`` / ``"mps"``).
  Defaults to None (sentence-transformers picks).
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.provider_types.registry import register_provider_type
from podcast_scraper.enrichment.scorers.embedding import TopicEmbeddingProvider


class _LazyEncoder:
    """Wrap a sentence-transformers SentenceTransformer behind a lazy load.

    First call triggers the model load; subsequent calls reuse the
    cached instance. Avoids the ``sentence_transformers`` import at
    module top so the CLI doesn't fail on ``.[dev]``-only installs.
    """

    def __init__(self, model_id: str, device: str | None) -> None:
        self._model_id = model_id
        self._device = device
        self._model: Any = None

    def __call__(self, text: str) -> list[float]:
        if self._model is None:
            self._model = self._load()
        vec = self._model.encode(text)
        return [float(x) for x in vec]

    def _load(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover — extras not installed
            raise RuntimeError(
                "sentence_transformer_local provider requires the [ml] or "
                "[search] extra: pip install -e '.[ml]'"
            ) from exc
        if self._device:
            return SentenceTransformer(self._model_id, device=self._device)
        return SentenceTransformer(self._model_id)


def _make_provider(params: dict[str, Any]) -> TopicEmbeddingProvider:
    model = params.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(
            "sentence_transformer_local requires params.model " "(e.g. 'all-MiniLM-L6-v2')"
        )
    device_raw = params.get("device")
    device = device_raw if isinstance(device_raw, str) and device_raw else None
    encoder = _LazyEncoder(model, device)
    return TopicEmbeddingProvider(embed_text=encoder)


register_provider_type(
    name="sentence_transformer_local",
    protocol="EmbeddingProvider",
    description=(
        "Local sentence-transformers checkpoint (lazy-loaded, " "requires [ml] / [search] extras)."
    ),
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["model"],
        "properties": {
            "model": {
                "type": "string",
                "description": "sentence-transformers model id (e.g. 'all-MiniLM-L6-v2').",
            },
            "device": {
                "type": "string",
                "enum": ["cpu", "cuda", "mps"],
                "description": "torch device. Omit to let sentence-transformers pick.",
            },
        },
    },
    factory=_make_provider,
)


__all__: list[str] = []
