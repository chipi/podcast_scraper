"""Integration test: ``HFSeq2SeqBackend.generate`` wires the generation config (model mocked).

Moved out of ``tests/unit/podcast_scraper/providers/ml/test_hf_seq2seq_backend.py`` (#1657).
Unlike its siblings there — which mock at the ``transformers.*`` seam and need no ML stack —
this one exercises the real ``generate()`` body, which reaches ``import torch``
(``hf_seq2seq_backend.py:306`` raises ``ModuleNotFoundError`` without it).

``torch`` is not in ``[dev]``, the only extra CI installs for ``test-unit``, so per
``.ai-coding-guidelines.md`` ("Any test that needs FastAPI, httpx, torch, spaCy, lancedb,
etc. belongs in tests/integration/") this belongs in the integration tier. Only this one
class moved; the rest of the original file stays in the unit tier where it correctly runs
without torch.

The model and tokenizer remain mocked — real ML belongs in ``tests/e2e/``.
"""

from __future__ import annotations

from unittest import mock

import pytest

from podcast_scraper.providers.ml.hf_seq2seq_backend import HFSeq2SeqBackend

pytestmark = [pytest.mark.integration, pytest.mark.module_ml_providers]


class TestGenerateWiresGenerationConfig:
    """Verify generate() feeds ``model.generate(**inputs, generation_config=gen_cfg)``
    and returns the decoded string stripped."""

    def test_generate_call_shape(self):
        b = HFSeq2SeqBackend("facebook/bart-base", device="cpu")
        # Simulate a loaded backend with mocked model + tokenizer.
        fake_tokenizer = mock.MagicMock()
        fake_tokenizer.return_value = {"input_ids": mock.Mock(), "attention_mask": mock.Mock()}
        for v in fake_tokenizer.return_value.values():
            v.to = mock.Mock(return_value=v)
        fake_tokenizer.decode = mock.Mock(return_value="  a summary  ")
        fake_model = mock.MagicMock()
        fake_model.config.max_position_embeddings = 1024
        fake_model.parameters = mock.Mock(return_value=iter([mock.Mock(device="cpu")]))
        fake_model.generate = mock.Mock(return_value=[[1, 2, 3]])
        b.model = fake_model
        b.tokenizer = fake_tokenizer
        b._loaded = True

        gen_cfg = mock.Mock()
        result = b.generate("input text", gen_cfg)

        assert result == "a summary"
        fake_model.generate.assert_called_once()
        _, kwargs = fake_model.generate.call_args
        assert kwargs["generation_config"] is gen_cfg
