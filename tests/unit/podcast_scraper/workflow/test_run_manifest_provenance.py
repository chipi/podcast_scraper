"""Unit tests for A1 — transcription provenance must name the engine that actually ran.

The manifest has always stamped the resolved transcription model into a field called
``whisper_model``. A Deepgram run therefore recorded ``whisper_model="nova-3"`` — nova-3
being a Deepgram model — which during a provenance audit read as "the wrong profile ran"
and triggered a false alarm. It also makes it impossible to select one engine's episodes
for reprocessing in a mixed Whisper+Deepgram corpus.

The fix is additive: provider-neutral ``transcription_provider`` / ``transcription_model``
are populated on every run, and ``whisper_model`` is retained as a legacy alias so existing
readers and on-disk manifests keep working.
"""

import os
import sys
import unittest

PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper.workflow.run_manifest import RunManifest


class TestRunManifestProvenanceFields(unittest.TestCase):
    def test_provider_neutral_fields_exist(self):
        manifest = RunManifest(
            run_id="r1",
            created_at="2026-08-12T00:00:00Z",
            created_by="tester",
        )
        self.assertTrue(hasattr(manifest, "transcription_provider"))
        self.assertTrue(hasattr(manifest, "transcription_model"))

    def test_legacy_whisper_model_field_is_retained(self):
        """Removing it would break existing readers and previously-written manifests."""
        manifest = RunManifest(
            run_id="r1",
            created_at="2026-08-12T00:00:00Z",
            created_by="tester",
        )
        self.assertTrue(hasattr(manifest, "whisper_model"))

    def test_deepgram_run_records_deepgram_not_whisper(self):
        """The exact scenario that caused the false 'wrong profile' scare."""
        manifest = RunManifest(
            run_id="r1",
            created_at="2026-08-12T00:00:00Z",
            created_by="tester",
            transcription_provider="deepgram",
            transcription_model="nova-3",
            whisper_model="nova-3",
        )
        self.assertEqual(manifest.transcription_provider, "deepgram")
        self.assertEqual(manifest.transcription_model, "nova-3")
        self.assertNotIn(
            "whisper",
            (manifest.transcription_provider or "").lower(),
            "provenance must not imply Whisper when Deepgram transcribed",
        )

    def test_legacy_and_neutral_model_agree(self):
        """While both exist they must never disagree, or audits get a third answer."""
        manifest = RunManifest(
            run_id="r1",
            created_at="2026-08-12T00:00:00Z",
            created_by="tester",
            transcription_model="nova-3",
            whisper_model="nova-3",
        )
        self.assertEqual(manifest.transcription_model, manifest.whisper_model)

    def test_mixed_corpus_can_be_partitioned_by_engine(self):
        """The latent harm A1 describes: targeting one engine for reprocessing.

        With only ``whisper_model`` this is impossible — every row says a model name with
        no indication of which engine produced it.
        """
        rows = [
            RunManifest(
                run_id="a",
                created_at="t",
                created_by="u",
                transcription_provider="deepgram",
                transcription_model="nova-3",
            ),
            RunManifest(
                run_id="b",
                created_at="t",
                created_by="u",
                transcription_provider="whisper",
                transcription_model="large-v3",
            ),
        ]
        deepgram_runs = [r for r in rows if r.transcription_provider == "deepgram"]
        self.assertEqual(len(deepgram_runs), 1)
        self.assertEqual(deepgram_runs[0].run_id, "a")

    def test_defaults_are_none_not_misleading_strings(self):
        manifest = RunManifest(run_id="r", created_at="t", created_by="u")
        self.assertIsNone(manifest.transcription_provider)
        self.assertIsNone(manifest.transcription_model)


if __name__ == "__main__":
    unittest.main()
