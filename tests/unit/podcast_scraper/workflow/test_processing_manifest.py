"""Unit tests for the per-episode processing manifest (RFC-109 / ADR-130).

Covers the module contract (stage-block shape, layered versioning, read-modify-write accumulation,
quality-flag dedup, cost roll-up) and the ``episode_processor`` wiring that writes the ASR /
diarization / naming blocks from each stage's own result fields.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow import episode_processor, processing_manifest as pm


@pytest.mark.unit
class TestProcessingManifestModule(unittest.TestCase):
    def _tmp(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        rel = "transcripts/0006 - X.txt"
        return d, rel

    def test_manifest_path_is_sibling_json(self):
        p = pm.manifest_path("/out", "transcripts/0006 - X.txt")
        self.assertEqual(p, "/out/transcripts/0006 - X.manifest.json")

    def test_stage_block_drops_unset_keys_and_rounds(self):
        blk = pm.stage_block(
            ran=True,
            method="whisper",
            model=None,  # unset -> absent (honest: nobody owns it)
            method_version="asr-gate-1",
            duration_s=3.14159,
            metrics={"speech_coverage": 0.935, "missing": None},
        )
        self.assertEqual(blk["ran"], True)
        self.assertEqual(blk["method"], "whisper")
        self.assertNotIn("model", blk)  # None dropped, not defaulted
        self.assertEqual(blk["duration_s"], 3.142)
        self.assertEqual(blk["metrics"], {"speech_coverage": 0.935})  # None metric dropped

    def test_composition_version_deterministic_and_order_independent(self):
        a = pm.pipeline_composition_version(["naming", "asr", "diarization"])
        b = pm.pipeline_composition_version(["asr", "diarization", "naming"])
        self.assertEqual(a, b)  # canonical order, input order irrelevant
        self.assertTrue(a.startswith("pc-"))
        # a different SUBSET of stages is a different composition
        self.assertNotEqual(a, pm.pipeline_composition_version(["asr", "naming"]))

    def test_update_stage_init_then_accumulate(self):
        d, rel = self._tmp()
        pm.update_stage(
            d,
            rel,
            "asr",
            pm.stage_block(ran=True, model="turbo", cost_usd=0.0, method_version="asr-gate-1"),
            episode_id="ep1",
            feed_id="feed1",
            run_id="run1",
            quality_flags=["asr_failover"],
        )
        pm.update_stage(
            d,
            rel,
            "naming",
            pm.stage_block(ran=True, method_version="naming-3", cost_usd=0.0011),
            quality_flags=["empty_host_anchor"],
        )
        data = json.load(open(pm.manifest_path(d, rel)))
        # both stages present after the read-modify-write
        self.assertEqual(set(data["stages"]), {"asr", "naming"})
        self.assertEqual(data["stages"]["asr"]["model"], "turbo")
        self.assertEqual(data["stages"]["naming"]["method_version"], "naming-3")
        # identity captured on init; provenance stamped
        self.assertEqual(data["episode_id"], "ep1")
        self.assertEqual(data["feed_id"], "feed1")
        self.assertEqual(data["run_id"], "run1")
        self.assertEqual(data["schema_version"], pm.MANIFEST_SCHEMA_VERSION)
        self.assertIn("git_sha", data)  # ground-truth backstop key always present
        self.assertIn("git_dirty", data)
        # composition reflects the two stages that ran
        self.assertEqual(
            data["pipeline_composition_version"],
            pm.pipeline_composition_version(["asr", "naming"]),
        )
        # cost rolled up across stages
        self.assertAlmostEqual(data["cost_usd_total"], 0.0011, places=6)
        # flags merged from both writes
        self.assertEqual(set(data["quality_flags"]), {"asr_failover", "empty_host_anchor"})

    def test_quality_flag_dedup_across_rewrites(self):
        d, rel = self._tmp()
        pm.update_stage(d, rel, "asr", pm.stage_block(ran=True), quality_flags=["asr_failover"])
        pm.update_stage(
            d, rel, "asr", pm.stage_block(ran=True), quality_flags=["asr_failover", "asr_failover"]
        )
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertEqual(data["quality_flags"], ["asr_failover"])  # no dup

    def test_no_rel_path_is_noop(self):
        self.assertIsNone(pm.update_stage("/out", "", "asr", pm.stage_block(ran=True)))


@pytest.mark.unit
class TestWriteProcessingManifestWiring(unittest.TestCase):
    """episode_processor._write_processing_manifest builds each block from its stage's result."""

    def _cfg(self, **kw):
        base = dict(
            rss_url="https://example.com/feed.xml",
            run_id="run-xyz",
            transcription_provider="tailnet_dgx_whisper",
            dgx_whisper_model="turbo",
            diarization_provider="pyannote",
            transcription_speech_coverage_min=0.85,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def _job(self):
        from podcast_scraper.models.entities import TranscriptionJob

        return TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)

    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        return d, "transcripts/0006 - X.txt"

    def test_all_three_stage_blocks_written(self):
        d, rel = self._setup()
        result = {
            # ASR (failover episode: actual model differs from configured turbo)
            "asr_speech_coverage": 0.97,
            "model_used": "Systran/faster-whisper-large-v3:speech_coverage_failover",
            "speech_coverage_failover": {"primary_speech_coverage": 0.66},
            # diarization (cloud diarizer reported a billed cost)
            "diarization_num_speakers": 4,
            "diarization_speech_seconds": 1553.0,
            "diarization_cost_usd": 0.004,
            # naming diagnostics (roster's own output)
            "speaker_diagnostics": {
                "summary": {
                    "num_speakers": 4,
                    "named": 3,
                    "unresolved": 1,
                    "truly_unknown": 1,
                    "unattributed_talk_share": 0.42,
                    "unattributed_alarm": True,
                    "by_voice_type": {"host": 1, "guest": 2, "unknown": 1},
                    "unbound_names": ["Jane Doe"],
                    "show_centric": False,
                },
                "voices": [
                    {"voice": "SPEAKER_00", "role": "host", "named": True},
                    {"voice": "SPEAKER_01", "role": "guest", "named": True},
                    {"voice": "SPEAKER_03", "role": "guest", "named": False},
                ],
            },
        }
        episode_processor._write_processing_manifest(
            result,
            self._cfg(),
            self._job(),
            rel,
            d,
            asr_elapsed=12.5,
            asr_call_metrics=SimpleNamespace(estimated_cost=0.008),  # cloud ASR billed cost
        )
        data = json.load(open(pm.manifest_path(d, rel)))

        asr = data["stages"]["asr"]
        self.assertIn("large-v3", asr["model"])  # ACTUAL failover model, not configured turbo
        self.assertEqual(asr["metrics"]["speech_coverage"], 0.97)
        self.assertEqual(asr["duration_s"], 12.5)
        self.assertEqual(asr["cost_usd"], 0.008)  # cloud ASR cost captured
        self.assertTrue(asr["failover"])

        diar = data["stages"]["diarization"]
        self.assertEqual(diar["metrics"]["num_speakers"], 4)
        self.assertEqual(diar["metrics"]["speech_seconds"], 1553.0)
        self.assertEqual(diar["cost_usd"], 0.004)  # cloud diarization cost captured

        # cost rolls up ASR + diarization (naming is local, no cost)
        self.assertAlmostEqual(data["cost_usd_total"], 0.012, places=6)

        naming = data["stages"]["naming"]
        self.assertEqual(naming["metrics"]["named"], 3)
        self.assertEqual(naming["metrics"]["unattributed_talk_share"], 0.42)
        self.assertTrue(naming["metrics"]["host_named"])

        # flags: failover fired; a dominant voice unnamed; a title guest unplaced. Host WAS named
        # and the feed is not show-centric -> no empty_host_anchor.
        flags = set(data["quality_flags"])
        self.assertIn("asr_failover", flags)
        self.assertIn("unnamed_dominant_voice", flags)
        self.assertIn("guest_in_title_not_placed", flags)
        self.assertNotIn("empty_host_anchor", flags)

        # identity + provenance
        self.assertEqual(data["run_id"], "run-xyz")
        self.assertEqual(data["feed_id"], "https://example.com/feed.xml")

    def test_asr_cost_sums_primary_and_failover(self):
        # A cloud ASR that failed over billed TWICE: primary + failover re-transcription.
        d, rel = self._setup()
        result = {
            "asr_speech_coverage": 0.97,
            "model_used": "large-v3:speech_coverage_failover",
            "speech_coverage_failover": {"primary_speech_coverage": 0.6},
            "asr_failover_cost_usd": 0.02,  # second (failover) ASR call
        }
        episode_processor._write_processing_manifest(
            result,
            self._cfg(),
            self._job(),
            rel,
            d,
            asr_call_metrics=SimpleNamespace(estimated_cost=0.01),  # primary call
        )
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertAlmostEqual(data["stages"]["asr"]["cost_usd"], 0.03, places=6)

    def test_empty_host_anchor_flag_when_no_host_named(self):
        d, rel = self._setup()
        result = {
            "asr_speech_coverage": 0.95,
            "speaker_diagnostics": {
                "summary": {
                    "named": 1,
                    "unattributed_alarm": False,
                    "unbound_names": [],
                    "show_centric": False,
                },
                "voices": [{"voice": "SPEAKER_00", "role": "guest", "named": True}],
            },
        }
        episode_processor._write_processing_manifest(result, self._cfg(), self._job(), rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertIn("empty_host_anchor", data["quality_flags"])

    def test_show_centric_host_not_flagged(self):
        d, rel = self._setup()
        result = {
            "asr_speech_coverage": 0.95,
            "speaker_diagnostics": {
                "summary": {
                    "named": 0,
                    "unattributed_alarm": False,
                    "unbound_names": [],
                    "show_centric": True,  # news desk: unnamed host is expected
                },
                "voices": [{"voice": "SPEAKER_00", "role": "host", "named": False}],
            },
        }
        episode_processor._write_processing_manifest(result, self._cfg(), self._job(), rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertNotIn("empty_host_anchor", data["quality_flags"])

    def test_noop_when_no_stage_signals(self):
        d, rel = self._setup()
        episode_processor._write_processing_manifest(
            {"segments": []}, self._cfg(), self._job(), rel, d
        )
        self.assertFalse(os.path.exists(pm.manifest_path(d, rel)))


@pytest.mark.unit
class TestDownstreamManifestBlocks(unittest.TestCase):
    """metadata_generation._write_downstream_manifest_blocks writes summary/GI/KG from each
    stage's own result object, appending to the manifest the transcript-side stages started."""

    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        return d, "transcripts/0006 - X.txt"

    def test_summary_gi_kg_blocks_appended_to_existing_manifest(self):
        from podcast_scraper.workflow import metadata_generation as mg

        d, rel = self._setup()
        # transcript-side stage already wrote an ASR block -> the downstream write must APPEND.
        pm.update_stage(d, rel, "asr", pm.stage_block(ran=True, model="turbo"), episode_id="ep1")

        cfg = SimpleNamespace(summary_provider="gemini", run_id="run-9")
        summary_meta = SimpleNamespace(word_count=210, schema_status="valid")
        call_metrics = SimpleNamespace(estimated_cost=0.0012)
        gi_meta = SimpleNamespace(insight_count=12, schema_version="1.0")
        kg_meta = SimpleNamespace(node_count=41, edge_count=88, schema_version="2.0")

        mg._write_downstream_manifest_blocks(
            output_dir=d,
            transcript_file_path=rel,
            feed_id="feed-1",
            episode_id="ep1",
            cfg=cfg,
            summary_metadata=summary_meta,
            summary_elapsed=4.2,
            summary_call_metrics=call_metrics,
            gi_meta=gi_meta,
            gi_elapsed=6.1,
            gi_cost=0.0031,
            kg_meta=kg_meta,
            kg_elapsed=2.0,
            kg_cost=0.0007,
        )
        data = json.load(open(pm.manifest_path(d, rel)))

        # all SIX stages now present (asr from before + the three downstream)
        self.assertEqual(set(data["stages"]), {"asr", "summary", "gi", "kg"})
        self.assertEqual(data["stages"]["summary"]["metrics"]["word_count"], 210)
        self.assertEqual(data["stages"]["summary"]["method"], "gemini")
        self.assertEqual(data["stages"]["summary"]["cost_usd"], 0.0012)
        self.assertEqual(data["stages"]["gi"]["metrics"]["insight_count"], 12)
        self.assertEqual(data["stages"]["gi"]["cost_usd"], 0.0031)
        self.assertEqual(data["stages"]["kg"]["metrics"]["node_count"], 41)
        self.assertEqual(data["stages"]["kg"]["metrics"]["edge_count"], 88)
        self.assertEqual(data["stages"]["kg"]["cost_usd"], 0.0007)
        # cost rolled up across summary + GI + KG; composition reflects all present stages
        self.assertAlmostEqual(data["cost_usd_total"], 0.0012 + 0.0031 + 0.0007, places=6)
        self.assertEqual(
            data["pipeline_composition_version"],
            pm.pipeline_composition_version(["asr", "summary", "gi", "kg"]),
        )

    def test_gi_all_gated_flag_when_zero_insights(self):
        from podcast_scraper.workflow import metadata_generation as mg

        d, rel = self._setup()
        mg._write_downstream_manifest_blocks(
            output_dir=d,
            transcript_file_path=rel,
            feed_id="f",
            episode_id="e",
            cfg=SimpleNamespace(run_id=None),
            summary_metadata=None,
            summary_elapsed=None,
            summary_call_metrics=None,
            gi_meta=SimpleNamespace(insight_count=0, schema_version="1.0"),
            gi_elapsed=1.0,
            gi_cost=None,
            kg_meta=None,
            kg_elapsed=None,
            kg_cost=None,
        )
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertIn("gi_all_gated", data["quality_flags"])
        self.assertEqual(set(data["stages"]), {"gi"})  # summary/kg absent -> not written

    def test_noop_without_transcript_path(self):
        from podcast_scraper.workflow import metadata_generation as mg

        d, _ = self._setup()
        # No exception, no file — nothing to key the manifest on.
        mg._write_downstream_manifest_blocks(
            output_dir=d,
            transcript_file_path=None,
            feed_id="f",
            episode_id="e",
            cfg=SimpleNamespace(run_id=None),
            summary_metadata=SimpleNamespace(word_count=1, schema_status="valid"),
            summary_elapsed=1.0,
            summary_call_metrics=None,
            gi_meta=None,
            gi_elapsed=None,
            gi_cost=None,
            kg_meta=None,
            kg_elapsed=None,
            kg_cost=None,
        )


@pytest.mark.unit
class TestEpisodeCostProbe(unittest.TestCase):
    """The probe captures THIS episode's GI/KG cost while forwarding everything to the shared
    pipeline_metrics (so run-level totals + all other counters stay correct under parallelism)."""

    class _FakeMetrics:
        """Minimal stand-in mirroring the real metrics recorder surface the probe wraps."""

        def __init__(self):
            self.llm_gi_cost_usd = 0.0
            self.llm_kg_cost_usd = 0.0
            self.gi_evidence_extract_quotes_calls = 0

        def record_llm_gi_call(self, i, o, cost_usd=None):
            if cost_usd:
                self.llm_gi_cost_usd += cost_usd

        def record_llm_gi_evidence_stage_call(self, stage, i, o, cost_usd=None):
            # Mirrors the real impl: substage buckets aside, llm_gi_cost_usd is bumped ONLY via the
            # parent record_llm_gi_call (so the aggregate is counted once).
            self.record_llm_gi_call(i, o, cost_usd=cost_usd)

        def record_llm_kg_call(self, i, o, cost_usd=None):
            if cost_usd:
                self.llm_kg_cost_usd += cost_usd

    def test_captures_per_episode_cost_and_forwards_to_inner(self):
        inner = self._FakeMetrics()
        probe = pm.EpisodeCostProbe(inner)

        probe.record_llm_gi_call(10, 20, cost_usd=0.001)
        probe.record_llm_gi_evidence_stage_call("extract_quotes", 5, 5, cost_usd=0.002)
        probe.record_llm_kg_call(3, 3, cost_usd=0.004)

        # per-episode capture on the probe
        self.assertAlmostEqual(probe.gi_cost_usd, 0.003, places=6)
        self.assertAlmostEqual(probe.kg_cost_usd, 0.004, places=6)
        # inner (shared run-total) still updated. Evidence-stage routes its own record_llm_gi_call
        # to inner, so inner sees the evidence cost once via that path (0.001 + 0.002).
        self.assertAlmostEqual(inner.llm_gi_cost_usd, 0.003, places=6)
        self.assertAlmostEqual(inner.llm_kg_cost_usd, 0.004, places=6)

    def test_attribute_writes_forward_to_inner(self):
        inner = self._FakeMetrics()
        probe = pm.EpisodeCostProbe(inner)
        # The GI/KG builders do `pipeline_metrics.<counter> += 1`; that must land on inner.
        probe.gi_evidence_extract_quotes_calls += 1
        self.assertEqual(inner.gi_evidence_extract_quotes_calls, 1)

    def test_two_probes_isolate_per_episode_cost(self):
        inner = self._FakeMetrics()
        p1 = pm.EpisodeCostProbe(inner)
        p2 = pm.EpisodeCostProbe(inner)
        p1.record_llm_gi_call(1, 1, cost_usd=0.005)
        p2.record_llm_gi_call(1, 1, cost_usd=0.009)
        # each probe sees only its own episode's cost; inner accumulates both (run total)
        self.assertAlmostEqual(p1.gi_cost_usd, 0.005, places=6)
        self.assertAlmostEqual(p2.gi_cost_usd, 0.009, places=6)
        self.assertAlmostEqual(inner.llm_gi_cost_usd, 0.014, places=6)


@pytest.mark.unit
class TestManifestEmitsPipelineStageEvent(unittest.TestCase):
    """P1/o11y: every manifest stage write also emits a canonical `pipeline_stage` event so the
    per-episode quality+cost signal reaches VictoriaLogs/Grafana, not just the sidecar file."""

    def test_update_stage_emits_pipeline_stage_event(self):
        from unittest.mock import patch

        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        rel = "transcripts/0006 - X.txt"
        with patch("podcast_scraper.obs.events.emit_event") as m:
            pm.update_stage(
                d,
                rel,
                "asr",
                pm.stage_block(ran=True, model="turbo", cost_usd=0.01, method_version="asr-gate-1"),
                episode_id="ep1",
                feed_id="feed1",
                run_id="run1",
                quality_flags=["asr_failover"],
            )
        self.assertEqual(m.call_count, 1)
        args, kwargs = m.call_args
        self.assertEqual(args[0], "pipeline_stage")
        self.assertEqual(kwargs["stage"], "asr")
        self.assertEqual(kwargs["episode_id"], "ep1")
        self.assertEqual(kwargs["run_id"], "run1")
        self.assertEqual(kwargs["cost_usd"], 0.01)
        self.assertEqual(kwargs["quality_flags"], ["asr_failover"])

    def test_emit_failure_never_breaks_write(self):
        from unittest.mock import patch

        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        rel = "transcripts/0006 - X.txt"
        with patch("podcast_scraper.obs.events.emit_event", side_effect=RuntimeError("boom")):
            path = pm.update_stage(d, rel, "asr", pm.stage_block(ran=True), episode_id="e")
        self.assertIsNotNone(path)  # write still succeeded despite emit blowing up
        self.assertTrue(os.path.exists(path))


@pytest.mark.unit
class TestAdvisorRegressions(unittest.TestCase):
    """Regression tests for the advisor findings — each exercises the REAL behavior that shipped
    broken (the original tests passed synthetic run_id/ids and never hit these paths)."""

    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        return d, "transcripts/0006 - X.txt"

    def _job(self):
        from podcast_scraper.models.entities import TranscriptionJob

        return TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)

    def _cfg(self, **kw):
        base = dict(rss_url="https://example.com/feed.xml", run_id=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_run_id_sourced_from_correlation_not_cfg(self):
        # advisor #1: cfg.run_id is None (real default); the resolved id lives in correlation.
        # Before the fix, manifest + every pipeline_stage event shipped run_id=null.
        from unittest.mock import patch

        from podcast_scraper.utils import correlation

        d, rel = self._setup()
        correlation.set_run_id("run-REAL-123")
        try:
            with patch("podcast_scraper.obs.events.emit_event") as m:
                episode_processor._write_processing_manifest(
                    {"asr_speech_coverage": 0.95}, self._cfg(), self._job(), rel, d
                )
        finally:
            correlation.set_run_id(None)
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertEqual(data["run_id"], "run-REAL-123")  # NOT null, NOT from cfg
        _, kwargs = m.call_args
        self.assertEqual(kwargs["run_id"], "run-REAL-123")  # the event carries it too

    def test_update_stage_overwrites_run_id_on_rerun(self):
        # advisor #3: a re-run over an existing manifest must not inherit the previous run's id.
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        rel = "transcripts/0006 - X.txt"
        pm.update_stage(d, rel, "asr", pm.stage_block(ran=True), run_id="run-1", episode_id="ep")
        pm.update_stage(d, rel, "asr", pm.stage_block(ran=True), run_id="run-2", episode_id="ep")
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertEqual(data["run_id"], "run-2")  # overwritten, not backfilled to run-1

    def test_pipeline_stage_event_omits_unstable_composition_version(self):
        # advisor #7: composition_version is recomputed per-RMW → must NOT be in per-stage event.
        from unittest.mock import patch

        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        rel = "transcripts/0006 - X.txt"
        with patch("podcast_scraper.obs.events.emit_event") as m:
            pm.update_stage(d, rel, "asr", pm.stage_block(ran=True), run_id="r", episode_id="e")
        _, kwargs = m.call_args
        self.assertNotIn("pipeline_composition_version", kwargs)
        # but it IS still in the file (final value)
        self.assertIn("pipeline_composition_version", json.load(open(pm.manifest_path(d, rel))))

    def test_bind_correlation_clears_id_when_job_has_no_episode(self):
        # advisor #9: a no-episode job must CLEAR the id, not leak the previous episode's id.
        from podcast_scraper.utils import correlation

        correlation.set_episode_id("prev-episode")
        try:
            episode_processor._bind_episode_correlation(self._job(), self._cfg())
            self.assertIsNone(correlation.get_episode_id())  # cleared, not "prev-episode"
        finally:
            correlation.set_episode_id(None)

    def test_relabel_shaped_result_writes_only_naming_block(self):
        # advisor #2: relabel produces a result with naming diagnostics but no ASR/diarization
        # fields — the manifest must still gain a (bumped) naming block so the reprocess query moves
        d, rel = self._setup()
        result = {
            "speaker_diagnostics": {
                "summary": {
                    "named": 2,
                    "unattributed_alarm": False,
                    "unbound_names": [],
                    "show_centric": False,
                },
                "voices": [{"voice": "SPEAKER_00", "role": "host", "named": True}],
            }
        }
        episode_processor._write_processing_manifest(result, self._cfg(), self._job(), rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        self.assertEqual(set(data["stages"]), {"naming"})  # only naming, no asr/diarization
        self.assertEqual(data["stages"]["naming"]["method_version"], pm.METHOD_VERSIONS["naming"])
