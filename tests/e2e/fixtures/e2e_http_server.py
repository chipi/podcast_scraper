"""E2E HTTP server for serving test fixtures.

This module provides a local HTTP server that serves RSS feeds, audio files,
and transcripts from the test fixtures directory. The server is designed for
E2E tests to use real HTTP clients with local fixture data.

Key Features:
- Serves RSS feeds, audio files, and transcripts
- GET and HEAD for static routes (HEAD avoids 404 when checking media size)
- Supports HTTP range requests (206 Partial Content) for streaming
- Path traversal protection
- URL mapping for flat fixture structure
- Configurable error scenarios
- Mock OpenAI API endpoints for E2E testing

Port notes:
- The ``e2e_server`` pytest fixture binds an ephemeral port (``0``), not a fixed port.
- Docstring examples use **18765**, matching ``make serve-e2e-mock`` default
  (**E2E_MOCK_PORT**), so they are not confused with FastAPI ``serve-api`` (**8000**).
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import re
import socketserver
import threading
import time
from email import message_from_bytes
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import pytest

from podcast_scraper import config

logger = logging.getLogger(__name__)


def _read_fixture_version() -> str:
    version_file = Path(__file__).resolve().parents[2] / "fixtures" / "FIXTURES_VERSION"
    return version_file.read_text(encoding="utf-8").strip()


_FIXTURE_VERSION = _read_fixture_version()
_VERSIONED_SUBDIRS: frozenset[str] = frozenset({"audio", "transcripts"})


def _anthropic_system_text(system: Any) -> str:
    """Normalize Anthropic `system` (str or list of content blocks) to plain text."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        chunks: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
            elif isinstance(block, str):
                chunks.append(block)
        return " ".join(chunks)
    return str(system)


def _bundled_gil_json(text: str) -> Optional[str]:
    """Shared #698 bundled-GIL response for EVERY provider mock (gemini/openai/anthropic).

    The bundled extract_quotes / score_entailment calls send an index-numbered prompt and
    expect a JSON object keyed by index — a shape the staged handlers don't emit. Returning
    it here uniformly keeps the bundled path exercised for ALL providers, not just whichever
    one an acceptance arm happens to use; without it the bundled call fails and silently
    degrades to the staged fallback, hiding the coverage gap (the Gemini incident).
    """
    if "Return JSON only." not in text:
        return None
    if "Insights:" in text and "Transcript (excerpt):" in text:
        # extract_quotes_bundled -> {index: [verbatim transcript snippet]} so quotes ground.
        transcript = text.split("Transcript (excerpt):")[1].split("Insights:")[0].strip()
        snippet = transcript[:60].strip() or "Evidence from transcript."
        block = text.split("Insights:")[1].split("Return JSON only.")[0]
        idxs = re.findall(r"^\s*(\d+):", block, re.MULTILINE) or ["0"]
        return json.dumps({i: [snippet] for i in idxs})
    if "Pairs:" in text and "premise:" in text:
        # score_entailment_bundled -> {index: support score in [0,1]}.
        block = text.split("Pairs:")[1].split("Return JSON only.")[0]
        idxs = re.findall(r"^\s*(\d+):", block, re.MULTILINE) or ["0"]
        return json.dumps({i: 0.9 for i in idxs})
    return None


def _gemini_text_response_data(request_data: Dict[str, Any], text_prompt: str) -> Dict[str, Any]:
    """Build Gemini generateContent response body for non-audio text requests."""
    generation_config = request_data.get("generationConfig", {})
    response_mime_type = generation_config.get("response_mime_type", "")

    # #698 bundled GIL evidence stack (shared across every provider mock).
    bundled_json = _bundled_gil_json(text_prompt)
    if bundled_json is not None:
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": bundled_json}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 40,
                "totalTokenCount": 140,
            },
        }

    if "Insight:" in text_prompt and (
        "quote_text" in text_prompt or "Transcript (excerpt):" in text_prompt
    ):
        if "Transcript (excerpt):" in text_prompt:
            excerpt = text_prompt.split("Transcript (excerpt):")[1].split("Insight:")[0]
        else:
            excerpt = text_prompt.split("Insight:")[0]
        excerpt = excerpt.strip()
        mock_quote = (
            excerpt[:60].strip() if len(excerpt) > 10 else (excerpt or "Evidence from transcript.")
        )
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps({"quote_text": mock_quote})}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 80,
                "candidatesTokenCount": 20,
                "totalTokenCount": 100,
            },
        }
    if "Premise:" in text_prompt and "Hypothesis:" in text_prompt:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "0.85"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 60,
                "candidatesTokenCount": 2,
                "totalTokenCount": 62,
            },
        }

    gc = request_data.get("generationConfig") or {}
    si_raw = gc.get("system_instruction") or gc.get("systemInstruction")
    si_blob = ""
    if isinstance(si_raw, str):
        si_blob = si_raw
    tsi = request_data.get("systemInstruction")
    if isinstance(tsi, dict):
        for p in tsi.get("parts") or []:
            if isinstance(p, dict) and "text" in p:
                si_blob += " " + str(p["text"])
    tp_low = text_prompt.lower()
    kg_user_heuristic = (
        "extract up to" in tp_low and "**topics**" in text_prompt and "transcript:" in tp_low
    )
    if "knowledge-graph fragment" in si_blob.lower() or kg_user_heuristic:
        kg_json = {
            "topics": [{"label": "E2E mock topic"}],
            "entities": [{"name": "E2E Entity", "entity_kind": "person"}],
        }
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(kg_json)}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 40,
                "totalTokenCount": 140,
            },
        }
    if response_mime_type == "application/json":
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "speakers": ["Host", "Guest"],
                                        "hosts": ["Host"],
                                        "guests": ["Guest"],
                                    }
                                )
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        }

    summary_length = min(200, len(text_prompt) // 10)
    summary = f"This is a test summary from Gemini. {text_prompt[:summary_length]}..."
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": summary}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150,
        },
    }


class E2EServerURLs:
    """URL helper class for E2E server."""

    def __init__(self, base_url: str):
        """Initialize URL helper.

        Args:
            base_url: Base URL of the E2E server (e.g., "http://127.0.0.1:18765")
        """
        self.base_url = base_url.rstrip("/")

    def feed(self, podcast_name: str) -> str:
        """Get RSS feed URL for a podcast.

        Args:
            podcast_name: Podcast name (e.g., "podcast1")

        Returns:
            Full RSS feed URL (e.g., "http://127.0.0.1:18765/feeds/podcast1/feed.xml")
        """
        return f"{self.base_url}/feeds/{podcast_name}/feed.xml"

    def audio(self, episode_id: str) -> str:
        """Get audio file URL for an episode.

        Args:
            episode_id: Episode ID (e.g., "p01_e01")

        Returns:
            Full audio URL (e.g., "http://127.0.0.1:18765/audio/p01_e01.mp3")
        """
        return f"{self.base_url}/audio/{episode_id}.mp3"

    def transcript(self, episode_id: str) -> str:
        """Get transcript file URL for an episode.

        Args:
            episode_id: Episode ID (e.g., "p01_e01")

        Returns:
            Full transcript URL (e.g., "http://127.0.0.1:18765/transcripts/p01_e01.txt")
        """
        return f"{self.base_url}/transcripts/{episode_id}.txt"

    def base(self) -> str:
        """Get base URL of the server.

        Returns:
            Base URL (e.g., "http://127.0.0.1:18765")
        """
        return self.base_url

    def openai_api_base(self) -> str:
        """Get OpenAI API base URL (points to E2E server).

        Returns:
            OpenAI API base URL (e.g., "http://127.0.0.1:18765/v1")
        """
        return f"{self.base_url}/v1"

    def gemini_api_base(self) -> str:
        """Get Gemini API base URL (points to E2E server).

        google-genai joins ``api_version`` (``v1beta``) onto ``base_url``. Do not
        append ``/v1beta`` here or requests hit ``.../v1beta/v1beta/models/...`` (404).

        Returns:
            Server root (e.g., ``http://127.0.0.1:18765``)
        """
        return self.base_url

    def mistral_api_base(self) -> str:
        """Get Mistral API base URL (points to E2E server).

        Mistral Python SDK appends ``/v1/...`` to ``server_url``. Do not include
        ``/v1`` here or requests become ``/v1/v1/chat/completions`` (404).

        Mock server paths: ``/v1/chat/completions``, ``/v1/audio/transcriptions``.

        Returns:
            Server root (e.g., ``http://127.0.0.1:18765``)
        """
        return self.base_url

    def grok_api_base(self) -> str:
        """Get Grok API base URL (points to E2E server).

        Grok uses OpenAI-compatible API format, so it uses the same endpoints:
        - /v1/chat/completions for chat models (speaker detection, summarization)
        - Note: Grok does NOT support audio transcription

        Returns:
            Grok API base URL (e.g., "http://127.0.0.1:18765/v1")
        """
        return f"{self.base_url}/v1"

    def deepseek_api_base(self) -> str:
        """Get DeepSeek API base URL (points to E2E server).

        DeepSeek uses OpenAI-compatible API format, so it uses the same endpoints:
        - /v1/chat/completions for chat models (speaker detection, summarization)
        - Note: DeepSeek does NOT support audio transcription

        Returns:
            DeepSeek API base URL (e.g., "http://127.0.0.1:18765/v1")
        """
        return f"{self.base_url}/v1"

    def ollama_api_base(self) -> str:
        """Get Ollama API base URL (points to E2E server).

        Ollama uses OpenAI-compatible API format, so it uses the same endpoints:
        - /v1/chat/completions for chat models (speaker detection, summarization)
        - Note: Ollama does NOT support audio transcription

        Returns:
            Ollama API base URL (e.g., "http://127.0.0.1:18765/v1")
        """
        return f"{self.base_url}/v1"

    def dgx_host_port(self) -> tuple[str, int]:
        """``(host, port)`` for the DGX tailnet clients (#954).

        The TailnetDgx{Whisper,Diarization} providers build ``http://{host}:{port}``
        themselves and append ``/v1/audio/transcriptions`` / ``/v1/diarize`` /
        ``/v1/models``, so callers point ``dgx_tailnet_host`` + the port fields at
        the e2e server rather than at a real GB10 box.
        """
        parsed = urlparse(self.base_url)
        return parsed.hostname or "127.0.0.1", int(parsed.port or 80)

    def anthropic_api_base(self) -> str:
        """Get Anthropic API base URL (points to E2E server).

        Anthropic uses its own API format:
        - /v1/messages for chat models (speaker detection, summarization)
        - Note: Anthropic does NOT support native audio transcription
        - Note: Anthropic SDK appends /v1/messages, so base URL should NOT include /v1

        Returns:
            Anthropic API base URL (e.g., "http://127.0.0.1:18765")
        """
        return self.base_url

    def deepgram_api_base(self) -> str:
        """Get Deepgram API base URL (points to E2E server).

        Deepgram's pre-recorded transcription SDK posts ``{base}/v1/listen``, so
        the base URL must NOT include ``/v1`` (the SDK appends it).

        Returns:
            Deepgram API base URL (e.g., "http://127.0.0.1:18765")
        """
        return self.base_url


class E2EHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for E2E server.

    This handler serves files from the test fixtures directory with URL mapping
    to support the flat fixture structure. Supports error scenarios for testing.
    """

    # Mapping: podcast name -> RSS filename
    PODCAST_RSS_MAP = {
        "podcast1": "p01_mtb.xml",
        # Same RSS as podcast1; use this slug when fast-fixture mode must not swap p01 → p01_fast
        # (e.g. acceptance runner multi-feed slot 0 — end-user samples stay generic 5 placeholders).
        "podcast1_mtb": "p01_mtb.xml",
        "podcast2": "p02_software.xml",
        "podcast3": "p03_scuba.xml",
        "podcast4": "p04_photo.xml",
        "podcast5": "p05_investing.xml",
        "edgecases": "p06_edge_cases.xml",
        # Multi-episode test feed with 5 short episodes (10-15 seconds each)
        "podcast1_multi_episode": "p01_multi.xml",
        # Three items, newest-first, all Path-1 transcripts (episode order / date filter E2E)
        "podcast1_episode_selection": "p01_episode_selection.xml",
        # Solo speaker podcast (host only, no guests)
        "podcast9_solo": "p09_biohacking.xml",
        # Long-form episodes for summarization threshold testing (Issue #283)
        "podcast7_sustainability": "p07_sustainability.xml",  # ~15k words, ~3.6k-4k tokens
        "podcast8_solar": "p08_solar.xml",  # ~20k words, ~4.8k-5k tokens
    }

    # Fast mode RSS mapping (uses shorter episodes for faster tests)
    PODCAST_RSS_MAP_FAST = {
        # Fast version with 1-minute episode (Path 2: Transcription)
        "podcast1": "p01_fast.xml",
        # Full p01 (3 items) while fast mode on; acceptance multi-feed matches p02–p05 depth
        "podcast1_mtb": "p01_mtb.xml",
        # Fast version with transcript URL (Path 1: Download)
        "podcast1_with_transcript": "p01_fast_with_transcript.xml",
        # Multi-episode test feed (also available in fast mode)
        "podcast1_multi_episode": "p01_multi.xml",
        "podcast1_episode_selection": "p01_episode_selection.xml",
        # Solo speaker podcast (host only, no guests) - available in fast mode
        "podcast9_solo": "p09_biohacking.xml",
        # Long-form episodes for summarization threshold testing (Issue #283)
        # Note: These are long episodes, but needed for threshold validation
        "podcast7_sustainability": "p07_sustainability.xml",  # ~15k words, ~3.6k-4k tokens
        "podcast8_solar": "p08_solar.xml",  # ~20k words, ~4.8k-5k tokens
    }

    # Allowed podcasts (shared across all handler instances)
    # If None, all podcasts are allowed. If set, only podcasts in this set are served.
    _allowed_podcasts: Optional[set[str]] = None
    _allowed_podcasts_lock = threading.Lock()

    # Use fast fixtures flag (shared across all handler instances)
    # When True, use PODCAST_RSS_MAP_FAST for available podcasts
    # When False, always use PODCAST_RSS_MAP (full fixtures)
    _use_fast_fixtures: bool = True
    _use_fast_fixtures_lock = threading.Lock()

    # Error behavior registry (shared across all handler instances)
    # Format: {url_path: {"status": 404|500, "delay": seconds}}
    _error_behaviors: Dict[str, Dict[str, Any]] = {}
    _error_behaviors_lock = threading.Lock()

    # Transient error registry: fail N times then succeed.
    # Format: {url_path: {"status": int, "fail_count": int, "hits": int}}
    _transient_errors: Dict[str, Dict[str, Any]] = {}
    _transient_errors_lock = threading.Lock()

    # Guardrail-violation injection registry (#999 / ADR-099).
    # Format: {url_path: violation_type}
    # Per-route violation_type vocabulary:
    #   /v1/audio/transcriptions:  "transcription:empty" | "transcription:length_floor"
    #   /v1/diarize:               "diarize:empty_segments"
    #   /v1/chat/completions:      "chat:empty_content" | "chat:thinking_prose"
    #                              | "chat:bad_json" | "chat:finish_length"
    #   /v1/messages:              "anthropic:empty_content" | "anthropic:thinking_prose"
    #                              | "anthropic:max_tokens"
    #   /v1beta/generateContent:   "gemini:empty_content" | "gemini:thinking_prose"
    #                              | "gemini:max_tokens"
    #   /api/generate:             "generate:empty" | "generate:thinking_prose"
    # Tests set an injection, run the pipeline, assert fallback fires.
    # Cleared at fixture teardown so tests don't leak state to siblings.
    _injected_violations: Dict[str, str] = {}
    _injected_violations_lock = threading.Lock()

    @classmethod
    def set_error_behavior(cls, url_path: str, status: int, delay: float = 0.0):
        """Set error behavior for a specific URL path.

        Args:
            url_path: URL path to apply error to (e.g., "/feeds/podcast1/feed.xml")
            status: HTTP status code to return (e.g., 404, 500)
            delay: Optional delay in seconds before responding
        """
        with cls._error_behaviors_lock:
            cls._error_behaviors[url_path] = {"status": status, "delay": delay}

    @classmethod
    def set_transient_error(cls, url_path: str, status: int, fail_count: int):
        """Make a URL fail ``fail_count`` times, then serve normally.

        Args:
            url_path: URL path (e.g., "/feeds/podcast1/transcripts/ep1.txt")
            status: HTTP status code to return during failures (e.g., 429, 503)
            fail_count: Number of requests that should fail before succeeding
        """
        with cls._transient_errors_lock:
            cls._transient_errors[url_path] = {
                "status": status,
                "fail_count": fail_count,
                "hits": 0,
            }

    @classmethod
    def clear_error_behavior(cls, url_path: str):
        """Clear error behavior for a specific URL path.

        Args:
            url_path: URL path to clear error behavior for
        """
        with cls._error_behaviors_lock:
            cls._error_behaviors.pop(url_path, None)

    @classmethod
    def clear_all_error_behaviors(cls):
        """Clear all error behaviors (permanent and transient)."""
        with cls._error_behaviors_lock:
            cls._error_behaviors.clear()
        with cls._transient_errors_lock:
            cls._transient_errors.clear()

    @classmethod
    def inject_violation(cls, url_path: str, violation_type: str) -> None:
        """Inject a guardrail-violating response on the next request to ``url_path``.

        For #999 / ADR-099 E2E tests. The handler for ``url_path`` checks the
        injection registry; if a violation is set, it returns a structurally-
        valid HTTP 200 response whose content fails the per-service guardrail
        check, then clears the injection (one-shot per call). The test asserts
        the consumer's fallback path fired.

        See ``_injected_violations`` class attribute for the vocabulary of
        ``violation_type`` values per route.
        """
        with cls._injected_violations_lock:
            cls._injected_violations[url_path] = violation_type

    @classmethod
    def clear_violations(cls) -> None:
        """Clear all guardrail-violation injections (call at fixture teardown)."""
        with cls._injected_violations_lock:
            cls._injected_violations.clear()

    @classmethod
    def _pop_injected_violation(cls, url_path: str) -> Optional[str]:
        """Atomic check-and-pop: returns the injected violation_type for
        ``url_path`` (and removes it) or None. Used by per-route handlers
        to enforce one-shot semantics — the same route's next call returns
        a normal mock response unless the test re-injects.
        """
        with cls._injected_violations_lock:
            return cls._injected_violations.pop(url_path, None)

    @classmethod
    def set_allowed_podcasts(cls, podcasts: Optional[set[str]]):
        """Set allowed podcasts for serving RSS feeds.

        Args:
            podcasts: Set of podcast names to allow, or None to allow all podcasts.
                     In fast mode, this should be {"podcast1"} to limit to one feed.
                     In full mode, this should be None to allow all feeds.
        """
        with cls._allowed_podcasts_lock:
            cls._allowed_podcasts = podcasts

    @classmethod
    def set_use_fast_fixtures(cls, use_fast: bool):
        """Set whether to use fast fixtures (shorter episodes) for RSS feeds.

        Args:
            use_fast: If True, use PODCAST_RSS_MAP_FAST for available podcasts.
                     If False, always use PODCAST_RSS_MAP (full fixtures).
                     Default is True for backward compatibility.
        """
        with cls._use_fast_fixtures_lock:
            cls._use_fast_fixtures = use_fast

    @classmethod
    def get_allowed_podcasts(cls) -> Optional[set[str]]:
        """Get currently allowed podcasts.

        Returns:
            Set of allowed podcast names, or None if all podcasts are allowed.
        """
        with cls._allowed_podcasts_lock:
            return cls._allowed_podcasts

    @classmethod
    def get_fixture_root(cls) -> Path:
        """Get the root directory for test fixtures.

        Returns:
            Path to tests/fixtures directory
        """
        # This file is in tests/e2e/fixtures/
        # Fixtures are in tests/fixtures/
        # Path structure: tests/e2e/fixtures/e2e_http_server.py
        #                 -> tests/e2e/fixtures (parent)
        #                 -> tests/e2e (parent.parent)
        #                 -> tests (parent.parent.parent)
        #                 -> tests/fixtures (parent.parent.parent / "fixtures")
        current_file = Path(__file__).resolve()
        # Go up: e2e_http_server.py -> fixtures -> e2e -> tests
        tests_dir = current_file.parent.parent.parent
        fixture_root = tests_dir / "fixtures"
        return fixture_root

    def do_GET(self):
        """Handle GET requests with URL mapping."""
        path = self.path.split("?")[0]  # Remove query string
        self._dispatch_http_get(path, head_only=False)

    def do_HEAD(self):
        """Handle HEAD requests (same routes as GET; no response body).

        Downloader checks media size with HEAD; without this, fixture URLs 404.
        """
        path = self.path.split("?")[0]  # Remove query string
        self._dispatch_http_get(path, head_only=True)

    def _dispatch_http_get(self, path: str, head_only: bool) -> None:
        """Handle GET or HEAD for RSS, static fixtures, and Ollama discovery routes."""
        # Check transient errors first (fail N times, then fall through)
        with self._transient_errors_lock:
            transient = self._transient_errors.get(path)
            if transient is not None:
                transient["hits"] += 1
                if transient["hits"] <= transient["fail_count"]:
                    status = transient["status"]
                    logger.debug(
                        "Transient error %d for %s (hit %d/%d)",
                        status,
                        path,
                        transient["hits"],
                        transient["fail_count"],
                    )
                    self.send_error(
                        status,
                        f"Transient {status} (hit "
                        f"{transient['hits']}/{transient['fail_count']})",
                    )
                    return
                # Past fail_count: fall through to normal serving

        # Check for permanent error behavior
        with self._error_behaviors_lock:
            error_behavior = self._error_behaviors.get(path)
        if error_behavior:
            # Apply delay if specified
            if error_behavior.get("delay", 0) > 0:
                time.sleep(error_behavior["delay"])
            # Return error status
            self.send_error(
                error_behavior["status"],
                f"Simulated {error_behavior['status']} error",
            )
            return

        # Route 1: RSS feeds
        # /feeds/podcast1/feed.xml -> rss/p01_mtb.xml
        if path.startswith("/feeds/") and path.endswith("/feed.xml"):
            # Extract podcast_name from URL path
            path_parts = path.split("/")
            if len(path_parts) < 3:
                self.send_error(400, "Invalid URL format")
                return
            podcast_name = path_parts[2]  # Extract "podcast1"

            # Validate podcast_name to prevent path injection
            # Even though it's only used as a dictionary key, validate for defense in depth
            if not podcast_name or not isinstance(podcast_name, str):
                self.send_error(400, "Invalid podcast name")
                return
            # Reject podcast_name containing path separators or ".."
            if "/" in podcast_name or "\\" in podcast_name or ".." in podcast_name:
                self.send_error(400, "Invalid podcast name")
                return
            # Reject podcast_name with whitespace or special characters
            if (
                podcast_name.strip() != podcast_name
                or not podcast_name.replace("_", "").replace("-", "").isalnum()
            ):
                self.send_error(400, "Invalid podcast name")
                return

            # Check if podcast is allowed (fast mode limitation)
            with self._allowed_podcasts_lock:
                allowed = self._allowed_podcasts
            if allowed is not None and podcast_name not in allowed:
                self.send_error(404, f"RSS feed not available in current test mode: {podcast_name}")
                return

            # Use fast RSS feed if fast fixtures mode is enabled
            # Fast mode uses shorter episodes for faster tests
            # Data quality/nightly mode uses original mock data with full episodes
            with self._use_fast_fixtures_lock:
                use_fast = self._use_fast_fixtures

            # Only use fast fixtures if explicitly enabled AND podcast is in fast map
            if use_fast and podcast_name in self.PODCAST_RSS_MAP_FAST:
                rss_file = self.PODCAST_RSS_MAP_FAST.get(podcast_name)
            else:
                # Use original mock data (for data quality tests or slow tests)
                rss_file = self.PODCAST_RSS_MAP.get(podcast_name)

            if rss_file:
                file_path = self._get_safe_rss_path(rss_file)
                if file_path is None:
                    self.send_error(403, "Invalid RSS file path")
                    return
                self._serve_file(file_path, content_type="application/xml", head_only=head_only)
                return
            self.send_error(404, "RSS feed not found")
            return

        # Route 2: Direct flat URLs for audio
        # /audio/p01_e01.mp3 -> audio/p01_e01.mp3
        if path.startswith("/audio/"):
            filename = path.split("/")[-1]  # Extract "p01_e01.mp3"
            # Validate filename first (returns None if invalid)
            # This validation follows CodeQL recommendations:
            # - No path separators, no "..", exactly one dot, allowlist pattern
            validation_result = self._validate_and_sanitize_filename("audio", filename)
            if validation_result is None:
                self.send_error(403, "Path traversal not allowed")
                return
            # Unpack validated values (these have passed all security checks)
            validated_subdir, validated_filename = validation_result
            # Use validated values for path construction (not original filename)
            # validated_filename is safe: validated according to CodeQL recommendations
            file_path = self._get_safe_fixture_path(validated_subdir, validated_filename)
            if file_path is None:
                # File doesn't exist (validation passed but file not found)
                self.send_error(404, "File not found")
                return
            self._serve_file(
                file_path,
                content_type="audio/mpeg",
                support_range=True,
                head_only=head_only,
            )
            return

        # Route 3: Direct flat URLs for transcripts
        # /transcripts/p01_e01.txt -> transcripts/p01_e01.txt
        if path.startswith("/transcripts/"):
            filename = path.split("/")[-1]  # Extract "p01_e01.txt"
            # Validate filename first (returns None if invalid)
            # This validation follows CodeQL recommendations:
            # - No path separators, no "..", exactly one dot, allowlist pattern
            validation_result = self._validate_and_sanitize_filename("transcripts", filename)
            if validation_result is None:
                self.send_error(403, "Path traversal not allowed")
                return
            # Unpack validated values (these have passed all security checks)
            validated_subdir, validated_filename = validation_result
            # Use validated values for path construction (not original filename)
            # validated_filename is safe: validated according to CodeQL recommendations
            file_path = self._get_safe_fixture_path(validated_subdir, validated_filename)
            if file_path is None:
                # File doesn't exist (validation passed but file not found)
                self.send_error(404, "File not found")
                return
            self._serve_file(file_path, content_type="text/plain", head_only=head_only)
            return

        # Route 4: Ollama API endpoints (for health checks and model validation)
        # /api/version -> Ollama health check
        if path == "/api/version":
            self._handle_ollama_version(head_only=head_only)
            return

        # /api/tags -> Ollama model list (for model validation)
        if path == "/api/tags":
            self._handle_ollama_tags(head_only=head_only)
            return

        # DGX faster-whisper (:8002) + pyannote (:8001) health probe (#954).
        # Both clients GET /v1/models; pyannote also answers /health.
        if path in ("/v1/models", "/health"):
            self._handle_dgx_probe(path, head_only=head_only)
            return

        # 404 for all other paths
        self.send_error(404, "File not found")

    def do_POST(self):
        """Handle POST requests for API endpoints."""
        path = self.path.split("?")[0]  # Remove query string
        # Debug: log the path being requested
        logger.debug("E2E server POST request to path: %s", path)

        # Check for error behavior first
        with self._error_behaviors_lock:
            error_behavior = self._error_behaviors.get(path)
        if error_behavior:
            # Apply delay if specified
            if error_behavior.get("delay", 0) > 0:
                time.sleep(error_behavior["delay"])
            # Return error status
            self.send_error(error_behavior["status"], f"Simulated {error_behavior['status']} error")
            return

        # Route: OpenAI API endpoints
        # /v1/chat/completions -> Mock chat completions (summarization, speaker detection,
        # GIL extract_quotes, GIL score_entailment, KG extract_kg_graph)
        if path == "/v1/chat/completions":
            self._handle_chat_completions()
            return

        # Route: OpenAI Whisper API endpoint
        # /v1/audio/transcriptions -> Mock audio transcriptions
        if path == "/v1/audio/transcriptions":
            self._handle_audio_transcriptions()
            return

        # Route: DGX pyannote diarization (#954)
        # /v1/diarize -> mock two-speaker diarization result
        if path == "/v1/diarize":
            self._handle_dgx_diarize()
            return

        # Route: Anthropic API endpoints
        # /v1/messages -> Mock Anthropic messages API (speaker detection, summarization)
        # Anthropic SDK appends /v1/messages to base URL, so we match /v1/messages
        if path == "/v1/messages" or path.endswith("/v1/messages"):
            self._handle_anthropic_messages()
            return

        # Route: Gemini API endpoints
        # /v1beta/models/{model}:generateContent -> Mock Gemini generateContent
        if path.startswith("/v1beta/models/") and path.endswith(":generateContent"):
            self._handle_gemini_generate_content()
            return

        # Route: Deepgram pre-recorded transcription
        # Deepgram SDK posts {base}/v1/listen -> Mock a diarized two-speaker response
        if path == "/v1/listen":
            self._handle_deepgram_listen()
            return

        # Ollama native: POST /api/generate (warmup / wait_until_ready in OllamaProvider)
        if path == "/api/generate":
            self._handle_ollama_generate()
            return

        # 404 for all other paths
        # Check if it's an OpenAI or Gemini endpoint
        if path.startswith("/v1/") or path.startswith("/v1beta/"):
            self.send_error(404, "API endpoint not found")
        else:
            self.send_error(404, "File not found")

    def _handle_chat_completions(self):
        """Handle OpenAI chat (summarization, speaker, GIL extract_quotes/score_entailment)."""
        # Guardrail-injection check (#999 / ADR-099): tests can pre-inject
        # an empty/thinking-prose/bad-JSON/finish_reason=length response.
        injection = type(self)._pop_injected_violation("/v1/chat/completions")
        if injection:
            self._emit_chat_violation(injection)
            return
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Request body required")
                return

            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode("utf-8"))

            # Extract request details
            messages = request_data.get("messages", [])
            user_message = next((m for m in messages if m.get("role") == "user"), {})
            raw_content = user_message.get("content", "")
            # Normalize content: OpenAI sends string; Mistral can send string or list of parts
            if isinstance(raw_content, list):
                user_content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content
                )
            else:
                user_content = raw_content if isinstance(raw_content, str) else ""
            system_message = next((m for m in messages if m.get("role") == "system"), {})
            raw_sys = system_message.get("content", "")
            if isinstance(raw_sys, list):
                system_content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_sys
                )
            else:
                system_content = raw_sys if isinstance(raw_sys, str) else ""
            response_format = request_data.get("response_format", {})

            # Determine response type: bundled GIL, speaker, GIL evidence, or summarization
            bundled_json = _bundled_gil_json(user_content)
            if bundled_json is not None:
                response_data = {
                    "id": "chatcmpl-test-bundled",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": bundled_json},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
                }
            elif response_format.get("type") == "json_object":
                # Speaker detection response
                response_data = {
                    "id": "chatcmpl-test-speaker",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", config.DEFAULT_OPENAI_SPEAKER_MODEL),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "speakers": ["Host", "Guest"],
                                        "hosts": ["Host"],
                                        "guests": ["Guest"],
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }
            elif "Insight:" in user_content and "quote_text" in user_content:
                # GIL extract_quotes (user prompt has transcript + insight, wants JSON quote_text)
                # Return a short substring from the transcript so transcript.find(quote) succeeds
                if "Transcript (excerpt):" in user_content:
                    excerpt = user_content.split("Transcript (excerpt):")[1].split("Insight:")[0]
                else:
                    excerpt = user_content.split("Insight:")[0]
                excerpt = excerpt.strip()
                mock_quote = (
                    excerpt[:60].strip()
                    if len(excerpt) > 10
                    else (excerpt or "Evidence from transcript.")
                )
                response_data = {
                    "id": "chatcmpl-test-extract-quote",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"quote_text": mock_quote}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
                }
            elif "Premise:" in user_content and "Hypothesis:" in user_content:
                # GIL score_entailment (premise + hypothesis; 0–1 instruction may be in system msg)
                response_data = {
                    "id": "chatcmpl-test-entailment",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "0.85"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 60, "completion_tokens": 2, "total_tokens": 62},
                }
            elif "knowledge-graph fragment" in system_content.lower():
                # KG extract_kg_graph (system from build_kg_transcript_system_prompt)
                kg_json = {
                    "topics": [{"label": "E2E mock topic"}],
                    "entities": [{"name": "E2E Entity", "entity_kind": "person"}],
                }
                response_data = {
                    "id": "chatcmpl-test-kg-extract",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(kg_json),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
                }
            else:
                # Summarization response (or generate_insights, cleaning, etc.)
                # Generate a simple summary based on text length
                summary_length = min(200, len(user_content) // 10)
                summary = (
                    f"This is a test summary of the transcript. {user_content[:summary_length]}..."
                )

                response_data = {
                    "id": "chatcmpl-test-summary",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": summary},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }

            # Send response
            response_json = json.dumps(response_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode("utf-8"))

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.send_error(500, f"Error handling chat completions: {e}")

    def _handle_deepgram_listen(self):
        """Handle Deepgram pre-recorded transcription (POST /v1/listen).

        The deepgram-sdk posts raw audio bytes and deserializes the JSON into a
        typed ``ListenV1Response``. Return a canned two-speaker diarized response
        so the real SDK request-build + response-deserialize path runs end-to-end
        against this server (the mock-server round-trip).
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                self.rfile.read(content_length)  # drain the posted audio bytes

            response_data = {
                "metadata": {
                    "model_info": {"name": "nova-3"},
                    "duration": 7.5,
                    "channels": 1,
                },
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": ("Welcome to the show. Thanks for having me."),
                                    "words": [
                                        {
                                            "word": "welcome",
                                            "punctuated_word": "Welcome",
                                            "start": 0.0,
                                            "end": 0.4,
                                            "speaker": 0,
                                        },
                                        {
                                            "word": "thanks",
                                            "punctuated_word": "Thanks",
                                            "start": 2.0,
                                            "end": 2.4,
                                            "speaker": 1,
                                        },
                                    ],
                                }
                            ]
                        }
                    ],
                    "utterances": [
                        {
                            "start": 0.0,
                            "end": 1.8,
                            "transcript": "Welcome to the show.",
                            "speaker": 0,
                        },
                        {
                            "start": 2.0,
                            "end": 3.6,
                            "transcript": "Thanks for having me.",
                            "speaker": 1,
                        },
                    ],
                },
            }
            response_json = json.dumps(response_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode("utf-8"))
        except Exception as e:  # noqa: BLE001 - mock server best-effort
            logger.error("Error handling Deepgram listen: %s", e, exc_info=True)
            self.send_error(500, f"Error handling Deepgram listen: {e}")

    def _handle_audio_transcriptions(self):
        """Handle OpenAI audio transcriptions API requests."""
        # Guardrail-injection check (#999 / ADR-099): tests can pre-inject a
        # structurally-valid HTTP 200 response whose content fails the whisper
        # length-floor or empty-response guardrail. The consumer's
        # GuardrailViolation handler then fires its fallback path.
        injection = type(self)._pop_injected_violation("/v1/audio/transcriptions")
        if injection:
            self._emit_transcription_violation(injection)
            return
        try:
            # Parse multipart form data
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_error(400, "Content-Type must be multipart/form-data")
                return

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Request body required")
                return

            body = self.rfile.read(content_length)

            # Parse multipart form data using email module: pull the uploaded
            # filename and the requested ``response_format`` so we can answer with
            # the type the client actually asked for.
            filename = "unknown_audio.mp3"
            response_format = "text"  # default to legacy text/plain unless asked otherwise
            try:
                msg = message_from_bytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
                for part in msg.walk():
                    if part.get_content_disposition() != "form-data":
                        continue
                    content_disposition = part.get("Content-Disposition", "")
                    field_name = ""
                    if 'name="' in content_disposition:
                        field_name = content_disposition.split('name="', 1)[1].split('"', 1)[0]

                    if "filename=" in content_disposition:
                        # Format: filename="audio.mp3" or filename=audio.mp3
                        parts = content_disposition.split("filename=")
                        if len(parts) > 1:
                            filename_part = parts[1].strip().strip('"').strip("'")
                            # Reject path separators / ".." then sanitize for safe echo.
                            if filename_part and not any(
                                bad in filename_part for bad in ("/", "\\", "..")
                            ):
                                sanitized = "".join(
                                    c for c in filename_part if c.isalnum() or c in "._-"
                                )
                                if sanitized:
                                    filename = sanitized
                    elif field_name == "response_format":
                        raw = part.get_payload(decode=True)
                        if raw:
                            response_format = raw.decode("utf-8", "ignore").strip()
            except Exception:
                # Parsing is best-effort for E2E; fall back to defaults.
                pass

            # Generate a realistic transcription response. A real minutes-long
            # episode yields hundreds of words, and the DGX whisper length-floor
            # guardrail (#1031, ~1.25 words/sec) rejects implausibly short
            # transcripts as truncation. The happy-path mock must therefore
            # clear the floor for the test-fixture audio (the DGX e2e suite uses
            # p01_e01.mp3, ~709 s -> ~886-word floor); repeat a sentence to a
            # comfortably-realistic length. filename is sanitized above and safe
            # to format. Consumers assert on the "test transcription" substring,
            # so the padding does not change any expectation.
            _body_sentence = "The audio contains spoken content that has been transcribed. "
            transcript = f"This is a test transcription of {filename}. " + _body_sentence * 200

            # Answer in the requested type. ``json``/``verbose_json`` (what the
            # OpenAI SDK and the DGX faster-whisper client both send) get a proper
            # JSON body with text + segments; ``text``/unspecified gets text/plain.
            if response_format in ("json", "verbose_json"):
                payload: Dict[str, Any] = {"text": transcript}
                if response_format == "verbose_json":
                    payload.update(
                        {
                            "task": "transcribe",
                            "language": "en",
                            "duration": 7.5,
                            "segments": [
                                {"id": 0, "start": 0.0, "end": 7.5, "text": transcript},
                            ],
                        }
                    )
                response_json = json.dumps(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_json)))
                self.end_headers()
                self.wfile.write(response_json.encode("utf-8"))
                return

            # Send response (text format)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(transcript)))
            self.end_headers()
            self.wfile.write(transcript.encode("utf-8"))

        except Exception as e:
            self.send_error(500, f"Error handling audio transcriptions: {e}")

    # ---------------------------------------------------------------------
    # Guardrail-violation payloads (#999 / ADR-099 E2E test support).
    # Each emitter returns a structurally-valid HTTP 200 whose CONTENT trips
    # the corresponding consumer-side guardrail. Tests assert that the
    # consumer's fallback path fired in response.
    # ---------------------------------------------------------------------

    def _send_json_200(self, body: Dict[str, Any]) -> None:
        payload = json.dumps(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def _emit_transcription_violation(self, violation_type: str) -> None:
        """Return a guardrail-violating whisper transcription response.

        violation_type ∈ {"transcription:empty", "transcription:length_floor"}.
        Default to ``length_floor`` for any unknown value (safer than 500).
        """
        if violation_type == "transcription:empty":
            text = ""
        else:  # transcription:length_floor (or unknown)
            text = "fragment"  # 1 word — well under any plausible duration-floor
        self._send_json_200(
            {
                "text": text,
                "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": text}],
            }
        )

    def _emit_diarize_violation(self, violation_type: str) -> None:
        """Return a guardrail-violating diarize response (empty segments).

        Only one violation_type today: ``diarize:empty_segments``.
        """
        self._send_json_200(
            {
                "model_name": "pyannote/speaker-diarization-community-1",
                "num_speakers": 0,
                "segments": [],
            }
        )

    def _emit_chat_violation(self, violation_type: str) -> None:
        """Return a guardrail-violating chat-completions response.

        Covers Ollama, OpenAI-compatible vLLM, etc. violation_type ∈ {
            'chat:empty_content', 'chat:thinking_prose',
            'chat:bad_json', 'chat:finish_length'
        }.
        """
        if violation_type == "chat:thinking_prose":
            content = "<think>let me think about this carefully</think>"
            finish_reason = "stop"
        elif violation_type == "chat:bad_json":
            content = "this is not { valid json"
            finish_reason = "stop"
        elif violation_type == "chat:finish_length":
            content = "This response was truncated mid-sentence becau"
            finish_reason = "length"
        else:  # chat:empty_content (or unknown)
            content = ""
            finish_reason = "stop"
        self._send_json_200(
            {
                "id": "chatcmpl-violation-injected",
                "object": "chat.completion",
                "created": 0,
                "model": "violation-injected",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(content),
                    "total_tokens": len(content),
                },
            }
        )

    def _emit_anthropic_violation(self, violation_type: str) -> None:
        """Return a guardrail-violating Anthropic /v1/messages response.

        violation_type ∈ {
            'anthropic:empty_content', 'anthropic:thinking_prose',
            'anthropic:max_tokens'
        }.
        """
        if violation_type == "anthropic:thinking_prose":
            text = "<think>Let me think about this carefully before I respond.</think>"
            stop_reason = "end_turn"
        elif violation_type == "anthropic:max_tokens":
            text = "This response was truncated mid-sent"
            # ADR-100: tests assume callers normalize Anthropic's "max_tokens" → "length"
            stop_reason = "length"
        else:  # anthropic:empty_content (or unknown)
            text = ""
            stop_reason = "end_turn"
        self._send_json_200(
            {
                "id": "msg-violation-injected",
                "type": "message",
                "role": "assistant",
                "model": "violation-injected",
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": max(len(text), 1)},
            }
        )

    def _emit_gemini_violation(self, violation_type: str) -> None:
        """Return a guardrail-violating Gemini generateContent response.

        violation_type ∈ {
            'gemini:empty_content', 'gemini:thinking_prose',
            'gemini:max_tokens'
        }.
        """
        if violation_type == "gemini:thinking_prose":
            text = "<think>Okay, so I need to figure out the right answer here.</think>"
            finish_reason = "STOP"
        elif violation_type == "gemini:max_tokens":
            text = "Truncated answer mid-thoug"
            finish_reason = "length"  # ADR-100 limitation: helper trips only on lowercase "length"
        else:  # gemini:empty_content (or unknown)
            text = ""
            finish_reason = "STOP"
        self._send_json_200(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": text}], "role": "model"},
                        "finishReason": finish_reason,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": max(len(text), 1),
                    "totalTokenCount": max(len(text), 1) + 1,
                },
            }
        )

    def _emit_generate_violation(self, violation_type: str) -> None:
        """Return a guardrail-violating Ollama /api/generate response.

        violation_type ∈ {'generate:empty', 'generate:thinking_prose'}.
        """
        if violation_type == "generate:thinking_prose":
            response = "<think>okay so I need to think about this</think>"
        else:  # generate:empty (or unknown)
            response = ""
        self._send_json_200(
            {
                "model": "violation-injected",
                "created_at": "2026-06-15T00:00:00Z",
                "response": response,
                "done": True,
                "done_reason": "stop",
            }
        )

    def _handle_dgx_probe(self, path: str, head_only: bool = False):
        """DGX health/model probe: GET /v1/models (faster-whisper + pyannote) and
        GET /health (pyannote). OpenAI-style envelope so ``check_*_health`` passes."""
        if path == "/health":
            body: Dict[str, Any] = {
                "status": "ok",
                "model": "pyannote/speaker-diarization-community-1",
            }
        else:  # /v1/models
            body = {
                "object": "list",
                "data": [
                    {"id": "large-v3", "object": "model", "owned_by": "openai"},
                    {"id": "pyannote/speaker-diarization-community-1", "object": "model"},
                ],
            }
        response_json = json.dumps(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        if not head_only:
            self.wfile.write(response_json.encode("utf-8"))

    def _handle_dgx_diarize(self):
        """Mock the DGX pyannote ``POST /v1/diarize`` (#954): canned two-speaker
        result so the real TailnetDgxDiarizationProvider HTTP round-trip runs."""
        # Guardrail-injection check (#999 / ADR-099): empty-segments injection
        # for the pyannote-side preventive guardrail.
        injection = type(self)._pop_injected_violation("/v1/diarize")
        if injection:
            # Drain any audio body the client uploaded first so the socket
            # state stays clean for the canned response.
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length:
                self.rfile.read(content_length)
            self._emit_diarize_violation(injection)
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length:
                self.rfile.read(content_length)  # drain the uploaded audio
            body = {
                "model_name": "pyannote/speaker-diarization-community-1",
                "num_speakers": 2,
                "segments": [
                    {"start": 0.0, "end": 4.5, "speaker": "SPEAKER_00"},
                    {"start": 4.5, "end": 9.0, "speaker": "SPEAKER_01"},
                ],
            }
            response_json = json.dumps(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode("utf-8"))
        except Exception as e:  # noqa: BLE001 - mock server best-effort
            self.send_error(500, f"Error handling DGX diarize: {e}")

    def _handle_ollama_version(self, head_only: bool = False):
        """Handle Ollama version API requests (health check).

        Ollama uses GET /api/version to check if the server is running.
        Returns a simple version response.
        """
        try:
            # Return a simple version response (Ollama format)
            version_response = {"version": "1.0.0"}
            response_json = json.dumps(version_response)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            if not head_only:
                self.wfile.write(response_json.encode("utf-8"))

        except Exception as e:
            self.send_error(500, f"Error handling Ollama version: {e}")

    def _handle_ollama_tags(self, head_only: bool = False):
        """Handle Ollama tags API requests (model list).

        Ollama uses GET /api/tags to list available models.
        Returns a list of models that are commonly used in tests.
        """
        try:
            # Return a list of models that are commonly used in tests
            # This matches what the integration tests expect
            models_response = {
                "models": [
                    {"name": "llama3.3:latest"},
                    {"name": "llama3.2:latest"},
                    {"name": "llama3.1:latest"},
                    {"name": "llama3.1:8b"},
                    {"name": "llama3:latest"},
                    # Ollama model tags returned to fixture clients
                    {"name": "mistral:7b"},
                    {"name": "mistral-nemo:12b"},
                    {"name": "mistral-small3.2:latest"},
                    {"name": "qwen2.5:7b"},
                    {"name": "qwen2.5:32b"},
                    {"name": "qwen3.5:9b"},
                    {"name": "qwen3.5:27b"},
                    {"name": "qwen3.5:35b"},
                    {"name": "qwen3.5:35b-a3b"},
                    {"name": "phi3:mini"},
                    # Ollama tags used by fixture-backed full-pipeline runs
                    {"name": "gemma2:9b"},
                ]
            }
            response_json = json.dumps(models_response)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            if not head_only:
                self.wfile.write(response_json.encode("utf-8"))

        except Exception as e:
            self.send_error(500, f"Error handling Ollama tags: {e}")

    def _handle_ollama_generate(self) -> None:
        """Handle Ollama native POST /api/generate (model warm-up ping).

        OllamaProvider.warmup posts JSON: model, prompt, stream, options.
        The E2E mock must return 200 so warm-up does not log 404 warnings.
        """
        # Guardrail-injection check (#999 / ADR-099): empty/thinking-prose
        # injection for the Ollama generate guardrail. Drain the body first
        # so socket state stays clean.
        injection = type(self)._pop_injected_violation("/api/generate")
        if injection:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length:
                self.rfile.read(content_length)
            self._emit_generate_violation(injection)
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Request body required")
                return

            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode("utf-8"))
            model_name = request_data.get("model", "unknown")

            # Minimal Ollama-style response (client only checks HTTP 200)
            payload = {
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": "ok",
                "done": True,
                "context": [],
                "total_duration": 1,
                "load_duration": 1,
                "prompt_eval_count": 1,
                "eval_count": 1,
            }
            response_json = json.dumps(payload)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode("utf-8"))

        except Exception as e:
            self.send_error(500, f"Error handling Ollama generate: {e}")

    def _handle_anthropic_messages(self):
        """Handle Anthropic messages API requests.

        Anthropic API uses POST /v1/messages
        This handler supports:
        - Speaker detection (when system prompt contains "speaker" or "NER")
        - GIL extract_quotes / score_entailment (Insight+quote_text or Premise+Hypothesis)
        - Summarization (default)
        - Guardrail violation injection (#1003 / ADR-100): see ``inject_violation``
        """
        # Drain the request body first so error injection doesn't desync the keep-alive socket.
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                _drained_body = self.rfile.read(content_length)
            else:
                _drained_body = b""
        except Exception:
            _drained_body = b""

        injection = type(self)._pop_injected_violation("/v1/messages")
        if injection is not None:
            self._emit_anthropic_violation(injection)
            return

        try:
            if not _drained_body:
                self.send_error(400, "Request body required")
                return
            request_data = json.loads(_drained_body.decode("utf-8"))

            # Extract request details
            messages = request_data.get("messages", [])
            user_message = next((m for m in messages if m.get("role") == "user"), {})
            # Handle content that might be a string or list
            user_content_raw = user_message.get("content", "")
            if isinstance(user_content_raw, list):
                # Extract text from content blocks
                user_content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in user_content_raw
                )
            else:
                user_content = str(user_content_raw)
            system = request_data.get("system", "")

            # GIL extract_quotes: user has Transcript (excerpt) + Insight, wants JSON quote_text
            bundled_json = _bundled_gil_json(user_content)
            if bundled_json is not None:
                response_data = {
                    "id": "msg-test-bundled",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": bundled_json}],
                    "model": request_data.get("model", "claude-haiku-4-5"),
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 40},
                }
            elif "Insight:" in user_content and (
                "quote_text" in user_content or "Transcript (excerpt):" in user_content
            ):
                if "Transcript (excerpt):" in user_content:
                    excerpt = user_content.split("Transcript (excerpt):")[1].split("Insight:")[0]
                else:
                    excerpt = user_content.split("Insight:")[0]
                excerpt = excerpt.strip()
                mock_quote = (
                    excerpt[:60].strip()
                    if len(excerpt) > 10
                    else (excerpt or "Evidence from transcript.")
                )
                response_data = {
                    "id": "msg-test-extract-quote",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps({"quote_text": mock_quote})}],
                    "model": request_data.get("model", "claude-haiku-4-5"),
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 80, "output_tokens": 20},
                }
            elif "Premise:" in user_content and "Hypothesis:" in user_content:
                # GIL score_entailment: user has Premise + Hypothesis, wants 0–1 number
                response_data = {
                    "id": "msg-test-entailment",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "0.85"}],
                    "model": request_data.get("model", "claude-haiku-4-5"),
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 60, "output_tokens": 2},
                }
            elif "knowledge-graph fragment" in _anthropic_system_text(system).lower():
                # KG extract_kg_graph (system from build_kg_transcript_system_prompt)
                kg_json = {
                    "topics": [{"label": "E2E mock topic"}],
                    "entities": [{"name": "E2E Entity", "entity_kind": "person"}],
                }
                response_data = {
                    "id": "msg-test-kg-extract",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(kg_json)}],
                    "model": request_data.get("model", "claude-haiku-4-5"),
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 40},
                }
            else:
                # Determine response type based on system prompt
                # If system prompt contains "speaker" or "NER", it's speaker detection
                is_speaker_detection = (
                    "speaker" in system.lower()
                    or "ner" in system.lower()
                    or "name" in system.lower()
                )

                if is_speaker_detection:
                    # Speaker detection response (Anthropic format)
                    response_data = {
                        "id": "msg-test-speaker",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "speakers": ["Host", "Guest"],
                                        "hosts": ["Host"],
                                        "guests": ["Guest"],
                                    }
                                ),
                            }
                        ],
                        "model": request_data.get("model", "claude-haiku-4-5"),
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                        },
                    }
                else:
                    # Summarization response (Anthropic format)
                    summary_length = min(200, len(user_content) // 10)
                    summary = (
                        f"This is a test summary of the transcript. "
                        f"{user_content[:summary_length]}..."
                    )
                    response_data = {
                        "id": "msg-test-summary",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": summary}],
                        "model": request_data.get("model", "claude-haiku-4-5"),
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                        },
                    }

            # Send response
            response_json = json.dumps(response_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode("utf-8"))
            self.wfile.flush()  # Ensure response is sent

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error("Error handling Anthropic messages: %s", e, exc_info=True)
            self.send_error(500, f"Error handling Anthropic messages: {e}")

    def _handle_gemini_generate_content(self):
        """Handle Gemini generateContent API requests.

        Gemini API uses POST /v1beta/models/{model}:generateContent
        This handler supports:
        - Audio transcription (multimodal input with audio)
        - GIL extract_quotes / score_entailment (Insight+quote_text or Premise+Hypothesis)
        - Text generation (summarization, speaker detection)
        - Guardrail violation injection (#1003 / ADR-100): see ``inject_violation``
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
            else:
                body = b""
        except Exception:
            body = b""

        injection = type(self)._pop_injected_violation("/v1beta/generateContent")
        if injection is not None:
            self._emit_gemini_violation(injection)
            return

        try:
            if not body:
                self.send_error(400, "Request body required")
                return
            request_data = json.loads(body.decode("utf-8"))

            # Extract contents from request
            contents = request_data.get("contents", [])

            # Check if request contains audio (multimodal) or just text
            has_audio = False
            text_prompt = ""
            for content in contents:
                # Handle both formats:
                # 1. content is a dict with "parts" key: {"parts": [...]}
                # 2. content is a list: [...]
                # 3. content is a string: "..."
                if isinstance(content, dict) and "parts" in content:
                    parts = content.get("parts", [])
                    for part in parts:
                        if isinstance(part, dict) and "mime_type" in part:
                            mime_type = part.get("mime_type", "")
                            if mime_type.startswith("audio/"):
                                has_audio = True
                        elif isinstance(part, dict) and "text" in part:
                            # The google-genai SDK serializes text parts as
                            # {"text": ...} dicts, not raw strings — extract those too
                            # (accumulate; otherwise JSON calls saw an empty prompt and
                            # got the default text summary, failing every JSON parse).
                            text_prompt += str(part["text"])
                        elif isinstance(part, str):
                            text_prompt += part
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "mime_type" in part:
                            mime_type = part.get("mime_type", "")
                            if mime_type.startswith("audio/"):
                                has_audio = True
                        elif isinstance(part, dict) and "text" in part:
                            text_prompt += str(part["text"])
                        elif isinstance(part, str):
                            text_prompt += part
                elif isinstance(content, str):
                    text_prompt += content

            # Determine response type
            if has_audio:
                # Audio transcription response
                transcript = (
                    "This is a test transcription from Gemini. "
                    "The audio contains spoken content that has been transcribed."
                )
                response_data = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": transcript}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 50,
                        "totalTokenCount": 150,
                    },
                }
            else:
                response_data = _gemini_text_response_data(request_data, text_prompt)

            # Send response
            response_json = json.dumps(response_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode("utf-8"))

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.send_error(500, f"Error handling Gemini generateContent: {e}")

    def _validate_and_sanitize_filename(
        self, subdir: str, filename: str
    ) -> Optional[tuple[str, str]]:
        """Validate and sanitize filename for safe path construction.

        This helper validates user input according to CodeQL recommendations:
        - No path separators ("/", "\\")
        - No ".." segments
        - Exactly one "." character
        - Matches allowlist pattern

        Args:
            subdir: Subdirectory name ("audio" or "transcripts")
            filename: Filename from request (untrusted input)

        Returns:
            Tuple of (validated_subdir, validated_filename) if valid, None otherwise
        """
        # Validate subdir parameter (prevent path injection via subdir)
        if not subdir or not isinstance(subdir, str):
            return None
        # Reject subdir containing path separators or ".."
        if "/" in subdir or "\\" in subdir or ".." in subdir or subdir.strip() != subdir:
            return None
        # Only allow known safe subdirs
        if subdir not in ("audio", "transcripts"):
            return None

        # Reject empty filenames
        if not filename or not filename.strip():
            return None

        # Reject filenames containing path separators or ".." segments
        # Check for ".." explicitly (not just replacement) per CodeQL recommendation
        # This prevents attacks like ".../...//" which could become "../" after processing
        if "/" in filename or "\\" in filename or ".." in filename:
            return None

        # Reject filenames with more than one "." character (safety check)
        # Only one dot allowed (for the file extension)
        if filename.count(".") != 1:
            return None

        # Validate filename contains only safe characters (alphanumeric, underscore, hyphen, dot)
        # This is less restrictive than the allowlist pattern but still secure
        # We check for safety (no path separators, no "..") and reasonable characters
        # The allowlist pattern was too restrictive for testing 404 behavior
        # Security: We still reject path separators and ".." above, so this is safe
        safe_char_pattern = re.compile(r"^[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+$")
        if not safe_char_pattern.match(filename):
            return None

        # Optional: Validate extension matches subdir (defense in depth)
        # This helps ensure we're serving the right type of file
        if subdir == "audio":
            if not filename.endswith(".mp3"):
                return None
        elif subdir == "transcripts":
            if not filename.endswith(".txt"):
                return None
        else:
            # This should never happen due to validation above, but defensive check
            return None

        # Return sanitized values (these have passed all validation checks)
        return (subdir, filename)

    def _build_validated_path(self, base_dir: Path, validated_filename: str) -> Optional[Path]:
        """Build and validate a path using pre-validated components.

        This helper constructs a path from validated components and verifies
        the result is safe. Follows CodeQL recommended pattern:
        1. Normalize path using os.path.normpath
        2. Verify normalized path is within base directory using string comparison
        3. Only then construct Path object and verify it's a file

        Args:
            base_dir: Base directory (must be within fixture root, already validated)
            validated_filename: Filename that has passed validation
                               (no path separators, matches allowlist, exactly one dot)

        Returns:
            Validated Path if construction succeeds and path is safe, None otherwise
        """
        # validated_filename has been validated according to CodeQL recommendations:
        # - No path separators ("/", "\\")
        # - No ".." segments (checked explicitly, not just replacement)
        # - Exactly one "." character (allowlist pattern)
        # - Matches allowlist regex pattern

        # Explicit validation check right before path construction (for CodeQL)
        # Ensure validated_filename is still safe (defense in depth)
        if not validated_filename or not isinstance(validated_filename, str):
            return None
        if "/" in validated_filename or "\\" in validated_filename or ".." in validated_filename:
            return None

        # Explicit validation for base_dir (for CodeQL)
        # Ensure base_dir is a Path object
        if not isinstance(base_dir, Path):
            return None

        # Follow CodeQL recommended pattern: normalize first, then verify containment
        # Convert base_dir to absolute string path for normalization
        try:
            base_dir_str = str(base_dir.resolve())
        except (OSError, RuntimeError):
            return None

        # Construct full path string and normalize it (CodeQL recommended approach)
        # This normalizes any ".." or "." components that might have been missed
        full_path_str = os.path.join(base_dir_str, validated_filename)
        normalized_path_str = os.path.normpath(full_path_str)

        # Verify normalized path is within base directory using string comparison
        # This is the key check CodeQL understands - string comparison after normalization
        # CodeQL recognizes this pattern as safe path validation
        if not normalized_path_str.startswith(base_dir_str):
            return None

        # Now construct Path object from normalized string (safe per verification above)
        # CodeQL sees normalized_path_str as safe because it passed string verification
        try:
            candidate = Path(normalized_path_str).resolve()
        except (OSError, RuntimeError):
            # Path resolution failed (e.g., broken symlink, invalid path)
            return None

        # Verify candidate is a file and is still within base_dir (defense in depth)
        # Additional check using Path methods for extra safety
        if not candidate.is_file():
            return None
        # Verify path containment (prevents symlink attacks)
        # This check happens AFTER resolve() to catch any path traversal attempts
        try:
            if not candidate.is_relative_to(base_dir):
                return None
        except (ValueError, AttributeError):
            # Fallback for edge cases
            return None

        return candidate

    def _get_safe_fixture_path(self, subdir: str, filename: str) -> Optional[Path]:
        """Safely construct and validate a fixture file path.

        This method prevents path traversal attacks by:
        1. Using an allowlist pattern to validate filename format
        2. Rejecting filenames with path separators, "..", or multiple dots
        3. Building the path relative to a fixed root directory
        4. Normalizing the path with resolve()
        5. Verifying the normalized path is within the intended root and is a file

        Args:
            subdir: Subdirectory name ("audio" or "transcripts")
            filename: Filename from request (untrusted input)

        Returns:
            Safe Path if validation passes, None if input is invalid
        """
        # Validate and sanitize inputs (prevents path injection)
        validation_result = self._validate_and_sanitize_filename(subdir, filename)
        if validation_result is None:
            return None

        # Unpack validated values (these have passed all security checks)
        validated_subdir, validated_filename = validation_result

        # validated_subdir has been validated: one of ("audio", "transcripts")
        # No path separators, no "..", safe to use in path construction
        fixture_root = self.get_fixture_root()

        # Explicit validation check right before path construction (for CodeQL)
        # Ensure validated_subdir is still safe (defense in depth)
        if not validated_subdir or not isinstance(validated_subdir, str):
            return None
        if "/" in validated_subdir or "\\" in validated_subdir or ".." in validated_subdir:
            return None
        if validated_subdir not in ("audio", "transcripts"):
            return None

        # Build base directory using validated subdir + fixture version segment.
        # validated_subdir is safe: validated to be one of ("audio", "transcripts").
        # _FIXTURE_VERSION is read from tests/fixtures/FIXTURES_VERSION at module load
        # (server-side constant, not user input). All validation checks passed above.
        base_dir = fixture_root / validated_subdir / _FIXTURE_VERSION

        # Verify base_dir is within fixture_root (defense in depth)
        # This ensures that even if validated_subdir somehow contained path traversal,
        # the resolved path would still be within fixture_root
        try:
            # Explicit validation right before resolve() (for CodeQL)
            if not isinstance(base_dir, Path):
                return None
            # resolved_base is the result of resolving base_dir (which uses validated_subdir)
            resolved_base = base_dir.resolve()
            # Explicit validation right before is_relative_to() check (for CodeQL)
            if not isinstance(resolved_base, Path):
                return None
            if not resolved_base.is_relative_to(fixture_root):
                return None
            # Use resolved base for all subsequent operations
            base_dir = resolved_base
        except (OSError, RuntimeError, ValueError, AttributeError):
            return None

        # Build path using validated components
        # validated_filename is safe: validated according to CodeQL recommendations
        return self._build_validated_path(base_dir, validated_filename)

    def _validate_and_sanitize_rss_filename(self, rss_file: str) -> Optional[str]:
        """Validate and sanitize RSS filename for safe path construction.

        This helper validates user input according to CodeQL recommendations:
        - No path separators ("/", "\\")
        - No ".." segments
        - Exactly one "." character
        - Matches allowlist pattern

        Args:
            rss_file: RSS filename from dictionary lookup (untrusted input)

        Returns:
            Validated filename if valid, None otherwise
        """
        # Reject empty filenames
        if not rss_file or not rss_file.strip():
            return None

        # Reject filenames containing path separators or ".." segments
        # Check for ".." explicitly (not just replacement) per CodeQL recommendation
        # This prevents attacks like ".../...//" which could become "../" after processing
        if "/" in rss_file or "\\" in rss_file or ".." in rss_file:
            return None

        # Reject filenames with more than one "." character (allowlist pattern)
        # Expected format: pXX_*.xml (e.g., p01_mtb.xml, p01_fast.xml, p01_multi.xml)
        # Only one dot allowed (for the file extension)
        if rss_file.count(".") != 1:
            return None

        # Validate filename against allowlist pattern
        # Pattern: p<digits>_<alphanumeric_underscore>\.xml
        # Examples: p01_mtb.xml, p01_fast.xml, p01_multi.xml, p06_edge_cases.xml
        allowed_pattern = re.compile(r"^p\d+_[a-z0-9_]+\.xml$")
        if not allowed_pattern.match(rss_file):
            return None

        # Return sanitized value (has passed all validation checks)
        return rss_file

    def _get_safe_rss_path(self, rss_file: str) -> Optional[Path]:
        """Safely construct and validate an RSS file path.

        This method prevents path traversal attacks by:
        1. Using an allowlist pattern to validate RSS filename format
        2. Rejecting filenames with path separators, "..", or multiple dots
        3. Building the path relative to a fixed root directory
        4. Normalizing the path with resolve()
        5. Verifying the normalized path is within the intended root and is a file

        Args:
            rss_file: RSS filename from dictionary lookup (should be validated)

        Returns:
            Safe Path if validation passes, None if input is invalid
        """
        # Validate and sanitize input (prevents path injection)
        validated_rss_file = self._validate_and_sanitize_rss_filename(rss_file)
        if validated_rss_file is None:
            return None

        # validated_rss_file has been validated according to CodeQL recommendations:
        # - No path separators ("/", "\\")
        # - No ".." segments (checked explicitly, not just replacement)
        # - Exactly one "." character (allowlist pattern)
        # - Matches allowlist regex pattern (p\d+_[a-z0-9_]+\.xml)
        # Safe to use in path construction per CodeQL guidelines

        # Explicit validation check right before path construction (for CodeQL)
        # Ensure validated_rss_file is still safe (defense in depth)
        if not validated_rss_file or not isinstance(validated_rss_file, str):
            return None
        if "/" in validated_rss_file or "\\" in validated_rss_file or ".." in validated_rss_file:
            return None

        # Build base directory (hardcoded "rss", not user-controlled)
        fixture_root = self.get_fixture_root()
        # "rss" is a hardcoded string literal, not user-controlled, safe to use
        base_dir = fixture_root / "rss"

        # Verify base_dir is within fixture_root (defense in depth)
        try:
            # Explicit validation right before resolve() (for CodeQL)
            if not isinstance(base_dir, Path):
                return None
            # resolved_base is the result of resolving base_dir (hardcoded "rss")
            resolved_base = base_dir.resolve()
            # Explicit validation right before is_relative_to() check (for CodeQL)
            if not isinstance(resolved_base, Path):
                return None
            if not resolved_base.is_relative_to(fixture_root):
                return None
            # Use resolved base for all subsequent operations
            base_dir = resolved_base
        except (OSError, RuntimeError, ValueError, AttributeError):
            return None

        # Build path using validated components
        # validated_rss_file is safe: validated according to CodeQL recommendations
        # All validation checks passed above, safe to use in path construction
        return self._build_validated_path(base_dir, validated_rss_file)

    def _validate_served_file_path(self, file_path: Path) -> Optional[Path]:
        """Validate that a file path is safe to serve.

        This method performs final validation on a file path before serving.
        Follows CodeQL recommended pattern: normalize first, then verify containment.

        This validation follows CodeQL recommendations:
        - Normalize path using os.path.normpath
        - Verify normalized path is within fixture root using string comparison
        - Verify it's actually a file (not a directory)

        Args:
            file_path: Path to validate (should come from validated helper methods)

        Returns:
            Resolved and validated Path if safe, None if invalid
        """
        fixture_root = self.get_fixture_root()
        try:
            # Explicit validation: ensure file_path is a Path object
            if not isinstance(file_path, Path):
                return None

            # Follow CodeQL recommended pattern: normalize first, then verify containment
            # Convert to absolute string path for normalization
            fixture_root_str = str(fixture_root.resolve())
            file_path_str = str(file_path.resolve())

            # Normalize the path (CodeQL recommended approach)
            # This normalizes any ".." or "." components
            normalized_path_str = os.path.normpath(file_path_str)

            # Verify normalized path is within fixture root using string comparison
            # This is the key check CodeQL understands - string comparison after normalization
            # CodeQL recognizes this pattern as safe path validation
            if not normalized_path_str.startswith(fixture_root_str):
                return None

            # Now construct Path object from normalized string (safe per verification above)
            # CodeQL sees normalized_path_str as safe because it passed string verification
            resolved_path = Path(normalized_path_str).resolve()

            # Verify it's actually a file (not a directory or symlink to directory)
            # Additional check using Path methods for extra safety
            if not resolved_path.is_file():
                return None

            # Return validated path (safe to use in file operations)
            # resolved_path has passed all validation checks above
            return resolved_path
        except (OSError, RuntimeError, ValueError, AttributeError):
            return None

    def _serve_file(
        self,
        file_path: Path,
        content_type: str,
        support_range: bool = False,
        head_only: bool = False,
    ):
        """Serve a file with proper headers and range request support.

        Args:
            file_path: Path to file to serve (must be validated and within fixture root)
            content_type: Content-Type header value
            support_range: Whether to support HTTP range requests (206 Partial Content)
            head_only: If True, send headers only (HTTP HEAD)
        """
        # Validate file_path is within fixture root (defense in depth)
        # Even though file_path comes from validated helper methods, we verify here
        # to satisfy static analysis tools and provide additional security layer
        validated_path = self._validate_served_file_path(file_path)
        if validated_path is None:
            self.send_error(403, "Path traversal not allowed")
            return

        # validated_path is now guaranteed safe: within fixture root, is a file
        # All subsequent operations use validated_path, not the original file_path
        # Explicit validation check right before file operations (for CodeQL)
        if not isinstance(validated_path, Path):
            self.send_error(500, "Invalid file path type")
            return

        # Final validation: ensure validated_path is still within fixture root
        # Follow CodeQL recommended pattern: normalize and verify using string comparison
        fixture_root = self.get_fixture_root()
        try:
            # Explicit validation: ensure validated_path is a Path object
            if not isinstance(validated_path, Path):
                self.send_error(403, "Invalid file path type")
                return

            # Follow CodeQL recommended pattern: normalize first, then verify containment
            # Convert to absolute string paths for normalization
            fixture_root_str = str(fixture_root.resolve())
            validated_path_str = str(validated_path.resolve())

            # Normalize the path (CodeQL recommended approach)
            normalized_path_str = os.path.normpath(validated_path_str)

            # Verify normalized path is within fixture root using string comparison
            # This is the key check CodeQL understands - string comparison after normalization
            # CodeQL recognizes this pattern as safe path validation
            if not normalized_path_str.startswith(fixture_root_str):
                self.send_error(403, "Path traversal not allowed")
                return

            # Reconstruct Path from normalized string (safe per verification above)
            # CodeQL sees normalized_path_str as safe because it passed string verification
            validated_path = Path(normalized_path_str).resolve()

            # Verify it's actually a file (not a directory or symlink to directory)
            if not validated_path.is_file():
                self.send_error(404, "File not found")
                return
        except (OSError, RuntimeError, ValueError, AttributeError):
            self.send_error(403, "Invalid file path")
            return

        try:
            # validated_path has passed all validation checks above
            # Safe to use in file operations per CodeQL guidelines
            # CodeQL recognizes normalized_path_str as safe, and validated_path is derived from it
            file_size = validated_path.stat().st_size

            # Check for Range request
            range_header = self.headers.get("Range")
            if support_range and range_header:
                # Parse Range header (e.g., "bytes=0-1023")
                try:
                    range_spec = range_header.replace("bytes=", "")
                    parts = range_spec.split("-")
                    if len(parts) != 2:
                        raise ValueError("Invalid range format")
                    start_str, end_str = parts
                    start = int(start_str) if start_str else 0
                    end = int(end_str) if end_str else file_size - 1
                    end = min(end, file_size - 1)

                    # Send 206 Partial Content
                    self.send_response(206)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()

                    if head_only:
                        return

                    # Send partial content
                    # validated_path is safe: normalized and verified using string comparison
                    # All validation checks passed above, safe to open
                    # CodeQL recognizes validated_path as safe because it was verified using
                    # string comparison after normalization (CodeQL recommended pattern)
                    try:
                        with open(validated_path, "rb") as f:
                            f.seek(start)
                            self.wfile.write(f.read(end - start + 1))
                        return
                    except BrokenPipeError:
                        # Client disconnected before response completed - normal and harmless
                        return
                except (ValueError, IndexError):
                    # Invalid range header, fall through to full file
                    pass

            # Send full file (200 OK)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            if support_range:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            if head_only:
                return

            # validated_path is safe: normalized and verified using string comparison
            # All validation checks passed above, safe to open
            # CodeQL recognizes validated_path as safe because it was verified using
            # string comparison after normalization (CodeQL recommended pattern)
            try:
                with open(validated_path, "rb") as f:
                    self.wfile.write(f.read())
            except BrokenPipeError:
                # Client disconnected before response completed - this is normal and harmless
                # Don't log as error, just ignore it
                pass

        except BrokenPipeError:
            # Client disconnected before response completed - this is normal and harmless
            # Don't log as error, just ignore it
            pass
        except Exception as e:
            # Only log actual errors, not client disconnections
            try:
                self.send_error(500, f"Error serving file: {e}")
            except BrokenPipeError:
                # Client already disconnected, can't send error response
                pass

    def log_message(self, format, *args):
        """Suppress server log messages during tests."""
        pass


class E2EHTTPServer:
    """E2E HTTP server for serving test fixtures."""

    def __init__(self, port: int = 0):
        """Initialize E2E server.

        Args:
            port: Port number (0 = auto-assign)
        """
        self.port = port
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.base_url: Optional[str] = None
        self.urls: Optional[E2EServerURLs] = None

    def start(self):
        """Start the E2E server."""
        handler = lambda *args, **kwargs: E2EHTTPRequestHandler(*args, **kwargs)  # noqa: E731
        self.server = socketserver.TCPServer(("127.0.0.1", self.port), handler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.urls = E2EServerURLs(self.base_url)

        # Start server in background thread
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        # Wait for server to be ready with explicit connection check
        # This prevents race conditions where tests start before server is accepting
        self._wait_for_server_ready()

    def _wait_for_server_ready(self, timeout: float = 5.0, interval: float = 0.1):
        """Wait for server to be ready to accept connections.

        Args:
            timeout: Maximum time to wait in seconds
            interval: Time between connection attempts in seconds

        Raises:
            RuntimeError: If server doesn't start within timeout
        """
        import socket

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the server
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(("127.0.0.1", self.port))
                sock.close()
                if result == 0:
                    # Connection successful - server is ready
                    return
            except (OSError, socket.error):
                pass
            time.sleep(interval)

        raise RuntimeError(f"E2E server failed to start within {timeout} seconds")

    def stop(self):
        """Stop the E2E server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.thread:
                self.thread.join(timeout=1.0)

    def reset(self):
        """Reset server state (for test isolation).

        Clears all error behaviors and resets allowed podcasts to None (all allowed).
        """
        E2EHTTPRequestHandler.clear_all_error_behaviors()
        E2EHTTPRequestHandler.set_allowed_podcasts(None)

    def set_allowed_podcasts(self, podcasts: Optional[set[str]]):
        """Set allowed podcasts for serving RSS feeds.

        Args:
            podcasts: Set of podcast names to allow, or None to allow all podcasts.
        """
        E2EHTTPRequestHandler.set_allowed_podcasts(podcasts)

    def set_error_behavior(self, url_path: str, status: int, delay: float = 0.0):
        """Set error behavior for a specific URL path.

        Args:
            url_path: URL path to apply error to (e.g., "/feeds/podcast1/feed.xml")
            status: HTTP status code to return (e.g., 404, 500)
            delay: Optional delay in seconds before responding
        """
        E2EHTTPRequestHandler.set_error_behavior(url_path, status, delay)

    def clear_error_behavior(self, url_path: str):
        """Clear error behavior for a specific URL path.

        Args:
            url_path: URL path to clear error behavior for
        """
        E2EHTTPRequestHandler.clear_error_behavior(url_path)


@pytest.fixture(scope="session")
def e2e_server():
    """E2E HTTP server fixture (session-scoped).

    This fixture provides a local HTTP server that serves test fixtures.
    The server is started once per test session and stopped after all tests.

    Usage:
        def test_something(e2e_server):
            rss_url = e2e_server.urls.feed("podcast1")
            audio_url = e2e_server.urls.audio("p01_e01")
            # Use URLs in tests...

    Real API Mode:
        When USE_REAL_OPENAI_API=1, the server still starts to serve fixture feeds,
        but the mock OpenAI endpoints are not used (real API is called instead).
        This allows testing real API with known fixture data.
    """
    # Always start the server, even in real API mode (needed for fixture feeds)
    # The mock OpenAI endpoints won't be used when USE_REAL_OPENAI_API=1
    server = E2EHTTPServer()
    server.start()
    yield server
    server.stop()
