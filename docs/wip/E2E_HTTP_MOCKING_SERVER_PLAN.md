# E2E HTTP Mocking Server Infrastructure Plan

## Overview

This document outlines the plan for creating a sophisticated local HTTP server infrastructure for E2E tests. The server will serve real RSS feeds, transcripts, and audio files from a structured folder, allowing E2E tests to use real HTTP clients without hitting external networks.

## Goals

1. **No External Network Calls**: E2E tests should not hit external networks (enforced with network blocking)

2. **Real HTTP Client**: E2E tests should use the real HTTP client (`downloader.fetch_url`) without mocking

3. **Real Data Files**: E2E tests should use real RSS feeds, transcripts, and audio files

4. **Organized Structure**: Test data should be organized in a clear folder structure

5. **Reusable**: The server should be reusable across all E2E tests

6. **Flexible**: The server should support various scenarios (success, errors, edge cases)

7. **Safe**: Hard network guard prevents accidental external network calls

8. **Debuggable**: Request logging and replay artifacts for troubleshooting

9. **Performant**: Optimized fixture scope for fast test execution

## Architecture

### Folder Structure

```text

tests/
├── fixtures/
│   └── e2e_server/                   # E2E server fixture structure (manually maintained)
│       ├── feeds/                    # RSS feed files
│       │   ├── podcast1/
│       │   │   ├── feed.xml          # RSS feed with absolute URLs
│       │   │   └── episodes/
│       │   │       ├── episode1/
│       │   │       │   ├── transcript.vtt
│       │   │       │   ├── transcript.srt
│       │   │       │   ├── transcript.json
│       │   │       │   ├── transcript.txt
│       │   │       │   └── audio.mp3
│       │   │       ├── episode2/
│       │   │       │   └── ...
│       │   │       └── ...
│       │   ├── podcast2/
│       │   │   ├── feed.xml          # RSS feed with relative URLs (tests app resolution)
│       │   │   └── episodes/
│       │   │       └── ...
│       │   └── edge_cases/
│       │       ├── malformed_rss.xml # Pre-written malformed RSS
│       │       ├── missing_transcript.xml
│       │       └── special_chars.xml
│       ├── transcripts/              # Standalone transcript files (if needed)
│       │   ├── sample.vtt
│       │   ├── sample.srt
│       │   └── sample.json
│       ├── audio/                    # Standalone audio files (if needed)
│       │   ├── short_test.mp3        # Small test file (< 10 seconds)
│       │   └── short_test.m4a
│       ├── manifest.json              # Optional: Documentation manifest (for reference)
│       └── README.md                  # Documentation on fixture structure
└── workflow_e2e/
    └── test_*.py                     # E2E tests using the server

```

### Manifest Structure (Optional Documentation)

**`tests/fixtures/e2e_server/manifest.json`** - Optional manifest for documentation/reference (not required for server operation):

```json
{
  "feeds": {
    "podcast1": {
      "title": "Test Podcast 1",
      "description": "A test podcast for E2E testing",
      "feed_file": "feeds/podcast1/feed.xml",
      "episodes": [
        {
          "id": "episode1",
          "title": "Episode 1: Introduction",
          "description": "First episode of the podcast",
          "transcript": {
            "vtt": "feeds/podcast1/episodes/episode1/transcript.vtt",
            "srt": "feeds/podcast1/episodes/episode1/transcript.srt",
            "json": "feeds/podcast1/episodes/episode1/transcript.json",
            "txt": "feeds/podcast1/episodes/episode1/transcript.txt"
          },
          "audio": {
            "mp3": "feeds/podcast1/episodes/episode1/audio.mp3",
            "mime_type": "audio/mpeg",
            "duration_seconds": 8.5,
            "size_bytes": 136000
          },
          "published_date": "2024-01-15T10:00:00Z"
        }
      ]
    }
  },
  "edge_cases": {
    "malformed_rss": {
      "file": "feeds/edge_cases/malformed_rss.xml",
      "description": "Intentionally malformed RSS for error testing"
    }
  }
}

```

**Note**: The manifest is optional and serves as documentation. The server operates directly on the file structure. Fixtures are manually maintained and checked into version control.

### Server Implementation

**Location**: `tests/fixtures/e2e_server/server.py`

**Key Components:**

1. **E2EHTTPServer** - Main HTTP server class

2. **E2EHTTPRequestHandler** - Custom request handler

3. **E2EServerFixture** - Pytest fixture for easy test integration

4. **URL Routing** - Map URLs to file paths based on folder structure

5. **Behavior Registry** - Per-test configurable response behaviors

6. **Request Logger** - Capture and log all requests for debugging

7. **Network Guard** - Block all non-localhost network calls

**Server Features:**

- Serve RSS feeds from `feeds/` directory (with template support for base URL injection)

- Serve transcripts from episode directories or `transcripts/` directory

- Serve audio files from episode directories or `audio/` directory

- Support various HTTP scenarios (success, errors, timeouts) via configurable behaviors

- Support different RSS feed formats (standard RSS, Podcasting 2.0, iTunes)

- Support different transcript formats (VTT, SRT, JSON, plain text)

- Support different audio formats (MP3, M4A, WAV)

- Support relative URLs in RSS feeds (test app's URL resolution, not server's)

- Support special characters in filenames and URLs (with proper encoding)

- Support error scenarios (404, 500, timeouts) via configurable behaviors

- Support HTTP Range requests (206 Partial Content) for audio files

- Support proper HTTP headers (Content-Length, ETag, Last-Modified, Accept-Ranges)

- Path traversal protection (security)

- Request logging and replay artifacts for debugging

- URL helper methods for convenient test authoring

## Implementation Plan

### Stage 0: Network Guard Setup

**Goal**: Add hard network blocking to prevent accidental external network calls.

**Tasks:**

1. Add `pytest-socket` dependency (or implement monkeypatch for `socket.create_connection`)

2. Configure pytest-socket to block all sockets except localhost/127.0.0.1

3. Add pytest fixture `network_block` that enables socket blocking for E2E tests

4. Test that external network calls fail immediately with clear error message

**Implementation Options:**

**Option A: pytest-socket (recommended)**

```python
# conftest.py or pytest.ini
[pytest]
socket_enabled = false
socket_allow_hosts = ["127.0.0.1", "localhost"]

```

**Option B: Monkeypatch (no dependency)**

```python
# tests/fixtures/e2e_server/network_guard.py
import socket

_original_create_connection = socket.create_connection

def _guarded_create_connection(address, *args, **kwargs):
    host, port = address
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"Network call blocked: attempted to connect to {host}:{port}")
    return _original_create_connection(address, *args, **kwargs)

@pytest.fixture(autouse=True, scope="function")
def network_block(monkeypatch):
    """Block all non-localhost network calls in E2E tests."""
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)

```

**Files to Create:**

- `tests/fixtures/e2e_server/network_guard.py` (if using monkeypatch)

- Update `pyproject.toml` to add `pytest-socket` dependency (if using pytest-socket)

### Stage 1: Basic Server Infrastructure

**Goal**: Create a basic HTTP server that can serve files from a structured folder with security and performance optimizations.

**Tasks:**

1. Create `tests/fixtures/e2e_server/` directory structure

2. Implement `E2EHTTPServer` class (extends `http.server.HTTPServer`)

3. Implement `E2EHTTPRequestHandler` class (extends `http.server.SimpleHTTPRequestHandler`)

4. Implement basic file serving (RSS feeds, transcripts, audio files)

5. Add path traversal protection (resolve paths and verify they're within fixture root)

6. Add URL encoding safety (handle `%2e%2e` and other encoding edge cases)

7. Create pytest fixture `e2e_server` with session scope and function-scoped reset

8. Add basic URL routing (map URLs to file paths)

9. Add URL helper methods (`e2e_server.urls.for_episode()`, etc.)

**Security Implementation:**

```python
def _resolve_path(self, url_path: str) -> Path:
    """Resolve URL path to file path with security checks."""
    # Decode URL encoding
    decoded_path = urllib.parse.unquote(url_path)

    # Resolve to absolute path
    fixture_root = Path(__file__).parent / "e2e_server"
    resolved = (fixture_root / decoded_path.lstrip("/")).resolve()

    # Security check: ensure path is within fixture root
    try:
        resolved.relative_to(fixture_root.resolve())
    except ValueError:
        raise ValueError(f"Path traversal attempt detected: {url_path}")

    return resolved

```

**Fixture Scope Optimization:**

```python
@pytest.fixture(scope="session")
def e2e_server_session():
    """Session-scoped server (one server for entire test run)."""
    server = E2EHTTPServer(port=0)
    server.start()
    yield server
    server.shutdown()

@pytest.fixture(scope="function")
def e2e_server(e2e_server_session):
    """Function-scoped reset (clear behaviors/logs between tests)."""
    e2e_server_session.reset()  # Clear behaviors, logs, etc.
    yield e2e_server_session
    # No shutdown needed (session-scoped)

```

**Files to Create:**

- `tests/fixtures/e2e_server/__init__.py`

- `tests/fixtures/e2e_server/server.py`

- `tests/fixtures/e2e_server/feeds/` (directory, generated by script)

- `tests/fixtures/e2e_server/transcripts/` (directory, if needed)

- `tests/fixtures/e2e_server/audio/` (directory, if needed)

**Example Usage:**

```python
def test_basic_e2e(e2e_server):
    """Test basic E2E workflow with local HTTP server."""
    # Use helper method for URL generation
    rss_url = e2e_server.urls.feed("podcast1")
    # Or: rss_url = f"{e2e_server.base_url}/feeds/podcast1/feed.xml"

    # Run pipeline with real HTTP client
    cfg = Config(rss=rss_url, output_dir=tmpdir)
    result = run_pipeline(cfg)

    assert result.episodes_processed > 0

```

### Stage 2: RSS Feed Support

**Goal**: Add support for serving RSS feeds with proper content types. Use templates for convenience feeds, keep relative URLs in test feeds to exercise app's URL resolution.

**Tasks:**

1. Add RSS feed file serving with `application/xml` content type

2. Support multiple RSS feed formats (standard RSS, Podcasting 2.0, iTunes)

3. Add template support for convenience feeds (inject `{base_url}` at runtime)

4. Keep some feeds with relative URLs as-is (to test app's URL resolution)

5. Add sample RSS feed files to `feeds/` directory

6. Add RSS feed templates to `feeds/templates/` directory

7. Test RSS feed serving

**Template-Based Feed Generation:**

```python
# feeds/templates/podcast1_feed.xml.template
<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <link>{base_url}</link>
    <item>
      <title>Episode 1</title>
      <enclosure url="{base_url}/feeds/podcast1/episodes/episode1/audio.mp3" type="audio/mpeg"/>
      <podcast:transcript url="{base_url}/feeds/podcast1/episodes/episode1/transcript.vtt" type="text/vtt"/>
    </item>
  </channel>
</rss>

```

**Server Template Handling:**

```python
def _serve_rss_template(self, template_path: Path, base_url: str) -> bytes:
    """Load RSS template and inject base_url."""
    template_content = template_path.read_text()
    return template_content.format(base_url=base_url).encode("utf-8")

```

**Test Feeds with Relative URLs (for testing app's resolution):**

```xml
<!-- feeds/edge_cases/relative_urls.xml -->
<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>Episode 1</title>
      <!-- Relative URL - app should resolve this -->
      <enclosure url="../episodes/episode1/audio.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>

```

**Key Principle**: Server does NOT resolve relative URLs. The app under test should handle URL resolution. Templates are only for convenience (avoiding hand-editing base URLs).

### Stage 3: Transcript and Audio File Support with HTTP Behaviors

**Goal**: Add support for serving transcripts and audio files with proper content types and HTTP behaviors (Range requests, headers).

**Tasks:**

1. Add transcript file serving (VTT, SRT, JSON, plain text) with proper content types

2. Add audio file serving (MP3, M4A, WAV) with proper content types

3. Support HTTP Range requests (206 Partial Content) for audio files

4. Add proper HTTP headers:
   - `Content-Length` (file size)
   - `Accept-Ranges: bytes` (for Range request support)
   - `ETag` (optional, for caching tests)
   - `Last-Modified` (optional, for conditional requests)

5. Support streaming for large files (audio files)

6. Add sample transcript files to `episodes/` directories

7. Add sample audio files to `episodes/` directories (small test files)

8. Test transcript and audio file serving

9. Test Range request handling

**Content Type Mapping:**

- `.vtt` → `text/vtt`

- `.srt` → `text/srt`

- `.json` → `application/json`

- `.txt` → `text/plain`

- `.mp3` → `audio/mpeg`

- `.m4a` → `audio/mp4`

- `.wav` → `audio/wav`

**Range Request Implementation:**

```python
def _handle_range_request(self, file_path: Path, range_header: str) -> tuple[bytes, int, dict]:
    """Handle HTTP Range request (206 Partial Content)."""
    file_size = file_path.stat().st_size

    # Parse Range header: "bytes=0-1023"
    match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not match:
        return None, 416, {}  # Range Not Satisfiable

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else file_size - 1

    # Read requested range
    with file_path.open("rb") as f:
        f.seek(start)
        content = f.read(end - start + 1)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(len(content)),
        "Accept-Ranges": "bytes",
    }

    return content, 206, headers  # 206 Partial Content

```

### Stage 4: Configurable Error Scenario Support

**Goal**: Add configurable error scenarios via behavior registry (not just fixed endpoints).

**Tasks:**

1. Implement behavior registry system (per-path configurable response sequences)

2. Support various error scenarios:
   - HTTP status codes (404, 500, 503, etc.)
   - Timeouts (slow responses)
   - Malformed responses
   - Retry scenarios (fail once, then succeed)
   - Throttling (slow streaming)

3. Add convenience methods for common scenarios

4. Add error scenario RSS feeds to `feeds/edge_cases/`

5. Test error handling in E2E tests

**Behavior Registry Implementation:**

```python
from collections import deque
from typing import Callable, Sequence

class ResponseBehavior:
    """Defines a response behavior (status, body, headers, delay)."""
    def __init__(self, status: int = 200, body: bytes = b"", headers: dict = None, delay: float = 0.0):
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.delay = delay

class BehaviorRegistry:
    """Registry for per-path response behaviors."""
    def __init__(self):
        self._behaviors: dict[str, deque] = {}

    def add(self, path: str, behavior: ResponseBehavior | Sequence[ResponseBehavior]):
        """Add behavior(s) for a path."""
        if isinstance(behavior, ResponseBehavior):
            behavior = [behavior]
        self._behaviors[path] = deque(behavior)

    def get(self, path: str) -> ResponseBehavior | None:
        """Get next behavior for a path (pops from queue)."""
        if path not in self._behaviors:
            return None
        behaviors = self._behaviors[path]
        if not behaviors:
            return None
        return behaviors.popleft()

    def clear(self):
        """Clear all behaviors (for test isolation)."""
        self._behaviors.clear()

```

**Usage Examples:**

```python
def test_retry_behavior(e2e_server):
    """Test retry logic: fail once, then succeed."""
    # Configure behavior: fail first request, succeed second
    e2e_server.behaviors.add(
        "/feeds/podcast1/feed.xml",
        [
            ResponseBehavior(status=500, body=b"Internal Server Error"),
            ResponseBehavior(status=200, body=feed_content),
        ]
    )

    # Test that app retries and succeeds

def test_timeout_behavior(e2e_server):
    """Test timeout handling."""
    e2e_server.behaviors.add(
        "/feeds/podcast1/feed.xml",
        ResponseBehavior(status=200, body=feed_content, delay=10.0)  # 10 second delay
    )

    # Test that app handles timeout correctly

def test_slow_streaming(e2e_server):
    """Test slow streaming (throttling)."""
    e2e_server.behaviors.add(
        "/feeds/podcast1/episodes/ep1/audio.mp3",
        ResponseBehavior(status=200, body=audio_content, delay=0.1)  # 100ms per chunk
    )

    # Test that app handles slow streaming

```

**Convenience Methods:**

```python
# In E2EServer class
def route(self, path: str):
    """Return route builder for fluent API."""
    return RouteBuilder(self, path)

# Usage
e2e_server.route("/feeds/podcast1/feed.xml").respond(status=500, body=b"Error")
e2e_server.route("/feeds/podcast1/feed.xml").respond_sequence(
    [ResponseBehavior(status=500), ResponseBehavior(status=200, body=feed_content)]
)

```

### Stage 5: Request Logging and Debugging

**Goal**: Add comprehensive request logging and debugging features.

**Tasks:**

1. Implement request logger (capture all requests: method, path, query, headers, response code, duration)

2. Add request log access methods (`e2e_server.logs.requests`, `e2e_server.logs.for_path()`)

3. Add pytest hook to dump logs on test failure

4. Add replay artifact generation (save request/response pairs for debugging)

5. Add URL helper methods for convenient test authoring:
   - `e2e_server.urls.feed(podcast_name)` → `{base_url}/feeds/{podcast_name}/feed.xml`
   - `e2e_server.urls.episode_transcript(podcast, episode, format)` → transcript URL
   - `e2e_server.urls.episode_audio(podcast, episode, format)` → audio URL

6. Support multiple episodes in a single RSS feed

7. Support concurrent requests (multiple episodes downloading simultaneously)

8. Support special characters in URLs and filenames (with proper encoding)

**Request Logger Implementation:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RequestLog:
    """Log entry for a single HTTP request."""
    method: str
    path: str
    query: Optional[str]
    headers: dict
    response_status: int
    response_size: int
    duration: float
    timestamp: datetime

class RequestLogger:
    """Logger for HTTP requests."""
    def __init__(self):
        self._logs: list[RequestLog] = []

    def log(self, log_entry: RequestLog):
        """Log a request."""
        self._logs.append(log_entry)

    @property
    def requests(self) -> list[RequestLog]:
        """Get all logged requests."""
        return self._logs.copy()

    def for_path(self, path: str) -> list[RequestLog]:
        """Get all requests for a specific path."""
        return [log for log in self._logs if log.path == path]

    def clear(self):
        """Clear all logs (for test isolation)."""
        self._logs.clear()

```

**Pytest Hook for Log Dumping:**

```python
# In conftest.py
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Dump E2E server logs on test failure."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        # Check if test used e2e_server fixture
        if "e2e_server" in item.fixturenames:
            e2e_server = item.funcargs.get("e2e_server")
            if e2e_server and e2e_server.logs.requests:
                print("\n=== E2E Server Request Log ===")
                for log in e2e_server.logs.requests:
                    print(f"{log.method} {log.path} -> {log.response_status} ({log.duration:.3f}s)")
                print("=" * 40)

```

**URL Helper Methods:**

```python
class URLHelper:
    """Helper methods for generating URLs."""
    def __init__(self, base_url: str):
        self.base_url = base_url

    def feed(self, podcast_name: str) -> str:
        """Get RSS feed URL for a podcast."""
        return f"{self.base_url}/feeds/{podcast_name}/feed.xml"

    def episode_transcript(self, podcast: str, episode: str, format: str = "vtt") -> str:
        """Get transcript URL for an episode."""
        return f"{self.base_url}/feeds/{podcast}/episodes/{episode}/transcript.{format}"

    def episode_audio(self, podcast: str, episode: str, format: str = "mp3") -> str:
        """Get audio URL for an episode."""
        return f"{self.base_url}/feeds/{podcast}/episodes/{episode}/audio.{format}"

```

### Stage 6: Test Data Collection and Fixture Creation

**Goal**: Create and organize test data fixtures manually.

**Tasks:**

1. Create `tests/fixtures/e2e_server/` directory structure

2. Write RSS feed files with proper URLs (absolute URLs for convenience, relative URLs for testing app resolution)

3. Create or collect small audio files (< 10 seconds) for Whisper testing

4. Write sample transcript files (VTT, SRT, JSON, plain text formats)

5. Organize files in podcast/episode structure matching server routing

6. Create optional `manifest.json` for documentation/reference

7. Write `tests/fixtures/e2e_server/README.md` documenting fixture structure

**Test Data Requirements:**

- **RSS Feeds**: 3-5 feeds covering different scenarios
  - Basic feed with absolute URLs (e.g., `http://localhost:8000/feeds/podcast1/episodes/ep1/audio.mp3`)
  - Feed with relative URLs (e.g., `../episodes/ep1/audio.mp3`) - tests app's URL resolution
  - Multi-episode feed (multiple episodes in one feed)
  - Feed with missing transcripts (no transcript URLs)
  - Feed with special characters in titles/URLs

- **Audio Files**: 2-3 small test files (< 10 seconds each, MP3/M4A format)

- **Transcripts**: Samples in VTT, SRT, JSON, plain text formats

- **Edge Cases**: Pre-written malformed RSS, missing transcript scenarios

**RSS Feed Example (Absolute URLs):**

```xml
<!-- tests/fixtures/e2e_server/feeds/podcast1/feed.xml -->
<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast 1</title>
    <link>http://localhost:8000</link>
    <description>A test podcast for E2E testing</description>
    <item>
      <title>Episode 1: Introduction</title>
      <description>First episode of the podcast</description>
      <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
      <enclosure url="http://localhost:8000/feeds/podcast1/episodes/episode1/audio.mp3"
                 type="audio/mpeg" length="136000"/>
      <podcast:transcript url="http://localhost:8000/feeds/podcast1/episodes/episode1/transcript.vtt"
                          type="text/vtt"/>
    </item>
  </channel>
</rss>

```

**RSS Feed Example (Relative URLs - for testing app resolution):**

```xml
<!-- tests/fixtures/e2e_server/feeds/podcast2/feed.xml -->
<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast 2</title>
    <item>
      <title>Episode 1</title>
      <!-- Relative URL - app should resolve this -->
      <enclosure url="../episodes/episode1/audio.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>

```

**Note**: When creating fixtures manually, use `http://localhost:8000` as the base URL in RSS feeds. The server will serve on a dynamically assigned port, but tests can use the server's `base_url` property to construct correct URLs. For relative URLs, the app under test should handle resolution.

### Stage 7: E2E Test Migration

**Goal**: Migrate existing E2E tests to use the HTTP server instead of mocks.

**Tasks:**

1. Update `test_workflow_e2e.py` to use HTTP server

2. Update `test_service.py` to use HTTP server

3. Update `test_cli.py` to use HTTP server

4. Remove `unittest.mock.patch` for `fetch_url` in E2E tests

5. Add new E2E tests for error scenarios

6. Add new E2E tests for multi-episode workflows

7. Verify all E2E tests pass with HTTP server

**Migration Pattern:**

```python
# Before (mocked):
def test_e2e_transcript_download(self):
    responses = {
        downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        downloader.normalize_url(transcript_url): create_transcript_response(transcript_text, transcript_url),
    }
    with patch("podcast_scraper.downloader.fetch_url", side_effect=self._mock_http_map(responses)):
        # Test code

# After (HTTP server):
def test_e2e_transcript_download(self, e2e_server):
    base_url = e2e_server.base_url
    rss_url = f"{base_url}/feeds/podcast1/feed.xml"
    # Test code (no mocking needed)

```

### Stage 8: Documentation and Examples

**Goal**: Document the HTTP server infrastructure and provide examples.

**Tasks:**

1. Document server architecture and folder structure

2. Document how to add new test data

3. Document how to use the server in E2E tests

4. Add examples of common test scenarios

5. Update `TESTING_STRATEGY.md` with E2E test infrastructure details

## Technical Details

### Server Implementation

**Base Class**: `http.server.HTTPServer`
**Handler Class**: `http.server.SimpleHTTPRequestHandler` (customized)

**Note on Library Choice**: We use `http.server` for consistency with integration tests. If we need more advanced features later, we can consider `pytest-httpserver` or `aiohttp`, but `http.server` is sufficient for our needs and keeps dependencies minimal.

**Key Methods:**

- `do_GET()` - Handle GET requests (with behavior registry check, Range request support, logging)

- `_send_response()` - Send HTTP response with proper headers

- `_resolve_path()` - Resolve URL path to file path (with security checks)

- `_get_content_type()` - Determine content type from file extension

- `_handle_range_request()` - Handle HTTP Range requests (206 Partial Content)

- `_serve_file()` - Serve file with proper headers (Content-Length, ETag, Last-Modified)

- `_check_behavior_registry()` - Check if path has configured behavior

- `_log_request()` - Log request for debugging

**URL Routing:**

- `/feeds/{podcast_name}/feed.xml` → `e2e_server/feeds/{podcast_name}/feed.xml` (generated from template)

- `/feeds/{podcast_name}/episodes/{episode_name}/transcript.{ext}` → `e2e_server/feeds/{podcast_name}/episodes/{episode_name}/transcript.{ext}` (symlink/copy from `fixtures/transcripts/`)

- `/feeds/{podcast_name}/episodes/{episode_name}/audio.{ext}` → `e2e_server/feeds/{podcast_name}/episodes/{episode_name}/audio.{ext}` (symlink/copy from `fixtures/audio/`)

- Behavior registry can override any path with custom response

**Server Path Resolution:**

- Server serves from `tests/fixtures/e2e_server/` directory (manually maintained structure)

- Files are organized by podcast/episode structure

- All files are directly in the structure (no symlinks needed)

**Security Implementation:**

- Path traversal protection: resolve to absolute path and verify it's within fixture root

- URL encoding safety: handle `%2e%2e` and other encoding edge cases

- Use `pathlib.Path` for all path operations (cross-platform safety)

### Pytest Fixture

**Fixture Name**: `e2e_server`
**Scope**: `session` for server, `function` for reset (optimized for performance)
**Returns**: `E2EServer` object with:

- `base_url` - Base URL of the server (e.g., `http://localhost:8000`)

- `server` - HTTP server instance

- `port` - Server port number

- `behaviors` - Behavior registry (for configurable responses)

- `logs` - Request logger (for debugging)

- `urls` - URL helper (for convenient URL generation)

- `reset()` - Method to reset behaviors and logs (for test isolation)

- `shutdown()` - Method to shutdown the server (called automatically)

**Example:**

```python
@pytest.fixture(scope="session")
def e2e_server_session():
    """Session-scoped server (one server for entire test run)."""
    server = E2EHTTPServer(port=0)  # Port 0 = auto-assign
    server.start()
    yield server
    server.shutdown()

@pytest.fixture(scope="function")
def e2e_server(e2e_server_session):
    """Function-scoped reset (clear behaviors/logs between tests)."""
    e2e_server_session.reset()  # Clear behaviors, logs, etc.
    yield e2e_server_session
    # No shutdown needed (session-scoped)

```

**Usage Example:**

```python
def test_e2e_with_behavior(e2e_server):
    """Test with configured behavior."""
    # Configure error behavior
    e2e_server.behaviors.add(
        "/feeds/podcast1/feed.xml",
        ResponseBehavior(status=500, body=b"Error")
    )

    # Use URL helper
    rss_url = e2e_server.urls.feed("podcast1")

    # Run test
    cfg = Config(rss=rss_url, output_dir=tmpdir)
    result = run_pipeline(cfg)

    # Check logs
    assert len(e2e_server.logs.for_path("/feeds/podcast1/feed.xml")) > 0

```

### Test Data Organization

**Fixture Structure (Manually Maintained):**

- **RSS Feeds**: `tests/fixtures/e2e_server/feeds/{podcast_name}/feed.xml` - Manually written RSS feeds

- **Episode Files**: `tests/fixtures/e2e_server/feeds/{podcast_name}/episodes/{episode_name}/` - Transcript and audio files

- **Transcripts**: Located in episode directories or `tests/fixtures/e2e_server/transcripts/`

- **Audio Files**: Located in episode directories or `tests/fixtures/e2e_server/audio/`

- **Manifest**: `tests/fixtures/e2e_server/manifest.json` - Optional documentation/reference

- **Documentation**: `tests/fixtures/e2e_server/README.md` - How to add/modify fixtures

**Workflow:**

1. Developer manually creates RSS feeds, transcripts, and audio files in the fixture structure

2. Files are organized to match server routing (`/feeds/{podcast}/feed.xml`, `/feeds/{podcast}/episodes/{episode}/transcript.vtt`, etc.)

3. Server serves directly from `tests/fixtures/e2e_server/` directory

4. Tests use server URLs (structure matches routing)

**Benefits:**

- **Simplicity**: No generation step, fixtures are directly maintained

- **Transparency**: All fixture files visible in version control

- **Flexibility**: Easy to add new fixtures or modify existing ones

- **Version Control**: All fixtures checked into git

## Benefits

1. **Real HTTP Client Testing**: E2E tests use real HTTP client without mocking

2. **Real Data Testing**: E2E tests use real RSS feeds, transcripts, and audio files

3. **No External Network**: E2E tests don't hit external networks

4. **Reusable**: Server can be used across all E2E tests

5. **Flexible**: Server supports various scenarios (success, errors, edge cases)

6. **Maintainable**: Test data is organized in a clear folder structure

7. **Fast**: Local server is fast and reliable

8. **Deterministic**: Tests are deterministic (no external network flakiness)

## Challenges and Solutions

### Challenge 1: Port Conflicts

**Solution**: Use port 0 (auto-assign) or find available port dynamically

### Challenge 2: File Path Resolution

**Solution**: Use `pathlib.Path` for cross-platform path handling

### Challenge 3: Relative URL Resolution

**Solution**: Use templates for convenience feeds (inject base_url at runtime). Keep some feeds with relative URLs as-is to test app's URL resolution. Server does NOT resolve relative URLs.

### Challenge 4: Large Audio Files

**Solution**: Use small test files (< 10 seconds) and support streaming for larger files

### Challenge 5: Test Data Collection

**Solution**: Use public domain or create synthetic test data

## Success Criteria

1. ✅ E2E tests use real HTTP client (no mocking of `fetch_url`)

2. ✅ E2E tests use real RSS feed files (with template support)

3. ✅ E2E tests use real audio files (small test files)

4. ✅ E2E tests use real transcript files (VTT, SRT, JSON)

5. ✅ Server supports various scenarios via configurable behavior registry

6. ✅ Server is reusable across all E2E tests (session-scoped with function-scoped reset)

7. ✅ Test data is organized in a clear folder structure

8. ✅ Network guard prevents accidental external network calls

9. ✅ Path traversal protection prevents security issues

10. ✅ Request logging and debugging features available

11. ✅ HTTP Range requests supported for audio files

12. ✅ All existing E2E tests pass with HTTP server

13. ✅ New E2E tests added for error scenarios and multi-episode workflows

14. ✅ Documentation and examples provided

## Timeline

- **Stage 0**: Network guard setup (0.5 day)

- **Stage 1**: Basic server infrastructure with security (1-2 days)

- **Stage 2**: RSS feed support with templates (1 day)

- **Stage 3**: Transcript and audio file support with HTTP behaviors (1-2 days)

- **Stage 4**: Configurable error scenario support (1-2 days)

- **Stage 5**: Request logging and debugging (1 day)

- **Stage 6**: Test data collection (1-2 days)

- **Stage 7**: E2E test migration (2-3 days)

- **Stage 8**: Documentation (1 day)

**Total Estimated Time**: 9-14 days

## Key Improvements from Peer Review

This plan incorporates the following improvements based on peer review feedback:

1. **Hard Network Guard (Stage 0)**: Added `pytest-socket` or monkeypatch to block all non-localhost network calls, preventing accidental external network access.

2. **Security Hardening (Stage 1)**:
   - Path traversal protection (verify paths are within fixture root)
   - URL encoding safety (handle `%2e%2e` and other edge cases)
   - Use `pathlib.Path` for cross-platform safety

3. **Template-Based RSS Feeds (Stage 2)**:
   - Use templates for convenience feeds (inject `{base_url}` at runtime)
   - Keep relative URLs in test feeds to exercise app's URL resolution
   - Server does NOT resolve relative URLs (app should handle it)

4. **HTTP Behaviors for Audio Realism (Stage 3)**:
   - Range request support (206 Partial Content)
   - Proper HTTP headers (Content-Length, ETag, Last-Modified, Accept-Ranges)
   - Future-proof for chunked downloads

5. **Configurable Error Scenarios (Stage 4)**:
   - Behavior registry system (per-path configurable response sequences)
   - Support retry scenarios (fail once, then succeed)
   - Support throttling (slow streaming)
   - More flexible than fixed `/error/*` endpoints

6. **Request Logging and Debugging (Stage 5)**:
   - Comprehensive request logging (method, path, query, headers, response, duration)
   - Pytest hook to dump logs on test failure
   - URL helper methods for convenient test authoring
   - Replay artifacts for troubleshooting

7. **Performance Optimization**:
   - Session-scoped server (one server for entire test run)
   - Function-scoped reset (clear behaviors/logs between tests)
   - Faster than creating new server for each test

8. **Library Choice Rationale**:
   - Use `http.server` for consistency with integration tests
   - Note that `pytest-httpserver` could be considered if more advanced features are needed
   - Keep dependencies minimal

## Next Steps

1. Review and approve this evolved plan

2. Start with Stage 0 (Network Guard Setup)

3. Iterate through stages, testing as we go

4. Migrate E2E tests incrementally

5. Document and provide examples
