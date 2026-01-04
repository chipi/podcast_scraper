"""Pytest configuration for unit tests.

This module enforces network and filesystem I/O isolation for unit tests by
detecting and blocking any network calls and filesystem operations made during
test execution.

Network and filesystem I/O in unit tests are prohibited because:
1. Unit tests should be fast and isolated
2. Network/filesystem operations introduce flakiness and external dependencies
3. All I/O interactions should be mocked

Exceptions:
- tempfile operations are allowed (designed for testing)
- Operations within temp directories are allowed (detected automatically)
- test_filesystem.py tests are allowed (they need to test filesystem operations)

Integration and e2e tests are allowed to make network/filesystem calls
(if marked appropriately).

This conftest extends the main conftest.py by adding network and I/O isolation.
All helper functions from the main conftest are available via pytest's conftest
resolution.
"""

# Import all helper functions from parent conftest
# pytest automatically loads conftest.py files from parent directories,
# but we explicitly import here to make them available for direct imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent tests directory to path to import from main conftest
parent_tests_dir = Path(__file__).parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

# Import helper functions from main conftest
try:
    from conftest import (  # noqa: F401
        build_rss_xml_with_media,
        build_rss_xml_with_speakers,
        build_rss_xml_with_transcript,
        create_media_response,
        create_mock_spacy_model,
        create_rss_response,
        create_test_args,
        create_test_config,
        create_test_episode,
        create_test_feed,
        create_transcript_response,
        MockHTTPResponse,
        TEST_BASE_URL,
        TEST_CONTENT_TYPE_SRT,
        TEST_CONTENT_TYPE_VTT,
        TEST_CUSTOM_OUTPUT_DIR,
        TEST_EPISODE_TITLE,
        TEST_EPISODE_TITLE_SPECIAL,
        TEST_FEED_TITLE,
        TEST_FEED_URL,
        TEST_FULL_URL,
        TEST_MEDIA_TYPE_M4A,
        TEST_MEDIA_TYPE_MP3,
        TEST_MEDIA_URL,
        TEST_OUTPUT_DIR,
        TEST_PATH,
        TEST_RELATIVE_MEDIA,
        TEST_RELATIVE_TRANSCRIPT,
        TEST_RUN_ID,
        TEST_TRANSCRIPT_TYPE_SRT,
        TEST_TRANSCRIPT_TYPE_VTT,
        TEST_TRANSCRIPT_URL,
        TEST_TRANSCRIPT_URL_SRT,
    )
except ImportError:
    # If import fails, pytest will still load parent conftest automatically
    pass


class NetworkCallDetectedError(Exception):
    """Raised when a unit test attempts to make a network call."""

    def __init__(self, library_name: str, call_type: str):
        self.library_name = library_name
        self.call_type = call_type
        super().__init__(
            f"Network call detected in unit test: {library_name}.{call_type}()\n"
            f"Unit tests must not make network calls. Use mocks instead.\n"
            f"If this test needs network access, it should be moved to "
            f"integration/ or e2e/."
        )


class FilesystemIODetectedError(Exception):
    """Raised when a unit test attempts to perform filesystem I/O."""

    def __init__(self, operation: str, path: str | None = None):
        self.operation = operation
        self.path = path
        path_info = f" (path: {path})" if path else ""
        super().__init__(
            f"Filesystem I/O detected in unit test: {operation}{path_info}\n"
            f"Unit tests must not perform filesystem I/O. Use mocks or "
            f"tempfile operations instead.\n"
            f"Allowed: tempfile.mkdtemp(), tempfile.NamedTemporaryFile(), "
            f"operations within temp directories.\n"
            f"If this test needs filesystem access, it should be moved to "
            f"integration/ or e2e/."
        )


def _is_temp_path(path: str) -> bool:  # noqa: C901
    """Check if a path is within a temporary directory or allowed cache directory."""
    if not path:
        return False
    try:
        # Resolve to absolute path for comparison
        path_obj = Path(path)
        if not path_obj.is_absolute():
            # Resolve relative paths
            try:
                path_obj = path_obj.resolve()
            except (OSError, RuntimeError):
                # If resolve fails, use the path as-is
                pass

        path_str = str(path_obj)
        path_lower = path_str.lower()

        # Check common temp patterns (case-insensitive for Windows compatibility)
        temp_patterns = [
            "/tmp/",  # nosec B108 - intentional for test isolation detection
            "/var/tmp/",  # nosec B108 - intentional for test isolation detection
            "/var/folders/",  # macOS temp directories
            tempfile.gettempdir(),
        ]

        # Normalize paths for comparison
        for temp_pattern in temp_patterns:
            if temp_pattern:
                temp_str = str(temp_pattern).lower()
                if temp_str in path_lower or path_lower.startswith(temp_str):
                    return True

        # Check if path is under any temp directory
        temp_dirs = [
            Path(tempfile.gettempdir()),
            Path("/tmp"),  # nosec B108 - intentional for test isolation detection
            Path("/var/tmp"),  # nosec B108 - intentional for test isolation detection
        ]

        for temp_dir in temp_dirs:
            try:
                if path_obj.is_relative_to(temp_dir):
                    return True
            except (ValueError, AttributeError):
                # Python < 3.9 doesn't have is_relative_to
                try:
                    path_obj.relative_to(temp_dir)
                    return True
                except ValueError:
                    pass

        # Allow cache directories (for model loading, etc.)
        # These are read-only or write-once operations that don't affect test isolation
        cache_patterns = [
            "/.cache/",
            "/.local/share/",
            "/.local/cache/",
            "/Library/Caches/",  # macOS
            "\\AppData\\Local\\",  # Windows
        ]

        for cache_pattern in cache_patterns:
            if cache_pattern.lower() in path_lower:
                return True

        # Allow reading from site-packages (for installed models like spaCy)
        # This is read-only access to installed packages
        site_packages_patterns = [
            "/site-packages/",
            "/dist-packages/",  # Debian/Ubuntu
        ]
        for pattern in site_packages_patterns:
            if pattern.lower() in path_lower:
                return True

        # Allow Python cache files (.pyc, __pycache__)
        # These are automatically created by Python during import
        python_cache_patterns = [
            ".pyc",
            "__pycache__",
            ".pyo",
            ".pyd",
        ]
        for pattern in python_cache_patterns:
            if pattern.lower() in path_lower:
                return True

        # Check if path is under common cache directories
        try:
            import platformdirs

            cache_dirs = [
                Path(platformdirs.user_cache_dir()),
                Path(platformdirs.user_data_dir()),
            ]
            for cache_dir in cache_dirs:
                try:
                    if path_obj.is_relative_to(cache_dir):
                        return True
                except (ValueError, AttributeError):
                    try:
                        path_obj.relative_to(cache_dir)
                        return True
                    except ValueError:
                        pass
        except (ImportError, Exception):
            pass  # platformdirs might not be available

    except (OSError, ValueError, RuntimeError):
        pass
    return False


def _is_filesystem_test(request) -> bool:
    """Check if the current test is in test_filesystem.py (not test_filesystem_isolation.py)."""
    if hasattr(request, "node"):
        # Get the test file path
        test_file = getattr(request.node, "fspath", None) or getattr(request.node, "path", None)
        if test_file:
            test_file_str = str(test_file)
            # Check for exact match: test_filesystem.py (not test_filesystem_isolation.py)
            return (
                "test_filesystem.py" in test_file_str
                and "test_filesystem_isolation" not in test_file_str
            )
    return False


def _create_network_blocker(library_name: str, call_type: str):
    """Create a function that blocks network calls and raises an error."""

    def blocker(*args, **kwargs):
        raise NetworkCallDetectedError(library_name, call_type)

    return blocker


def _create_filesystem_blocker(operation: str, request):
    """Create a function that blocks filesystem I/O and raises an error."""

    def blocker(*args, **kwargs):
        # Check if this is test_filesystem.py (allowed exception)
        if _is_filesystem_test(request):
            # Allow filesystem operations in test_filesystem.py
            # Call the original function
            import builtins

            return getattr(builtins, operation)(*args, **kwargs)

        # Check if path is in temp directory
        path = None
        if args:
            path = args[0] if isinstance(args[0], (str, Path)) else str(args[0])
        elif "path" in kwargs:
            path = kwargs["path"]
        elif "file" in kwargs:
            path = kwargs["file"]

        if path and _is_temp_path(str(path)):
            # Allow operations within temp directories
            import builtins

            return getattr(builtins, operation)(*args, **kwargs)

        raise FilesystemIODetectedError(operation, path)

    return blocker


def _is_unit_test(request) -> bool:
    """Check if the current test is in the unit/ directory.

    Returns True only for tests in tests/unit/ directory.
    Returns False for tests in tests/integration/ or tests/e2e/.

    Uses multiple detection methods for reliability across different
    pytest configurations (local, CI, parallel with xdist).
    """
    # Method 1: Check nodeid (most reliable, always available)
    # nodeid format: "tests/unit/podcast_scraper/test_config.py::TestConfig::test_method"
    if hasattr(request, "node"):
        nodeid = getattr(request.node, "nodeid", "")
        if nodeid:
            # Explicitly check for non-unit test paths first
            if "tests/integration/" in nodeid or "tests/e2e/" in nodeid:
                return False
            if "tests/unit/" in nodeid:
                return True

    # Method 2: Check fspath/path attributes
    if hasattr(request, "node"):
        test_file = getattr(request.node, "fspath", None) or getattr(request.node, "path", None)
        if test_file:
            test_file_str = str(test_file)
            # Explicitly check for non-unit test paths first
            if "/tests/integration/" in test_file_str or "\\tests\\integration\\" in test_file_str:
                return False
            if "/tests/e2e/" in test_file_str or "\\tests\\e2e\\" in test_file_str:
                return False
            # Check if the test is in tests/unit/ directory
            if "/tests/unit/" in test_file_str or "\\tests\\unit\\" in test_file_str:
                return True

    # Method 3: Check module name as fallback
    if hasattr(request, "module"):
        module_file = getattr(request.module, "__file__", "")
        if module_file:
            if "/tests/integration/" in module_file or "/tests/e2e/" in module_file:
                return False
            if "/tests/unit/" in module_file:
                return True

    # Default: assume NOT a unit test (safer - don't block network for unknown tests)
    return False


@pytest.fixture(autouse=True)
def block_network_and_filesystem_io(request):  # noqa: C901
    """Automatically block network calls and filesystem I/O in unit tests.

    This fixture patches common network libraries and filesystem operations
    to detect and block any I/O made during unit test execution.

    Exceptions:
    - tempfile operations are allowed (designed for testing)
    - Operations within temp directories are allowed
    - test_filesystem.py tests are allowed (they need to test filesystem operations)

    Only applies to tests in the unit/ directory.
    """
    # Skip this fixture for non-unit tests (integration, e2e)
    # This is important when running all tests together (e.g., nightly builds)
    if not _is_unit_test(request):
        yield
        return

    # Create blockers for common network libraries
    network_patches = []
    filesystem_patches = []

    # Block requests library
    try:
        import requests

        # Patch requests.get, requests.post, requests.put, requests.delete, etc.
        for method in ["get", "post", "put", "delete", "head", "options", "patch"]:
            if hasattr(requests, method):
                patcher = patch.object(
                    requests, method, side_effect=_create_network_blocker("requests", method)
                )
                network_patches.append(patcher)
                patcher.start()

        # Block requests.Session methods
        original_session_init = requests.Session.__init__

        def patched_session_init(self, *args, **kwargs):
            original_session_init(self, *args, **kwargs)
            # Patch all HTTP methods on the session
            for method in ["get", "post", "put", "delete", "head", "options", "patch"]:
                if hasattr(self, method):
                    setattr(self, method, _create_network_blocker("requests.Session", method))

        session_patcher = patch.object(requests.Session, "__init__", patched_session_init)
        network_patches.append(session_patcher)
        session_patcher.start()

    except ImportError:
        pass  # requests not available, skip

    # Block urllib3
    try:
        import urllib3

        # Block urllib3.PoolManager
        pool_manager_patcher = patch.object(
            urllib3, "PoolManager", side_effect=_create_network_blocker("urllib3", "PoolManager")
        )
        network_patches.append(pool_manager_patcher)
        pool_manager_patcher.start()

        # Block urllib3.request
        if hasattr(urllib3, "request"):
            request_patcher = patch.object(
                urllib3, "request", side_effect=_create_network_blocker("urllib3", "request")
            )
            network_patches.append(request_patcher)
            request_patcher.start()

    except ImportError:
        pass  # urllib3 not available, skip

    # Block urllib (standard library)
    try:
        import urllib.request

        urlopen_patcher = patch.object(
            urllib.request,
            "urlopen",
            side_effect=_create_network_blocker("urllib.request", "urlopen"),
        )
        network_patches.append(urlopen_patcher)
        urlopen_patcher.start()
    except ImportError:
        pass  # urllib not available (shouldn't happen in Python 3)

    # Block socket (low-level network access)
    try:
        import socket

        # Block socket.create_connection (most common way to create socket connections)
        create_connection_patcher = patch.object(
            socket,
            "create_connection",
            side_effect=_create_network_blocker("socket", "create_connection"),
        )
        network_patches.append(create_connection_patcher)
        create_connection_patcher.start()

    except ImportError:
        pass  # socket not available (shouldn't happen)

    # Block filesystem I/O operations
    # Note: We allow tempfile operations and operations within temp directories
    try:
        import builtins
        import shutil

        # Block builtin open() for file operations
        original_open = builtins.open

        def patched_open(file, *args, **kwargs):
            # Allow test_filesystem.py first (before any path checks)
            if _is_filesystem_test(request):
                return original_open(file, *args, **kwargs)

            # Get file path for checking
            file_path = None
            if hasattr(file, "name"):
                file_path = str(file.name)
            elif isinstance(file, (str, Path)):
                file_path = str(file)

            # Resolve to absolute path for checking
            if file_path:
                try:
                    resolved_path = Path(file_path).resolve()
                    file_path = str(resolved_path)
                except (OSError, RuntimeError):
                    # If resolve fails, use original path
                    pass

            # Allow tempfile operations (check if path is in temp directory)
            if file_path and _is_temp_path(file_path):
                return original_open(file, *args, **kwargs)

            # Block all other file operations
            raise FilesystemIODetectedError("open()", file_path or str(file))

        # Patch builtins.open - use string form to ensure it patches correctly
        # This ensures all references to open() are patched, not just builtins.open
        open_patcher = patch("builtins.open", patched_open)
        filesystem_patches.append(open_patcher)
        open_patcher.start()

        # Also patch __builtin__ for Python 2 compatibility (though we're Python 3 only)
        # This ensures the patch is applied globally
        try:
            import __builtin__ as builtin_module

            builtin_open_patcher = patch.object(builtin_module, "open", patched_open)
            filesystem_patches.append(builtin_open_patcher)
            builtin_open_patcher.start()
        except ImportError:
            pass  # Python 3 doesn't have __builtin__

        # Block os filesystem operations
        import os

        os_operations = [
            "makedirs",
            "mkdir",
            "remove",
            "unlink",
            "rmdir",
            "rename",
            "chmod",
            "chown",
        ]
        for op in os_operations:
            if hasattr(os, op):
                original_op = getattr(os, op)

                def create_os_blocker(operation_name, original_func, req):
                    def blocker(*args, **kwargs):
                        if _is_filesystem_test(req):
                            return original_func(*args, **kwargs)
                        # If dir_fd is provided, this is a relative path operation within a
                        # directory context (e.g., during shutil.rmtree() cleanup of temp dirs).
                        # Allow these operations for unlink/rmdir since they're cleanup operations.
                        if "dir_fd" in kwargs and operation_name in ("rmdir", "unlink"):
                            return original_func(*args, **kwargs)
                        path = args[0] if args else None
                        if path:
                            path_str = str(path)
                            # Try to resolve relative paths
                            try:
                                path_obj = Path(path_str)
                                if not path_obj.is_absolute():
                                    # Resolve relative to current working directory
                                    path_obj = path_obj.resolve()
                                path_str = str(path_obj)
                                # Check if the path is in a temp directory
                                if _is_temp_path(path_str):
                                    return original_func(*args, **kwargs)
                                # Check if the path exists and is in a temp directory
                                if path_obj.exists() and _is_temp_path(path_str):
                                    return original_func(*args, **kwargs)
                                # For rmdir/unlink operations during cleanup, check if path would
                                # be in temp by checking if any parent directory is a temp
                                # directory. This handles cases where shutil.rmtree() uses
                                # relative paths with dir_fd during temp directory cleanup
                                if operation_name in ("rmdir", "unlink"):
                                    try:
                                        # Check all parent directories
                                        for parent in path_obj.parents:
                                            if parent.exists() and _is_temp_path(str(parent)):
                                                return original_func(*args, **kwargs)
                                        # If path doesn't exist (being cleaned up), check if any
                                        # existing parent is in temp
                                        if not path_obj.exists():
                                            for parent in path_obj.parents:
                                                if parent.exists():
                                                    # Check if this parent is in a temp directory
                                                    if _is_temp_path(str(parent)):
                                                        return original_func(*args, **kwargs)
                                    except (OSError, RuntimeError):
                                        pass
                            except (OSError, RuntimeError):
                                # If we can't resolve or check, allow if path string suggests temp
                                if _is_temp_path(path_str):
                                    return original_func(*args, **kwargs)
                            # Check the path string itself
                            if _is_temp_path(path_str):
                                return original_func(*args, **kwargs)
                        raise FilesystemIODetectedError(
                            f"os.{operation_name}()", str(path) if path else None
                        )

                    return blocker

                patcher = patch.object(os, op, create_os_blocker(op, original_op, request))
                filesystem_patches.append(patcher)
                patcher.start()

        # Block shutil operations
        shutil_operations = ["copy", "copy2", "copyfile", "copytree", "move", "rmtree", "remove"]
        for op in shutil_operations:
            if hasattr(shutil, op):
                original_op = getattr(shutil, op)

                def create_shutil_blocker(operation_name, original_func, req):
                    def blocker(*args, **kwargs):
                        if _is_filesystem_test(req):
                            return original_func(*args, **kwargs)
                        path = args[0] if args else None
                        if path:
                            path_str = str(path)
                            # Try to resolve relative paths
                            try:
                                path_obj = Path(path_str)
                                if not path_obj.is_absolute():
                                    # Resolve relative to current working directory
                                    path_obj = path_obj.resolve()
                                path_str = str(path_obj)
                                # Check if the path is in a temp directory
                                if _is_temp_path(path_str):
                                    return original_func(*args, **kwargs)
                                # Also check if the path exists and is in temp
                                # (needed for cleanup operations where path might be removed)
                                if path_obj.exists() and _is_temp_path(path_str):
                                    return original_func(*args, **kwargs)
                                # Check parent directories (for relative paths during cleanup)
                                # This handles cases where shutil.rmtree() uses relative paths
                                # with dir_fd during temp directory cleanup
                                try:
                                    for parent in path_obj.parents:
                                        if _is_temp_path(str(parent)):
                                            return original_func(*args, **kwargs)
                                except (OSError, RuntimeError):
                                    pass
                            except (OSError, RuntimeError):
                                # If we can't resolve or check, allow if path string suggests temp
                                if _is_temp_path(path_str):
                                    return original_func(*args, **kwargs)
                            # Check the path string itself
                            if _is_temp_path(path_str):
                                return original_func(*args, **kwargs)
                        raise FilesystemIODetectedError(
                            f"shutil.{operation_name}()", str(path) if path else None
                        )

                    return blocker

                patcher = patch.object(shutil, op, create_shutil_blocker(op, original_op, request))
                filesystem_patches.append(patcher)
                patcher.start()

        # Block pathlib.Path write operations
        original_path_write_text = Path.write_text
        original_path_write_bytes = Path.write_bytes
        original_path_mkdir = Path.mkdir
        original_path_unlink = Path.unlink
        original_path_rmdir = Path.rmdir

        def patched_write_text(self, *args, **kwargs):
            req = request  # Capture request in closure
            if _is_filesystem_test(req):
                return original_path_write_text(self, *args, **kwargs)
            if _is_temp_path(str(self)):
                return original_path_write_text(self, *args, **kwargs)
            raise FilesystemIODetectedError("Path.write_text()", str(self))

        def patched_write_bytes(self, *args, **kwargs):
            req = request  # Capture request in closure
            if _is_filesystem_test(req):
                return original_path_write_bytes(self, *args, **kwargs)
            if _is_temp_path(str(self)):
                return original_path_write_bytes(self, *args, **kwargs)
            raise FilesystemIODetectedError("Path.write_bytes()", str(self))

        def patched_mkdir(self, *args, **kwargs):
            req = request  # Capture request in closure
            if _is_filesystem_test(req):
                return original_path_mkdir(self, *args, **kwargs)
            if _is_temp_path(str(self)):
                return original_path_mkdir(self, *args, **kwargs)
            raise FilesystemIODetectedError("Path.mkdir()", str(self))

        def patched_unlink(self, *args, **kwargs):
            req = request  # Capture request in closure
            if _is_filesystem_test(req):
                return original_path_unlink(self, *args, **kwargs)
            if _is_temp_path(str(self)):
                return original_path_unlink(self, *args, **kwargs)
            raise FilesystemIODetectedError("Path.unlink()", str(self))

        def patched_rmdir(self, *args, **kwargs):
            req = request  # Capture request in closure
            if _is_filesystem_test(req):
                return original_path_rmdir(self, *args, **kwargs)
            if _is_temp_path(str(self)):
                return original_path_rmdir(self, *args, **kwargs)
            raise FilesystemIODetectedError("Path.rmdir()", str(self))

        path_write_text_patcher = patch.object(Path, "write_text", patched_write_text)
        path_write_bytes_patcher = patch.object(Path, "write_bytes", patched_write_bytes)
        path_mkdir_patcher = patch.object(Path, "mkdir", patched_mkdir)
        path_unlink_patcher = patch.object(Path, "unlink", patched_unlink)
        path_rmdir_patcher = patch.object(Path, "rmdir", patched_rmdir)

        filesystem_patches.extend(
            [
                path_write_text_patcher,
                path_write_bytes_patcher,
                path_mkdir_patcher,
                path_unlink_patcher,
                path_rmdir_patcher,
            ]
        )

        for patcher in filesystem_patches[-5:]:
            patcher.start()

    except ImportError:
        pass  # Some modules might not be available

    # Yield control to the test
    yield

    # Clean up all patches
    for patcher in network_patches + filesystem_patches:
        try:
            patcher.stop()
        except Exception:  # nosec B110 - intentional: ignore cleanup errors
            pass  # Ignore errors during cleanup
