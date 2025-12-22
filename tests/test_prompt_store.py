#!/usr/bin/env python3
"""Tests for prompt_store module."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper.prompt_store import (
    clear_cache,
    get_prompt_dir,
    get_prompt_metadata,
    get_prompt_source,
    hash_text,
    PromptNotFoundError,
    render_prompt,
    set_prompt_dir,
)


class TestPromptStore(unittest.TestCase):
    """Test prompt_store module functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompt_dir = Path(self.temp_dir) / "prompts"
        self.prompt_dir.mkdir(parents=True)
        self.original_dir = get_prompt_dir()

    def tearDown(self):
        """Clean up test fixtures."""
        set_prompt_dir(self.original_dir)
        clear_cache()

    def test_set_and_get_prompt_dir(self):
        """Test setting and getting prompt directory."""
        test_dir = Path(self.temp_dir) / "custom_prompts"
        test_dir.mkdir()

        set_prompt_dir(test_dir)
        self.assertEqual(get_prompt_dir(), test_dir.resolve())

    def test_render_prompt_basic(self):
        """Test basic prompt rendering without parameters."""
        # Create a simple prompt file
        prompt_file = self.prompt_dir / "test" / "simple_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Hello, world!")

        set_prompt_dir(self.prompt_dir)
        result = render_prompt("test/simple_v1")
        self.assertEqual(result, "Hello, world!")

    def test_render_prompt_with_params(self):
        """Test prompt rendering with Jinja2 parameters."""
        # Create a prompt file with parameters
        prompt_file = self.prompt_dir / "test" / "greeting_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Hello, {{ person_name }}! You are {{ person_age }} years old.")

        set_prompt_dir(self.prompt_dir)
        result = render_prompt("test/greeting_v1", person_name="Alice", person_age=30)
        self.assertEqual(result, "Hello, Alice! You are 30 years old.")

    def test_render_prompt_strips_whitespace(self):
        """Test that rendered prompts are stripped of leading/trailing whitespace."""
        prompt_file = self.prompt_dir / "test" / "whitespace_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("  \n  Hello, world!  \n  ")

        set_prompt_dir(self.prompt_dir)
        result = render_prompt("test/whitespace_v1")
        self.assertEqual(result, "Hello, world!")

    def test_render_prompt_with_extension(self):
        """Test that prompt names can include .j2 extension."""
        prompt_file = self.prompt_dir / "test" / "with_ext.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Test prompt")

        set_prompt_dir(self.prompt_dir)
        result = render_prompt("test/with_ext.j2")
        self.assertEqual(result, "Test prompt")

    def test_render_prompt_not_found(self):
        """Test that PromptNotFoundError is raised for missing prompts."""
        set_prompt_dir(self.prompt_dir)

        with self.assertRaises(PromptNotFoundError) as context:
            render_prompt("nonexistent/prompt_v1")

        self.assertIn("not found", str(context.exception).lower())
        self.assertIn("nonexistent/prompt_v1", str(context.exception))

    def test_get_prompt_source(self):
        """Test getting raw prompt source text."""
        prompt_content = "Hello, {{ name }}!"
        prompt_file = self.prompt_dir / "test" / "source_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text(prompt_content)

        set_prompt_dir(self.prompt_dir)
        source = get_prompt_source("test/source_v1")
        self.assertEqual(source, prompt_content)

    def test_get_prompt_source_not_found(self):
        """Test that PromptNotFoundError is raised for missing prompt source."""
        set_prompt_dir(self.prompt_dir)

        # get_prompt_source reads directly from disk, so it raises FileNotFoundError
        # which is a subclass of PromptNotFoundError
        with self.assertRaises((PromptNotFoundError, FileNotFoundError)):
            get_prompt_source("nonexistent/source_v1")

    def test_hash_text(self):
        """Test SHA256 hashing of text."""
        text1 = "Hello, world!"
        text2 = "Hello, world!"
        text3 = "Different text"

        hash1 = hash_text(text1)
        hash2 = hash_text(text2)
        hash3 = hash_text(text3)

        # Same text should produce same hash
        self.assertEqual(hash1, hash2)

        # Different text should produce different hash
        self.assertNotEqual(hash1, hash3)

        # Hash should be 64-character hex string
        self.assertEqual(len(hash1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in hash1))

    def test_get_prompt_metadata_basic(self):
        """Test getting prompt metadata without parameters."""
        prompt_content = "Test prompt"
        prompt_file = self.prompt_dir / "test" / "meta_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text(prompt_content)

        set_prompt_dir(self.prompt_dir)
        metadata = get_prompt_metadata("test/meta_v1")

        self.assertEqual(metadata["name"], "test/meta_v1")
        self.assertEqual(metadata["file"], "test/meta_v1.j2")
        self.assertIn("sha256", metadata)
        self.assertEqual(metadata["sha256"], hash_text(prompt_content))

    def test_get_prompt_metadata_with_params(self):
        """Test getting prompt metadata with parameters."""
        prompt_file = self.prompt_dir / "test" / "meta_params_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Test prompt")

        set_prompt_dir(self.prompt_dir)
        params = {"name": "Alice", "age": 30}
        metadata = get_prompt_metadata("test/meta_params_v1", params=params)

        self.assertEqual(metadata["name"], "test/meta_params_v1")
        self.assertIn("params", metadata)
        self.assertEqual(metadata["params"], params)

    def test_get_prompt_metadata_not_found(self):
        """Test that PromptNotFoundError is raised for missing prompt metadata."""
        set_prompt_dir(self.prompt_dir)

        # get_prompt_metadata calls get_prompt_source which raises FileNotFoundError
        # which is a subclass of PromptNotFoundError
        with self.assertRaises((PromptNotFoundError, FileNotFoundError)):
            get_prompt_metadata("nonexistent/meta_v1")

    def test_caching(self):
        """Test that prompts are cached after first load."""
        prompt_file = self.prompt_dir / "test" / "cache_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Cached prompt")

        set_prompt_dir(self.prompt_dir)

        # First render
        result1 = render_prompt("test/cache_v1")
        self.assertEqual(result1, "Cached prompt")

        # Modify file (cache should prevent seeing changes)
        prompt_file.write_text("Modified prompt")

        # Second render should use cache
        result2 = render_prompt("test/cache_v1")
        self.assertEqual(result2, "Cached prompt")  # Still old value

        # Clear cache
        clear_cache()

        # Third render should see new content
        result3 = render_prompt("test/cache_v1")
        self.assertEqual(result3, "Modified prompt")

    def test_clear_cache(self):
        """Test that clear_cache clears the template cache."""
        prompt_file = self.prompt_dir / "test" / "clear_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Original")

        set_prompt_dir(self.prompt_dir)

        # Render to populate cache
        render_prompt("test/clear_v1")

        # Modify file
        prompt_file.write_text("Updated")

        # Should still see cached version
        result1 = render_prompt("test/clear_v1")
        self.assertEqual(result1, "Original")

        # Clear cache
        clear_cache()

        # Should now see updated version
        result2 = render_prompt("test/clear_v1")
        self.assertEqual(result2, "Updated")

    def test_environment_variable_prompt_dir(self):
        """Test that PROMPT_DIR environment variable is respected."""
        prompt_file = self.prompt_dir / "test" / "env_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Environment prompt")

        # Set environment variable
        os.environ["PROMPT_DIR"] = str(self.prompt_dir)

        try:
            # Should use env var directory
            result = render_prompt("test/env_v1")
            self.assertEqual(result, "Environment prompt")
        finally:
            # Clean up
            os.environ.pop("PROMPT_DIR", None)
            clear_cache()

    def test_jinja2_template_features(self):
        """Test that Jinja2 template features work correctly."""
        prompt_file = self.prompt_dir / "test" / "jinja2_v1.j2"
        prompt_file.parent.mkdir()
        prompt_content = """Hello, {{ person_name }}!
{% if person_age >= 18 %}
You are an adult.
{% else %}
You are a minor.
{% endif %}
Items: {% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}"""
        prompt_file.write_text(prompt_content)

        set_prompt_dir(self.prompt_dir)
        result = render_prompt(
            "test/jinja2_v1",
            person_name="Bob",
            person_age=25,
            items=["apple", "banana", "cherry"],
        )

        self.assertIn("Hello, Bob!", result)
        self.assertIn("You are an adult.", result)
        self.assertIn("Items: apple, banana, cherry", result)

    def test_complex_prompt_rendering(self):
        """Test rendering complex prompts with multiple parameters."""
        prompt_file = self.prompt_dir / "summarization" / "complex_v1.j2"
        prompt_file.parent.mkdir()
        prompt_content = """Summarize the following podcast episode transcript.

{% if title %}
Episode Title: {{ title }}
{% endif %}

Transcript:
{{ transcript }}

Guidelines:
- Write a detailed summary with {{ paragraphs_min }}-{{ paragraphs_max }} paragraphs
- Focus on key decisions, arguments, and lessons learned
- Ignore sponsorships, ads, and housekeeping
- Do not use quotes or speaker names
- Do not invent information not implied by the transcript"""
        prompt_file.write_text(prompt_content)

        set_prompt_dir(self.prompt_dir)
        result = render_prompt(
            "summarization/complex_v1",
            title="Test Episode",
            transcript="This is a test transcript.",
            paragraphs_min=3,
            paragraphs_max=6,
        )

        self.assertIn("Episode Title: Test Episode", result)
        self.assertIn("This is a test transcript.", result)
        self.assertIn("3-6 paragraphs", result)

    def test_prompt_metadata_consistency(self):
        """Test that metadata is consistent across calls."""
        prompt_file = self.prompt_dir / "test" / "consistency_v1.j2"
        prompt_file.parent.mkdir()
        prompt_file.write_text("Consistent prompt")

        set_prompt_dir(self.prompt_dir)

        metadata1 = get_prompt_metadata("test/consistency_v1")
        metadata2 = get_prompt_metadata("test/consistency_v1")

        # Metadata should be identical
        self.assertEqual(metadata1["name"], metadata2["name"])
        self.assertEqual(metadata1["file"], metadata2["file"])
        self.assertEqual(metadata1["sha256"], metadata2["sha256"])


if __name__ == "__main__":
    unittest.main()
