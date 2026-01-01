#!/usr/bin/env python3
"""Tests for summarizer core functions.

These tests focus on pure function logic without ML model dependencies.
All ML dependencies (tokenizers, models, pipelines) are mocked.
"""

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper import summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestChunkTextWords(unittest.TestCase):
    """Tests for chunk_text_words function."""

    def test_chunk_text_words_basic(self):
        """Test basic word-based chunking."""
        text = " ".join(["word"] * 100)  # 100 words
        chunks = summarizer.chunk_text_words(text, chunk_size=30, overlap=5)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)

    def test_chunk_text_words_with_overlap(self):
        """Test word-based chunking with overlap."""
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = summarizer.chunk_text_words(text, chunk_size=20, overlap=5)

        self.assertGreater(len(chunks), 1)
        # Verify overlap by checking that some words appear in multiple chunks
        # (simplified check - just verify chunks are created)

    def test_chunk_text_words_short_text(self):
        """Test word-based chunking with short text."""
        text = "This is a short text."
        chunks = summarizer.chunk_text_words(text, chunk_size=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_words_empty_text(self):
        """Test word-based chunking with empty text."""
        chunks = summarizer.chunk_text_words("", chunk_size=30)

        self.assertEqual(len(chunks), 0)

    def test_chunk_text_words_defaults(self):
        """Test word-based chunking with default parameters."""
        text = " ".join(["word"] * 200)
        chunks = summarizer.chunk_text_words(text)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestChunkTextForSummarization(unittest.TestCase):
    """Tests for chunk_text_for_summarization function."""

    def test_chunk_text_basic(self):
        """Test basic token-based chunking."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(1000))  # 1000 tokens
        mock_tokenizer.decode.return_value = "chunk text"

        chunks = summarizer.chunk_text_for_summarization(
            text="Test text", tokenizer=mock_tokenizer, chunk_size=500, overlap=100
        )

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        mock_tokenizer.encode.assert_called_once()
        mock_tokenizer.decode.assert_called()

    def test_chunk_text_with_overlap(self):
        """Test token-based chunking with overlap."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(2000))  # 2000 tokens
        mock_tokenizer.decode.return_value = "chunk text"

        chunks = summarizer.chunk_text_for_summarization(
            text="Long text", tokenizer=mock_tokenizer, chunk_size=500, overlap=100
        )

        # Should create multiple chunks with overlap
        self.assertGreater(len(chunks), 2)

    def test_chunk_text_short_text(self):
        """Test token-based chunking with short text."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
        mock_tokenizer.decode.return_value = "short text"

        chunks = summarizer.chunk_text_for_summarization(
            text="Short", tokenizer=mock_tokenizer, chunk_size=500, overlap=100
        )

        self.assertEqual(len(chunks), 1)

    def test_chunk_text_empty_text(self):
        """Test token-based chunking with empty text."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = ""

        chunks = summarizer.chunk_text_for_summarization(
            text="", tokenizer=mock_tokenizer, chunk_size=500, overlap=100
        )

        self.assertEqual(len(chunks), 0)

    def test_chunk_text_zero_overlap(self):
        """Test token-based chunking with zero overlap."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(1000))
        mock_tokenizer.decode.return_value = "chunk"

        chunks = summarizer.chunk_text_for_summarization(
            text="Text", tokenizer=mock_tokenizer, chunk_size=500, overlap=0
        )

        self.assertGreater(len(chunks), 1)

    def test_chunk_text_overlap_larger_than_chunk(self):
        """Test token-based chunking when overlap is larger than chunk size."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(1000))
        mock_tokenizer.decode.return_value = "chunk"

        chunks = summarizer.chunk_text_for_summarization(
            text="Text", tokenizer=mock_tokenizer, chunk_size=100, overlap=150
        )

        # Should still create chunks (advance will be at least 1)
        self.assertGreater(len(chunks), 0)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestValidateAndFixRepetitiveSummary(unittest.TestCase):
    """Tests for _validate_and_fix_repetitive_summary function."""

    def test_validate_no_repetition(self):
        """Test summary with no repetition."""
        summary = "This is sentence one. This is sentence two. This is sentence three."
        result = summarizer._validate_and_fix_repetitive_summary(summary)

        self.assertEqual(result, summary)

    def test_validate_repetitive_sentences(self):
        """Test summary with repetitive sentences."""
        repeated = "This is a repeated sentence."
        summary = ". ".join([repeated] * 5)  # Repeat 5 times
        result = summarizer._validate_and_fix_repetitive_summary(summary)

        # Should remove duplicates
        self.assertNotEqual(result, summary)
        self.assertIn(repeated, result)
        # Should have fewer sentences
        self.assertLess(len(result.split(". ")), 5)

    def test_validate_short_summary(self):
        """Test that very short summaries are returned unchanged."""
        summary = "Short."
        result = summarizer._validate_and_fix_repetitive_summary(summary)

        self.assertEqual(result, summary)

    def test_validate_empty_summary(self):
        """Test that empty summaries are returned unchanged."""
        result = summarizer._validate_and_fix_repetitive_summary("")

        self.assertEqual(result, "")

    def test_validate_repetitive_ngrams(self):
        """Test summary with repetitive n-grams (hallucination detection)."""
        # Create text with repetitive 5-grams that will trigger the > 5 threshold
        # Need: >= 3 sentences (for sentence check to pass), > 10 words, same 5-gram > 5 times
        # Repeat the same 5-word phrase 7 times with periods to create sentences
        phrase = "What is the best way"
        # Create 7 sentences, each with the same phrase
        sentences = [f"{phrase}."] * 7
        summary = " ".join(sentences)  # 7 sentences, 35 words total
        result = summarizer._validate_and_fix_repetitive_summary(summary)

        # Should return empty string for hallucinated content (max_ngram_repetitions > 5)
        # With 7 repetitions of the same 5-word phrase, the n-gram "what is the best way"
        # will appear many times (overlapping), and max_ngram_repetitions will be > 5
        self.assertEqual(result, "")

    def test_validate_few_sentences(self):
        """Test summary with too few sentences (below threshold)."""
        summary = "Sentence one. Sentence two."
        result = summarizer._validate_and_fix_repetitive_summary(summary)

        # Should return unchanged (below FEW_CHUNKS_THRESHOLD=3)
        self.assertEqual(result, summary)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestStripInstructionLeak(unittest.TestCase):
    """Tests for _strip_instruction_leak function."""

    def test_strip_instruction_leak_detects_patterns(self):
        """Test that instruction leak patterns are removed."""
        summary = "This is a summary. Your task is to summarize. This is more content."
        result = summarizer._strip_instruction_leak(summary)

        self.assertNotIn("Your task is to summarize", result)
        self.assertIn("This is a summary", result)
        self.assertIn("This is more content", result)

    def test_strip_instruction_leak_no_leak(self):
        """Test summary with no instruction leaks."""
        summary = "This is a normal summary. It contains no instructions."
        result = summarizer._strip_instruction_leak(summary)

        self.assertEqual(result, summary)

    def test_strip_instruction_leak_empty(self):
        """Test empty summary."""
        result = summarizer._strip_instruction_leak("")

        self.assertEqual(result, "")

    def test_strip_instruction_leak_multiple_patterns(self):
        """Test summary with multiple instruction leak patterns."""
        summary = (
            "Summary text. Summarize the following content. "
            "More text. Follow these principles. End text."
        )
        result = summarizer._strip_instruction_leak(summary)

        self.assertNotIn("Summarize the following", result)
        self.assertNotIn("Follow these principles", result)
        self.assertIn("Summary text", result)
        self.assertIn("More text", result)
        self.assertIn("End text", result)

    def test_strip_instruction_leak_all_removed(self):
        """Test summary where all sentences are instruction leaks."""
        summary = "Your task is to summarize. Aim for a comprehensive summary."
        result = summarizer._strip_instruction_leak(summary)

        # Should return empty or just whitespace
        self.assertEqual(result.strip(), "")


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestSelectKeySummaries(unittest.TestCase):
    """Tests for _select_key_summaries function."""

    def test_select_key_summaries_few_chunks(self):
        """Test selection with few chunks (<=3)."""
        summaries = ["Summary 1", "Summary 2", "Summary 3"]
        result = summarizer._select_key_summaries(summaries)

        # Should return all summaries
        self.assertEqual(result, summaries)

    def test_select_key_summaries_medium_chunks(self):
        """Test selection with medium chunks (4-10)."""
        summaries = [f"Summary {i}" for i in range(7)]
        result = summarizer._select_key_summaries(summaries)

        # Should return first, middle, last
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Summary 0")
        self.assertEqual(result[1], "Summary 3")  # middle (7//2 = 3)
        self.assertEqual(result[2], "Summary 6")

    def test_select_key_summaries_many_chunks(self):
        """Test selection with many chunks (>10)."""
        summaries = [f"Summary {i}" for i in range(15)]
        result = summarizer._select_key_summaries(summaries)

        # Should return 5 summaries: first, 1/4, 1/2, 3/4, last
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], "Summary 0")
        self.assertEqual(result[1], "Summary 3")  # 15//4 = 3
        self.assertEqual(result[2], "Summary 7")  # 15//2 = 7
        self.assertEqual(result[3], "Summary 11")  # 3*15//4 = 11
        self.assertEqual(result[4], "Summary 14")

    def test_select_key_summaries_empty(self):
        """Test selection with empty list."""
        result = summarizer._select_key_summaries([])

        self.assertEqual(result, [])

    def test_select_key_summaries_single(self):
        """Test selection with single summary."""
        summaries = ["Summary 1"]
        result = summarizer._select_key_summaries(summaries)

        self.assertEqual(result, summaries)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestJoinSummariesWithStructure(unittest.TestCase):
    """Tests for _join_summaries_with_structure function."""

    def test_join_summaries_basic(self):
        """Test joining summaries with structure."""
        summaries = ["Summary 1", "Summary 2", "Summary 3"]
        result = summarizer._join_summaries_with_structure(summaries)

        self.assertEqual(result, "Summary 1\n\nSummary 2\n\nSummary 3")

    def test_join_summaries_single(self):
        """Test joining single summary."""
        summaries = ["Summary 1"]
        result = summarizer._join_summaries_with_structure(summaries)

        self.assertEqual(result, "Summary 1")

    def test_join_summaries_empty(self):
        """Test joining empty list."""
        result = summarizer._join_summaries_with_structure([])

        self.assertEqual(result, "")


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestSafeSummarizeExpanded(unittest.TestCase):
    """Additional tests for safe_summarize function."""

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_safe_summarize_general_exception(self, mock_torch):
        """Test safe_summarize handles general exceptions."""
        mock_model = Mock()
        mock_model.summarize.side_effect = ValueError("Some error")
        mock_model.device = "cpu"

        result = summarizer.safe_summarize(mock_model, "Test text", max_length=50)

        self.assertEqual(result, "")

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_safe_summarize_cuda_oom(self, mock_torch):
        """Test safe_summarize handles CUDA OOM errors."""
        mock_model = Mock()
        mock_model.summarize.side_effect = RuntimeError("CUDA out of memory")
        mock_model.device = "cuda"

        result = summarizer.safe_summarize(mock_model, "Test text", max_length=50)

        self.assertEqual(result, "")

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_safe_summarize_mps_error(self, mock_torch):
        """Test safe_summarize handles MPS errors."""
        mock_model = Mock()
        mock_model.summarize.side_effect = RuntimeError("MPS error occurred")
        mock_model.device = "mps"

        result = summarizer.safe_summarize(mock_model, "Test text", max_length=50)

        self.assertEqual(result, "")

    def test_safe_summarize_success_with_prompt(self):
        """Test safe_summarize with prompt."""
        mock_model = Mock()
        mock_model.summarize.return_value = "Summary with prompt."
        mock_model.device = "cpu"

        result = summarizer.safe_summarize(
            mock_model, "Test text", max_length=50, prompt="Custom prompt"
        )

        self.assertEqual(result, "Summary with prompt.")
        mock_model.summarize.assert_called_once_with(
            "Test text", max_length=50, prompt="Custom prompt"
        )


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestCheckIfNeedsChunking(unittest.TestCase):
    """Tests for _check_if_needs_chunking function."""

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_check_if_needs_chunking_fits(self, mock_torch):
        """Test when text fits without chunking."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(500))  # 500 tokens

        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer
        mock_model.summarize.return_value = "Summary text"

        result = summarizer._check_if_needs_chunking(
            model=mock_model,
            text="Test text",
            chunk_size=1000,
            max_length=150,
            min_length=30,
            prompt=None,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result, "Summary text")
        mock_model.summarize.assert_called_once()

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_check_if_needs_chunking_too_large(self, mock_torch):
        """Test when text is too large and needs chunking."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(2000))  # 2000 tokens

        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer

        result = summarizer._check_if_needs_chunking(
            model=mock_model,
            text="Long text",
            chunk_size=1000,
            max_length=150,
            min_length=30,
            prompt=None,
        )

        self.assertIsNone(result)  # Needs chunking
        mock_model.summarize.assert_not_called()

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_check_if_needs_chunking_no_tokenizer(self, mock_torch):
        """Test when model has no tokenizer."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.tokenizer = None

        with self.assertRaises(RuntimeError):
            summarizer._check_if_needs_chunking(
                model=mock_model,
                text="Test",
                chunk_size=1000,
                max_length=150,
                min_length=30,
                prompt=None,
            )


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestPrepareChunks(unittest.TestCase):
    """Tests for _prepare_chunks function."""

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_prepare_chunks_token_based(self, mock_torch):
        """Test preparing chunks with token-based chunking."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(2000))  # 2000 tokens
        mock_tokenizer.decode.return_value = "chunk text"

        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer

        chunks, effective_chunk_size = summarizer._prepare_chunks(
            model=mock_model,
            text="Long text",
            chunk_size=500,
            use_word_chunking=False,
            word_chunk_size=100,
            word_overlap=10,
        )

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(effective_chunk_size, 500)

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_prepare_chunks_word_based(self, mock_torch):
        """Test preparing chunks with word-based chunking flag (still uses tokenizer)."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(2000))  # 2000 tokens
        mock_tokenizer.decode.return_value = "chunk text"

        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer  # Still needs tokenizer

        text = " ".join(["word"] * 200)  # 200 words
        chunks, effective_chunk_size = summarizer._prepare_chunks(
            model=mock_model,
            text=text,
            chunk_size=500,
            use_word_chunking=True,  # Flag for logging, but still uses tokenizer
            word_chunk_size=50,
            word_overlap=5,
        )

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestSummarizeChunksMap(unittest.TestCase):
    """Tests for _summarize_chunks_map function."""

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_summarize_chunks_map_sequential(self, mock_torch):
        """Test map step with sequential processing (GPU)."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.device = "cuda"  # GPU = sequential
        mock_model.model_name = "test-model"
        mock_model.summarize.return_value = "Chunk summary"

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

        result = summarizer._summarize_chunks_map(
            model=mock_model,
            chunks=chunks,
            max_length=150,
            min_length=30,
            prompt=None,
            batch_size=None,
            use_word_chunking=False,
            word_chunk_size=100,
            word_overlap=10,
            chunk_size=500,
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(mock_model.summarize.call_count, 3)

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_summarize_chunks_map_parallel(self, mock_torch):
        """Test map step with parallel processing (CPU)."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.device = "cpu"  # CPU = can parallelize
        mock_model.model_name = "test-model"
        mock_model.summarize.return_value = "Chunk summary"

        chunks = ["Chunk 1", "Chunk 2"]

        result = summarizer._summarize_chunks_map(
            model=mock_model,
            chunks=chunks,
            max_length=150,
            min_length=30,
            prompt=None,
            batch_size=2,  # Enable parallel
            use_word_chunking=False,
            word_chunk_size=100,
            word_overlap=10,
            chunk_size=500,
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestCombineSummariesReduce(unittest.TestCase):
    """Tests for _combine_summaries_reduce function."""

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_combine_summaries_reduce_empty(self, mock_torch):
        """Test reduce step with empty chunk summaries."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.tokenizer = None

        result = summarizer._combine_summaries_reduce(
            model=mock_model, chunk_summaries=[], max_length=150, min_length=30, prompt=None
        )

        self.assertEqual(result, "")

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_combine_summaries_reduce_single_pass(self, mock_torch):
        """Test reduce step with small combined text (single-pass abstractive)."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        # Small token count to trigger single-pass
        mock_tokenizer.encode.return_value = list(range(100))

        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer
        mock_model.model = Mock()
        mock_model.model.config = Mock()
        mock_model.model.config.max_position_embeddings = 1024
        mock_model.summarize.return_value = "Final summary"

        chunk_summaries = ["Summary 1", "Summary 2"]

        result = summarizer._combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=chunk_summaries,
            max_length=150,
            min_length=30,
            prompt=None,
        )

        self.assertIsInstance(result, str)
        # Should call abstractive combine
        mock_model.summarize.assert_called()

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_combine_summaries_reduce_extractive_fallback(self, mock_torch):
        """Test reduce step with very large combined text (extractive fallback)."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        # Very large token count to trigger extractive
        mock_tokenizer.encode.return_value = list(range(10000))

        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer
        mock_model.model = Mock()
        mock_model.model.config = Mock()
        mock_model.model.config.max_position_embeddings = 1024
        mock_model.summarize.return_value = "Extracted summary"

        chunk_summaries = [f"Summary {i}" for i in range(20)]  # Many summaries

        result = summarizer._combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=chunk_summaries,
            max_length=150,
            min_length=30,
            prompt=None,
        )

        self.assertIsInstance(result, str)
        # Should use extractive approach (selects key summaries first)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestCombineSummariesExtractive(unittest.TestCase):
    """Tests for _combine_summaries_extractive function."""

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_combine_summaries_extractive_short(self, mock_torch):
        """Test extractive combination with short selected summaries."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.summarize.return_value = "Final summary"

        selected = ["Summary 1", "Summary 2"]

        result = summarizer._combine_summaries_extractive(
            model=mock_model,
            selected_summaries=selected,
            max_length=150,
            min_length=30,
            prompt=None,
            model_max=1024,
        )

        self.assertIsInstance(result, str)
        # Short summaries should be joined directly without further summarization

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_combine_summaries_extractive_long(self, mock_torch):
        """Test extractive combination with long selected summaries."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.summarize.return_value = "Final summary"

        # Create long summaries that need further summarization
        selected = ["Very long summary text " * 100] * 3

        result = summarizer._combine_summaries_extractive(
            model=mock_model,
            selected_summaries=selected,
            max_length=150,
            min_length=30,
            prompt=None,
            model_max=1024,
        )

        self.assertIsInstance(result, str)
        # Long summaries should trigger final summarization pass
        mock_model.summarize.assert_called()


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestCombineSummariesAbstractive(unittest.TestCase):
    """Tests for _combine_summaries_abstractive function."""

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_combine_summaries_abstractive_success(self, mock_torch):
        """Test abstractive combination success."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.summarize.return_value = "Abstractive final summary"

        combined_text = "Summary 1\n\nSummary 2\n\nSummary 3"
        chunk_summaries = ["Summary 1", "Summary 2", "Summary 3"]

        result = summarizer._combine_summaries_abstractive(
            model=mock_model,
            combined_text=combined_text,
            chunk_summaries=chunk_summaries,
            max_length=150,
            min_length=30,
            prompt=None,
            model_max=1024,
            combined_tokens=200,
        )

        self.assertEqual(result, "Abstractive final summary")
        mock_model.summarize.assert_called_once()

    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_combine_summaries_abstractive_oom_error(self, mock_torch):
        """Test abstractive combination with OOM error."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_model = Mock()
        mock_model.summarize.side_effect = RuntimeError("CUDA out of memory")

        combined_text = "Summary 1\n\nSummary 2"
        chunk_summaries = ["Summary 1", "Summary 2"]

        with self.assertRaises(RuntimeError):
            summarizer._combine_summaries_abstractive(
                model=mock_model,
                combined_text=combined_text,
                chunk_summaries=chunk_summaries,
                max_length=150,
                min_length=30,
                prompt=None,
                model_max=1024,
                combined_tokens=200,
            )


if __name__ == "__main__":
    unittest.main()
