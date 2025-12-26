# Test Categorization for RFC-018

This document maps all 32 test files to their new locations and categories.

## Unit Tests → `tests/unit/podcast_scraper/`

**Criteria:** Test single module/function, fully mocked, fast, no network

1. `test_config_validation.py` → `test_config.py`
   - Tests config validation logic
   - Fully mocked, no external dependencies
   - **Network:** No

2. `test_downloader.py` → `test_downloader.py`
   - Tests HTTP downloader with mocked requests
   - Uses mocks for HTTP calls
   - **Network:** No (mocked)

3. `test_rss_parser.py` → `test_rss_parser.py`
   - Tests RSS parsing logic
   - Uses mocked XML data
   - **Network:** No

4. `test_filesystem.py` → `test_filesystem.py`
   - Tests filesystem operations
   - Uses temp directories
   - **Network:** No

5. `test_metadata.py` → `test_metadata.py`
   - Tests metadata generation
   - Unit tests for metadata models
   - **Network:** No

6. `test_prompt_store.py` → `test_prompt_store.py`
   - Tests prompt loading and caching
   - File-based, no network
   - **Network:** No

7. `test_speaker_detection.py` → `test_speaker_detection.py`
   - Tests speaker detection logic
   - Uses mocked spaCy models
   - **Network:** No

8. `test_api_versioning.py` → `test_api_versioning.py` (top-level in unit/)
   - Tests API version attributes
   - No external dependencies
   - **Network:** No

9. `test_package_imports.py` → `test_package_imports.py` (top-level in unit/)
   - Tests package import structure
   - No external dependencies
   - **Network:** No

10. `test_utilities.py` → `test_utilities.py` (top-level in unit/)
    - Tests utility functions
    - No external dependencies
    - **Network:** No

**Provider Unit Tests → Subdirectories:**

11. `test_speaker_detector_provider.py` → `speaker_detectors/test_factory.py` + `speaker_detectors/test_ner_detector.py` + `speaker_detectors/test_openai_detector.py`
    - Tests speaker detector providers
    - Fully mocked
    - **Network:** No

12. `test_summarization_provider.py` → `summarization/test_factory.py` + `summarization/test_local_provider.py` + `summarization/test_openai_provider.py`
    - Tests summarization provider factory
    - Fully mocked
    - **Network:** No

13. `test_transcription_provider.py` → `transcription/test_factory.py` + `transcription/test_whisper_provider.py` + `transcription/test_openai_provider.py`
    - Tests transcription provider factory
    - Fully mocked
    - **Network:** No

14. `test_openai_providers.py` → Split across provider subdirectories
    - Tests OpenAI provider implementations
    - Fully mocked (no real API calls)
    - **Network:** No

## Integration Tests → `tests/integration/`

**Criteria:** Test multiple components together, component interactions

15. `test_provider_integration.py` → `test_provider_integration.py`
    - Tests provider system integration
    - Multiple providers working together
    - **Network:** No

16. `test_protocol_compliance.py` → `test_protocol_compliance.py`
    - Tests protocol implementation across providers
    - Component interactions
    - **Network:** No

17. `test_protocol_compliance_extended.py` → `test_protocol_compliance_extended.py`
    - Extended protocol compliance tests
    - Component interactions
    - **Network:** No

18. `test_fallback_behavior.py` → `test_fallback_behavior.py`
    - Tests fallback chains between providers
    - Multiple components
    - **Network:** No

19. `test_parallel_summarization.py` → `test_parallel_summarization.py`
    - Tests parallel processing across components
    - Component interactions
    - **Network:** No

20. `test_provider_error_handling_extended.py` → `test_provider_error_handling_extended.py`
    - Tests error handling across providers
    - Component interactions
    - **Network:** No

21. `test_stage0_foundation.py` → `test_stage0_foundation.py`
    - Tests foundation stage (provider system)
    - Component integration
    - **Network:** No

## Workflow E2E Tests → `tests/workflow_e2e/`

**Criteria:** Test complete workflows, CLI, full pipelines

22. `test_cli.py` → `test_cli.py`
    - Tests CLI interface end-to-end
    - Full workflow tests
    - **Network:** Possibly (needs review)

23. `test_service.py` → `test_service.py`
    - Tests service mode end-to-end
    - Full workflow tests
    - **Network:** Possibly (needs review)

24. `test_integration.py` → `test_workflow_e2e.py` (RENAME)
    - Full workflow integration tests
    - End-to-end pipeline
    - **Network:** Yes (uses requests, but may be mocked)

25. `test_podcast_scraper.py` → `test_podcast_scraper.py`
    - Tests main pipeline end-to-end
    - Full workflow
    - **Network:** Possibly

26. `test_summarizer.py` → `test_summarizer.py`
    - Tests full summarization workflow
    - End-to-end
    - **Network:** No (uses local models)

27. `test_summarizer_edge_cases.py` → `test_summarizer_edge_cases.py`
    - Tests edge cases in summarization workflow
    - End-to-end
    - **Network:** No

28. `test_summarizer_security.py` → `test_summarizer_security.py`
    - Tests security aspects of summarization
    - End-to-end
    - **Network:** No

29. `test_eval_scripts.py` → `test_eval_scripts.py`
    - Tests evaluation scripts
    - End-to-end evaluation workflows
    - **Network:** No

30. `test_env_variables.py` → `test_env_variables.py`
    - Tests environment variable handling
    - End-to-end configuration
    - **Network:** No

## Notes

- Files that mention "network" may still use mocks - need to verify
- Provider tests may need to be split if they test multiple things
- Some files may have both unit and integration tests - will need to split

## Network Test Identification

**Verified: All tests use mocks (no real network calls)**

- `test_integration.py` - Uses `_mock_http_map` and mocked responses ✅
- `test_downloader.py` - Uses `MockHTTPResponse` and `create_rss_response` ✅
- `test_cli.py` - Uses mocked HTTP responses ✅
- `test_service.py` - Uses mocked HTTP responses ✅
- `test_podcast_scraper.py` - Uses mocked HTTP responses ✅

**Conclusion:** No tests currently hit the network. All use mocks. However, we should mark workflow_e2e tests that *could* hit network if mocks were removed, for future safety.

## Final Categorization Summary

- **Unit Tests:** 14 files (including provider tests that will be split)
- **Integration Tests:** 7 files
- **Workflow E2E Tests:** 9 files
- **Total:** 30 test files (excluding `conftest.py` and `__init__.py`)
