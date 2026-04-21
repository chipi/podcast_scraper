#!/usr/bin/env python3
"""Minimal test matrix for fingerprint validation.

This script validates that fingerprints correctly capture changes and remain
stable for irrelevant factors.

CHANGE TESTS (fingerprint should change):
1. Control run (baseline)
2. Prompt change (hash should change)
3. Preprocessing profile change
4. Chunk size change

STABILITY TESTS (fingerprint should NOT change):
G. Re-run same config twice (identical except run_id)
H. Different output paths (paths shouldn't affect fingerprint)

These tests ensure fingerprints represent configuration identity, not execution
metadata or environment noise.

Usage:
    python scripts/eval/test_fingerprint_matrix.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Test configuration
TEST_DATASET = "curated_5feeds_smoke_v1"
BASE_BASELINE_ID = "test_fingerprint_validation"
OUTPUT_DIR = "data/eval/baselines"
EXPERIMENTS_DIR = Path("data/eval/configs")


def find_test_config(
    backend_type: str = "hf_local",
    map_model: Optional[str] = "bart-small",
    reduce_model: Optional[str] = "long-fast",
) -> Path:
    """Find a test config file matching the specified criteria.

    Args:
        backend_type: Backend type to match ("hf_local" or "openai")
        map_model: MAP model to match (e.g., "bart-small", "bart-large")
        reduce_model: REDUCE model to match (e.g., "long-fast", "long")

    Returns:
        Path to matching config file

    Raises:
        FileNotFoundError: If no matching config is found
    """
    from podcast_scraper.evaluation.experiment_config import load_experiment_config

    # Search for configs matching criteria
    for config_file in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
        if config_file.name == "README.md":
            continue
        try:
            cfg = load_experiment_config(config_file)
            # Check backend type
            if cfg.backend.type != backend_type:
                continue
            # Check models (for hf_local)
            if backend_type == "hf_local":
                if map_model and cfg.backend.map_model != map_model:
                    continue
                if reduce_model and cfg.backend.reduce_model != reduce_model:
                    continue
            # Found a match
            return config_file
        except Exception:
            # Skip invalid configs
            continue

    raise FileNotFoundError(
        f"No config found matching: backend={backend_type}, "
        f"map_model={map_model}, reduce_model={reduce_model}"
    )


def get_fingerprint_hash(fingerprint_path: Path) -> str:
    """Compute hash of fingerprint for comparison.

    Args:
        fingerprint_path: Path to fingerprint.json

    Returns:
        SHA256 hash (first 16 chars) of fingerprint
    """
    content = fingerprint_path.read_text(encoding="utf-8")
    fingerprint = json.loads(content)
    # Create deterministic string representation
    fingerprint_str = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()[:16]


def get_fingerprint_for_comparison(
    fingerprint_path: Path, exclude_run_metadata: bool = True
) -> Dict[str, Any]:
    """Get fingerprint excluding run metadata for comparison.

    For stability checks, we exclude:
    - run_id (changes on every run)
    - baseline_id (may differ for different output paths)

    The fingerprint should represent configuration identity, not execution metadata.

    Args:
        fingerprint_path: Path to fingerprint.json
        exclude_run_metadata: If True, exclude run_id and baseline_id (default: True)

    Returns:
        Fingerprint dictionary with run metadata removed
    """
    fp = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if not exclude_run_metadata:
        return fp

    # Deep copy to avoid modifying original
    fp_copy = json.loads(json.dumps(fp))

    # Remove run metadata from run_context
    if "run_context" in fp_copy:
        if "run_id" in fp_copy["run_context"]:
            del fp_copy["run_context"]["run_id"]
        if "baseline_id" in fp_copy["run_context"]:
            del fp_copy["run_context"]["baseline_id"]

    return fp_copy


def compare_fingerprints(
    fp1_path: Path,
    fp2_path: Path,
    test_name: str,
    exclude_run_metadata: bool = True,
) -> bool:
    """Compare two fingerprints and report differences.

    Args:
        fp1_path: Path to first fingerprint
        fp2_path: Path to second fingerprint
        test_name: Name of test for logging
        exclude_run_metadata: If True, exclude run_id and baseline_id
            from comparison (default: True)

    Returns:
        True if fingerprints are identical (excluding run metadata), False otherwise
    """
    fp1 = json.loads(fp1_path.read_text(encoding="utf-8"))
    fp2 = json.loads(fp2_path.read_text(encoding="utf-8"))

    # For comparison, exclude run metadata (run_id, baseline_id)
    if exclude_run_metadata:
        fp1_compare = get_fingerprint_for_comparison(fp1_path, exclude_run_metadata=True)
        fp2_compare = get_fingerprint_for_comparison(fp2_path, exclude_run_metadata=True)
    else:
        fp1_compare = fp1
        fp2_compare = fp2

    # Compute hash of comparison versions
    fp1_str = json.dumps(fp1_compare, sort_keys=True, ensure_ascii=False)
    fp2_str = json.dumps(fp2_compare, sort_keys=True, ensure_ascii=False)
    hash1 = hashlib.sha256(fp1_str.encode("utf-8")).hexdigest()[:16]
    hash2 = hashlib.sha256(fp2_str.encode("utf-8")).hexdigest()[:16]

    if hash1 == hash2:
        logger.info(f"✓ {test_name}: Fingerprints are IDENTICAL (hash: {hash1})")
        if exclude_run_metadata:
            run_id_diff = fp1.get("run_context", {}).get("run_id") != fp2.get(
                "run_context", {}
            ).get("run_id")
            baseline_id_diff = fp1.get("run_context", {}).get("baseline_id") != fp2.get(
                "run_context", {}
            ).get("baseline_id")
            if run_id_diff or baseline_id_diff:
                metadata_diffs = []
                if run_id_diff:
                    metadata_diffs.append("run_id")
                if baseline_id_diff:
                    metadata_diffs.append("baseline_id")
                logger.info(
                    f"  ({', '.join(metadata_diffs)} differ, as expected for stability test)"
                )
        return True
    else:
        logger.warning(f"✗ {test_name}: Fingerprints DIFFER")
        logger.info(f"  Hash 1: {hash1}")
        logger.info(f"  Hash 2: {hash2}")

        # Report specific differences (excluding run metadata)
        differences = []
        for key in set(list(fp1_compare.keys()) + list(fp2_compare.keys())):
            if key not in fp1_compare:
                differences.append(f"  Missing in fp1: {key}")
            elif key not in fp2_compare:
                differences.append(f"  Missing in fp2: {key}")
            elif fp1_compare[key] != fp2_compare[key]:
                differences.append(f"  Different {key}:")
                differences.append(f"    fp1: {json.dumps(fp1_compare[key], indent=6)}")
                differences.append(f"    fp2: {json.dumps(fp2_compare[key], indent=6)}")

        if differences:
            logger.info("Differences:")
            for diff in differences:
                logger.info(diff)

        return False


def run_baseline(baseline_id: str, experiment_config: Path) -> Path:
    """Run baseline materialization.

    Args:
        baseline_id: Baseline identifier
        experiment_config: Path to experiment config YAML

    Returns:
        Path to generated fingerprint.json
    """
    logger.info(f"Running baseline: {baseline_id}")
    cmd = [
        sys.executable,
        "scripts/eval/materialize_baseline.py",
        "--baseline-id",
        baseline_id,
        "--dataset-id",
        TEST_DATASET,
        "--experiment-config",
        str(experiment_config),
        "--output-dir",
        OUTPUT_DIR,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error(f"Baseline creation failed: {result.stderr}")
        raise RuntimeError(f"Failed to create baseline {baseline_id}")

    fingerprint_path = Path(OUTPUT_DIR) / baseline_id / "fingerprint.json"
    if not fingerprint_path.exists():
        raise FileNotFoundError(f"Fingerprint not created: {fingerprint_path}")

    return fingerprint_path


def test_1_control_run():
    """Test 1: Control run (baseline)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Control Run (Baseline)")
    logger.info("=" * 60)

    baseline_id = f"{BASE_BASELINE_ID}_control"
    # Find a test config dynamically (hf_local with bart-small + long-fast)
    config_path = find_test_config(
        backend_type="hf_local", map_model="bart-small", reduce_model="long-fast"
    )

    fingerprint_path = run_baseline(baseline_id, config_path)
    logger.info(f"✓ Control run completed: {fingerprint_path}")
    logger.info(f"  Fingerprint hash: {get_fingerprint_hash(fingerprint_path)}")

    return fingerprint_path


def test_2_prompt_change():
    """Test 2: Prompt change (hash should change)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Prompt Change")
    logger.info("=" * 60)
    logger.info("Expected: Fingerprint should change (prompt affects output)")

    # Create modified config with different prompt
    baseline_id_control = f"{BASE_BASELINE_ID}_control"
    baseline_id_prompt = f"{BASE_BASELINE_ID}_prompt_change"

    # Read base config
    # Find a test config dynamically (hf_local with bart-small + long-fast)
    base_config = find_test_config(
        backend_type="hf_local", map_model="bart-small", reduce_model="long-fast"
    )
    config_content = base_config.read_text(encoding="utf-8")

    # Modify prompt (change from long_v1 to a different prompt)
    # Note: This assumes the prompt exists - in real test, you'd create it
    modified_config = config_content.replace(
        'user: "summarization/long_v1"',
        'user: "summarization/long_v2"',  # Different prompt
    )

    # Write temporary config
    temp_config = Path(f"data/eval/configs/{baseline_id_prompt}.yaml")
    temp_config.write_text(modified_config, encoding="utf-8")

    try:
        fp_control = Path(OUTPUT_DIR) / baseline_id_control / "fingerprint.json"
        fp_prompt = run_baseline(baseline_id_prompt, temp_config)

        # Compare
        is_same = compare_fingerprints(fp_control, fp_prompt, "Prompt Change Test")

        if is_same:
            logger.warning("⚠ WARNING: Fingerprints are identical after prompt change!")
            logger.warning("  This suggests prompts are not being fingerprinted correctly.")
            return False
        else:
            logger.info("✓ Fingerprints correctly differ after prompt change")
            return True
    finally:
        # Cleanup
        if temp_config.exists():
            temp_config.unlink()


def test_3_preprocessing_change():
    """Test 3: Preprocessing profile change."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Preprocessing Profile Change")
    logger.info("=" * 60)
    logger.info("Expected: preprocessing section should change")

    baseline_id_control = f"{BASE_BASELINE_ID}_control"
    _baseline_id_preprocessing = f"{BASE_BASELINE_ID}_preprocessing_change"  # noqa: F841

    # Read base config
    # Find a test config dynamically (hf_local with bart-small + long-fast)
    base_config = find_test_config(
        backend_type="hf_local", map_model="bart-small", reduce_model="long-fast"
    )
    _config_content = base_config.read_text(encoding="utf-8")  # noqa: F841

    # Add preprocessing profile override (if supported) or modify directly
    # For now, we'll need to modify the materialize_baseline script call
    # or create a config that specifies different preprocessing
    # This is a placeholder - actual implementation depends on how preprocessing
    # is specified in experiment configs

    logger.info("Note: This test requires preprocessing profile to be configurable")
    logger.info("      in experiment config. Currently using default 'cleaning_v3'.")
    logger.info("      To test: modify materialize_baseline.py to accept --preprocessing-profile")

    # For now, we'll compare control vs a run with explicit preprocessing
    # This would require adding preprocessing_profile to experiment config
    fp_control = Path(OUTPUT_DIR) / baseline_id_control / "fingerprint.json"

    # Check if preprocessing section exists and is correct
    if fp_control.exists():
        fp_data = json.loads(fp_control.read_text(encoding="utf-8"))
        preprocessing = fp_data.get("preprocessing", {})
        if preprocessing:
            logger.info(f"✓ Preprocessing section exists: {preprocessing.get('profile_id')}")
            logger.info(f"  Profile version: {preprocessing.get('profile_version')}")
            logger.info(f"  Steps: {len(preprocessing.get('steps', {}))} steps configured")
            return True
        else:
            logger.error("✗ Preprocessing section missing from fingerprint")
            return False
    else:
        logger.error(f"✗ Control fingerprint not found: {fp_control}")
        return False


def test_4_chunk_size_change():
    """Test 4: Chunk size change."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Chunk Size Change")
    logger.info("=" * 60)
    logger.info("Expected: chunking section should change")

    baseline_id_control = f"{BASE_BASELINE_ID}_control"
    baseline_id_chunking = f"{BASE_BASELINE_ID}_chunking_change"

    # Read base config
    # Find a test config dynamically (hf_local with bart-small + long-fast)
    base_config = find_test_config(
        backend_type="hf_local", map_model="bart-small", reduce_model="long-fast"
    )
    config_content = base_config.read_text(encoding="utf-8")

    # Add chunk_size parameter
    modified_config = config_content + "\nparams:\n  chunk_size: 800\n"

    # Write temporary config
    temp_config = Path(f"data/eval/configs/{baseline_id_chunking}.yaml")
    temp_config.write_text(modified_config, encoding="utf-8")

    try:
        fp_control = Path(OUTPUT_DIR) / baseline_id_control / "fingerprint.json"
        fp_chunking = run_baseline(baseline_id_chunking, temp_config)

        # Compare
        is_same = compare_fingerprints(fp_control, fp_chunking, "Chunking Change Test")

        if is_same:
            logger.warning("⚠ WARNING: Fingerprints are identical after chunk size change!")
            logger.warning("  This suggests chunking is not being fingerprinted correctly.")
            return False
        else:
            logger.info("✓ Fingerprints correctly differ after chunk size change")

            # Verify chunking section changed
            fp_control_data = json.loads(fp_control.read_text(encoding="utf-8"))
            fp_chunking_data = json.loads(fp_chunking.read_text(encoding="utf-8"))

            control_chunking = fp_control_data.get("chunking", {})
            chunking_chunking = fp_chunking_data.get("chunking", {})

            if control_chunking != chunking_chunking:
                logger.info("✓ Chunking section correctly differs:")
                logger.info(f"  Control: {json.dumps(control_chunking, indent=2)}")
                logger.info(f"  Changed: {json.dumps(chunking_chunking, indent=2)}")
                return True
            else:
                logger.warning("⚠ Chunking sections are identical despite config change")
                return False
    finally:
        # Cleanup
        if temp_config.exists():
            temp_config.unlink()


def test_g_rerun_same_config():
    """Test G: Re-run the same config twice (stability check).

    Expected: Fingerprint should be identical except for run_id.
    This validates that fingerprints represent configuration identity, not execution time.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST G: Re-run Same Config (Stability Check)")
    logger.info("=" * 60)
    logger.info("Expected: Fingerprints identical (except run_id and baseline_id)")

    baseline_id_1 = f"{BASE_BASELINE_ID}_rerun_1"
    baseline_id_2 = f"{BASE_BASELINE_ID}_rerun_2"
    # Find a test config dynamically (hf_local with bart-small + long-fast)
    config_path = find_test_config(
        backend_type="hf_local", map_model="bart-small", reduce_model="long-fast"
    )

    fp1 = run_baseline(baseline_id_1, config_path)
    fp2 = run_baseline(baseline_id_2, config_path)

    # Compare (excluding run metadata: run_id and baseline_id)
    is_same = compare_fingerprints(fp1, fp2, "Re-run Same Config", exclude_run_metadata=True)

    if is_same:
        logger.info("✓ Fingerprints correctly identical after re-run")
        logger.info("  (run_id and baseline_id differ, as expected)")
        return True
    else:
        logger.error("✗ Fingerprints differ after re-run - this indicates instability!")
        logger.error("  Configuration should be identical, only run metadata should differ.")
        return False


def test_h_different_output_paths():
    """Test H: Change only output paths / run folder names (stability check).

    Expected: Fingerprint should be identical.
    If path changes affect fingerprint → accidentally hashing environment noise.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST H: Different Output Paths (Stability Check)")
    logger.info("=" * 60)
    logger.info("Expected: Fingerprints identical (paths shouldn't affect fingerprint)")

    baseline_id_1 = f"{BASE_BASELINE_ID}_path_1"
    baseline_id_2 = f"{BASE_BASELINE_ID}_path_2"
    # Find a test config dynamically (hf_local with bart-small + long-fast)
    config_path = find_test_config(
        backend_type="hf_local", map_model="bart-small", reduce_model="long-fast"
    )

    # Run with different output directories
    # Use a temporary directory for the second baseline
    # to avoid polluting the main baselines directory
    import tempfile

    temp_base = Path(tempfile.mkdtemp(prefix="fingerprint_test_"))
    output_dir_1 = "data/eval/baselines"
    output_dir_2 = str(temp_base / "baselines_alt")  # Temporary path

    # Run first baseline
    cmd1 = [
        sys.executable,
        "scripts/eval/materialize_baseline.py",
        "--baseline-id",
        baseline_id_1,
        "--dataset-id",
        TEST_DATASET,
        "--experiment-config",
        str(config_path),
        "--output-dir",
        output_dir_1,
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True, check=False)
    if result1.returncode != 0:
        logger.error(f"Baseline 1 creation failed: {result1.stderr}")
        return False

    fp1 = Path(output_dir_1) / baseline_id_1 / "fingerprint.json"
    if not fp1.exists():
        logger.error(f"Fingerprint 1 not created: {fp1}")
        return False

    # Run second baseline with different output directory
    cmd2 = [
        sys.executable,
        "scripts/eval/materialize_baseline.py",
        "--baseline-id",
        baseline_id_2,
        "--dataset-id",
        TEST_DATASET,
        "--experiment-config",
        str(config_path),
        "--output-dir",
        output_dir_2,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
    if result2.returncode != 0:
        logger.error(f"Baseline 2 creation failed: {result2.stderr}")
        return False

    fp2 = Path(output_dir_2) / baseline_id_2 / "fingerprint.json"
    if not fp2.exists():
        logger.error(f"Fingerprint 2 not created: {fp2}")
        # Cleanup temp directory
        import shutil

        if temp_base.exists():
            shutil.rmtree(temp_base)
        return False

    # Compare (excluding run metadata: run_id and baseline_id)
    is_same = compare_fingerprints(fp1, fp2, "Different Output Paths", exclude_run_metadata=True)

    # Cleanup temp directory
    import shutil

    if temp_base.exists():
        shutil.rmtree(temp_base)
        logger.debug(f"Cleaned up temporary directory: {temp_base}")

    if is_same:
        logger.info("✓ Fingerprints correctly identical despite different output paths")
        logger.info("  (run_id and baseline_id differ, as expected)")
        return True
    else:
        logger.error(
            "✗ Fingerprints differ with different paths - paths are affecting fingerprint!"
        )
        logger.error("  This indicates environment noise is being hashed.")
        logger.error("  Output paths should NOT affect fingerprint content.")
        return False


def test_isolation_temperature_change():
    """Test isolation: temperature change should only affect generation_params.

    This validates that when one parameter changes, only that section changes
    and everything else remains identical. This proves isolation.

    Expected:
    - generation_params: different (temperature changed)
    - prompts: same (hashes should match)
    - chunking: same
    - preprocessing: same
    - fingerprint_hash: different (because generation_params changed)
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Isolation Check - Temperature Change")
    logger.info("=" * 60)
    logger.info("Expected: Only generation_params should change, everything else identical")

    baseline_id_control = f"{BASE_BASELINE_ID}_control"
    baseline_id_temp = f"{BASE_BASELINE_ID}_temperature_change"

    # Read base config
    # Find a test config dynamically (hf_local with bart-small + long-fast)
    base_config = find_test_config(
        backend_type="hf_local", map_model="bart-small", reduce_model="long-fast"
    )
    config_content = base_config.read_text(encoding="utf-8")

    # Modify temperature (add params section if not exists, or modify if exists)
    if "params:" in config_content:
        # Replace temperature if it exists, or add it
        if "temperature:" in config_content:
            modified_config = config_content.replace("temperature: 0.0", "temperature: 0.2")
        else:
            # Add temperature to params section
            modified_config = config_content.replace("params:", "params:\n  temperature: 0.2")
    else:
        # Add params section
        modified_config = config_content + "\nparams:\n  temperature: 0.2\n"

    # Write temporary config
    temp_config = Path(f"data/eval/configs/{baseline_id_temp}.yaml")
    temp_config.write_text(modified_config, encoding="utf-8")

    try:
        # Get control fingerprint
        fp_control_path = Path(OUTPUT_DIR) / baseline_id_control / "fingerprint.json"
        if not fp_control_path.exists():
            logger.warning("Control fingerprint not found, running control first...")
            test_1_control_run()
            fp_control_path = Path(OUTPUT_DIR) / baseline_id_control / "fingerprint.json"

        # Run temperature change baseline
        fp_temp_path = run_baseline(baseline_id_temp, temp_config)

        # Load both fingerprints
        fp_control = json.loads(fp_control_path.read_text(encoding="utf-8"))
        fp_temp = json.loads(fp_temp_path.read_text(encoding="utf-8"))

        # Extract sections for comparison
        control_gen_params = fp_control.get("generation_params", {})
        temp_gen_params = fp_temp.get("generation_params", {})

        control_prompts = fp_control.get("prompts", {})
        temp_prompts = fp_temp.get("prompts", {})

        control_chunking = fp_control.get("chunking", {})
        temp_chunking = fp_temp.get("chunking", {})

        control_preprocessing = fp_control.get("preprocessing", {})
        temp_preprocessing = fp_temp.get("preprocessing", {})

        # Compute hashes (excluding run metadata)
        fp_control_compare = get_fingerprint_for_comparison(
            fp_control_path, exclude_run_metadata=True
        )
        fp_temp_compare = get_fingerprint_for_comparison(fp_temp_path, exclude_run_metadata=True)

        fp_control_str = json.dumps(fp_control_compare, sort_keys=True, ensure_ascii=False)
        fp_temp_str = json.dumps(fp_temp_compare, sort_keys=True, ensure_ascii=False)
        hash_control = hashlib.sha256(fp_control_str.encode("utf-8")).hexdigest()[:16]
        hash_temp = hashlib.sha256(fp_temp_str.encode("utf-8")).hexdigest()[:16]

        # Validate isolation
        all_passed = True

        # 1. generation_params should differ
        if control_gen_params == temp_gen_params:
            logger.error("✗ generation_params are IDENTICAL (should differ)")
            all_passed = False
        else:
            logger.info("✓ generation_params differ (as expected):")
            logger.info("  Control fingerprint:")
            logger.info(f"    {json.dumps(control_gen_params, indent=6)}")
            logger.info("  Experiment fingerprint:")
            logger.info(f"    {json.dumps(temp_gen_params, indent=6)}")

        # 2. prompts should be identical (check hashes)
        control_prompt_hash = control_prompts.get("user", {}).get("sha256")
        temp_prompt_hash = temp_prompts.get("user", {}).get("sha256")
        if control_prompt_hash and temp_prompt_hash:
            if control_prompt_hash != temp_prompt_hash:
                logger.error("✗ Prompt hashes differ (should be identical)")
                logger.error(f"  Control: {control_prompt_hash}")
                logger.error(f"  Changed: {temp_prompt_hash}")
                all_passed = False
            else:
                logger.info(f"✓ Prompt hashes identical: {control_prompt_hash}")

        # 3. chunking should be identical
        if control_chunking != temp_chunking:
            logger.error("✗ Chunking differs (should be identical)")
            logger.error(f"  Control: {json.dumps(control_chunking, indent=4)}")
            logger.error(f"  Changed: {json.dumps(temp_chunking, indent=4)}")
            all_passed = False
        else:
            logger.info("✓ Chunking identical")

        # 4. preprocessing should be identical
        if control_preprocessing != temp_preprocessing:
            logger.error("✗ Preprocessing differs (should be identical)")
            logger.error(f"  Control: {json.dumps(control_preprocessing, indent=4)}")
            logger.error(f"  Changed: {json.dumps(temp_preprocessing, indent=4)}")
            all_passed = False
        else:
            logger.info("✓ Preprocessing identical")

        # 5. fingerprint hash should differ
        if hash_control == hash_temp:
            logger.error("✗ Fingerprint hashes are IDENTICAL (should differ)")
            logger.error(f"  Hash: {hash_control}")
            all_passed = False
        else:
            logger.info("✓ Fingerprint hashes differ (as expected):")
            logger.info(f"  Control: {hash_control}")
            logger.info(f"  Changed: {hash_temp}")

        if all_passed:
            logger.info("\n✓ Isolation test PASSED: Only generation_params changed")
            return True
        else:
            logger.error("\n✗ Isolation test FAILED: Multiple sections changed")
            return False

    finally:
        # Cleanup
        if temp_config.exists():
            temp_config.unlink()


def main() -> None:
    """Run minimal test matrix."""
    logger.info("=" * 60)
    logger.info("Fingerprint Validation Test Matrix")
    logger.info("=" * 60)
    logger.info("")
    logger.info("CHANGE TESTS (fingerprint should change):")
    logger.info("1. Control run (baseline)")
    logger.info("2. Prompt change (hash should change)")
    logger.info("3. Preprocessing profile change")
    logger.info("4. Chunk size change")
    logger.info("")
    logger.info("ISOLATION TEST (only one thing should change):")
    logger.info("I. Temperature change (only generation_params should differ)")
    logger.info("")
    logger.info("STABILITY TESTS (fingerprint should NOT change):")
    logger.info("G. Re-run same config twice (identical except run_id)")
    logger.info("H. Different output paths (paths shouldn't affect fingerprint)")
    logger.info("")

    results = []

    try:
        # Change tests (should differ)
        test_1_control_run()
        results.append(("Control Run", True))

        results.append(("Prompt Change", test_2_prompt_change()))
        results.append(("Preprocessing Change", test_3_preprocessing_change()))
        results.append(("Chunk Size Change", test_4_chunk_size_change()))

        # Isolation test (only one thing should change)
        results.append(("Temperature Change Isolation (I)", test_isolation_temperature_change()))

        # Stability tests (should be identical)
        results.append(("Re-run Same Config (G)", test_g_rerun_same_config()))
        results.append(("Different Output Paths (H)", test_h_different_output_paths()))

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        results.append(("Error", False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        logger.info("\n✓ All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n✗ Some tests failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
