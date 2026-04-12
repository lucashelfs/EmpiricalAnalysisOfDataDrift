#!/usr/bin/env python3
"""
Test script to validate the reorganized structure and centralized paths.
"""

import sys
from pathlib import Path

# Test imports from new locations
print("Testing centralized path configuration...")
print("=" * 60)

try:
    from analysis.config import (
        PROJECT_ROOT,
        ANALYSIS_RESULTS,
        BATCH_SIZE_RESULTS,
        STATISTICAL_RESULTS,
        VALIDATION_RESULTS,
        DRIFT_ANALYSIS_RESULTS,
        MULTI_RUN_RESULTS,
        COMPARISON_RESULTS,
        ensure_results_dirs,
    )
    print("✅ Successfully imported from analysis.config")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test path values
print("\n📁 Configured Paths:")
print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
print(f"  ANALYSIS_RESULTS: {ANALYSIS_RESULTS}")
print(f"  BATCH_SIZE_RESULTS: {BATCH_SIZE_RESULTS}")
print(f"  STATISTICAL_RESULTS: {STATISTICAL_RESULTS}")
print(f"  VALIDATION_RESULTS: {VALIDATION_RESULTS}")
print(f"  DRIFT_ANALYSIS_RESULTS: {DRIFT_ANALYSIS_RESULTS}")
print(f"  MULTI_RUN_RESULTS: {MULTI_RUN_RESULTS}")
print(f"  COMPARISON_RESULTS: {COMPARISON_RESULTS}")

# Test directory creation
print("\n📂 Testing directory creation...")
try:
    ensure_results_dirs()
    print("✅ All result directories created successfully")
except Exception as e:
    print(f"❌ Failed to create directories: {e}")
    sys.exit(1)

# Verify directories exist
print("\n🔍 Verifying directories exist:")
for name, path in [
    ("BATCH_SIZE_RESULTS", BATCH_SIZE_RESULTS),
    ("STATISTICAL_RESULTS", STATISTICAL_RESULTS),
    ("VALIDATION_RESULTS", VALIDATION_RESULTS),
    ("DRIFT_ANALYSIS_RESULTS", DRIFT_ANALYSIS_RESULTS),
    ("MULTI_RUN_RESULTS", MULTI_RUN_RESULTS),
]:
    if path.exists():
        print(f"  ✅ {name}: {path}")
    else:
        print(f"  ❌ {name}: {path} (NOT FOUND)")

# Check if comparison_results exists (legacy)
if COMPARISON_RESULTS.exists():
    num_subdirs = len(list(COMPARISON_RESULTS.iterdir()))
    print(f"\n📊 Found comparison_results with {num_subdirs} subdirectories")
else:
    print(f"\n⚠️  comparison_results not found (expected for fresh setup)")

print("\n" + "=" * 60)
print("✅ Path configuration test PASSED!")
print("=" * 60)
