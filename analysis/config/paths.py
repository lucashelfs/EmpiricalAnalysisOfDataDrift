"""
Centralized path configuration for analysis scripts.

This module provides consistent paths across all analysis scripts,
making it easy to reorganize outputs and maintain the codebase.
"""

from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Analysis directories
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
ANALYSIS_SCRIPTS = ANALYSIS_ROOT / "scripts"
ANALYSIS_RESULTS = ANALYSIS_ROOT / "results"

# Result subdirectories by analysis type
BATCH_SIZE_RESULTS = ANALYSIS_RESULTS / "batch_size"
STATISTICAL_RESULTS = ANALYSIS_RESULTS / "statistical"
VALIDATION_RESULTS = ANALYSIS_RESULTS / "validation"
DRIFT_ANALYSIS_RESULTS = ANALYSIS_RESULTS / "drift_analysis"
MULTI_RUN_RESULTS = ANALYSIS_RESULTS / "multi_run"

# Experiment results (for compatibility with existing scripts)
COMPARISON_RESULTS = PROJECT_ROOT / "comparison_results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

# Ensure result directories exist
def ensure_results_dirs():
    """Create all result directories if they don't exist."""
    for dir_path in [
        BATCH_SIZE_RESULTS,
        STATISTICAL_RESULTS,
        VALIDATION_RESULTS,
        DRIFT_ANALYSIS_RESULTS,
        MULTI_RUN_RESULTS,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_timestamped_dir(base_dir: Path, prefix: str = "run") -> Path:
    """
    Create a timestamped subdirectory for experiment results.

    Args:
        base_dir: Base directory for results
        prefix: Prefix for the directory name (default: "run")

    Returns:
        Path to the timestamped directory
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = base_dir / f"{prefix}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


# Legacy path compatibility (for scripts not yet updated)
def get_legacy_output_dir():
    """Returns the legacy comparison_results directory for backward compatibility."""
    return str(COMPARISON_RESULTS)
