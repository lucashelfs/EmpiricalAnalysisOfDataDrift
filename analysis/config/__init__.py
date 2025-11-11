"""Configuration package for analysis scripts."""

from .paths import (
    PROJECT_ROOT,
    ANALYSIS_ROOT,
    ANALYSIS_RESULTS,
    BATCH_SIZE_RESULTS,
    STATISTICAL_RESULTS,
    VALIDATION_RESULTS,
    DRIFT_ANALYSIS_RESULTS,
    MULTI_RUN_RESULTS,
    COMPARISON_RESULTS,
    ensure_results_dirs,
    get_timestamped_dir,
)

__all__ = [
    "PROJECT_ROOT",
    "ANALYSIS_ROOT",
    "ANALYSIS_RESULTS",
    "BATCH_SIZE_RESULTS",
    "STATISTICAL_RESULTS",
    "VALIDATION_RESULTS",
    "DRIFT_ANALYSIS_RESULTS",
    "MULTI_RUN_RESULTS",
    "COMPARISON_RESULTS",
    "ensure_results_dirs",
    "get_timestamped_dir",
]
