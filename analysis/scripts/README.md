# Analysis Scripts

This directory contains all analysis scripts organized by category.

## Directory Structure

```
analysis/scripts/
├── batch_size/          ✅ Updated - uses centralized paths
│   └── batch_size_analysis.py
├── statistical/         ⚠️  Needs updating
│   ├── fresh_statistical_significance_analysis.py
│   ├── statistical_significance_analysis.py
│   └── statistical_significance_summary.py
├── validation/          ⚠️  Needs updating
│   ├── fresh_validation_comparisor.py
│   ├── simple_fresh_comparisor.py
│   └── technique_performance_comparison.py
├── drift_analysis/      ⚠️  Needs updating
│   └── drift_detection_analysis.py
└── multi_run/           ⚠️  Needs updating
    ├── synthetic_multi_run_analysis.py
    └── test_large_dataset_with_plots.py
```

## Running Scripts

All scripts should be run from the project root using poetry:

```bash
poetry run python analysis/scripts/batch_size/batch_size_analysis.py
poetry run python analysis/scripts/statistical/statistical_significance_summary.py
# etc.
```

## Output Locations

Scripts output their results to `analysis/results/` organized by category:

- `analysis/results/batch_size/` - Batch size analysis outputs
- `analysis/results/statistical/` - Statistical analysis outputs
- `analysis/results/validation/` - Validation outputs
- `analysis/results/drift_analysis/` - Drift detection analysis outputs
- `analysis/results/multi_run/` - Multi-run analysis outputs

## Updating Scripts to Use Centralized Paths

Scripts that haven't been updated yet still use hard-coded paths. To update a script:

### 1. Add imports at the top

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from analysis.config import (
    STATISTICAL_RESULTS,  # or VALIDATION_RESULTS, DRIFT_ANALYSIS_RESULTS, etc.
    COMPARISON_RESULTS,
    ensure_results_dirs
)
```

### 2. Replace hard-coded output paths

**Before:**
```python
output_dir = "batch_size_analysis"
plt.savefig(f'{output_dir}/plot.png')
```

**After:**
```python
ensure_results_dirs()
output_dir = BATCH_SIZE_RESULTS  # or appropriate results directory
plt.savefig(output_dir / 'plot.png')
```

### 3. Update input paths to use Path objects

**Before:**
```python
results_path = "comparison_results/some_dir/file.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
```

**After:**
```python
comparison_dir = Path(COMPARISON_RESULTS)
results_path = comparison_dir / "some_dir" / "file.csv"
if results_path.exists():
    df = pd.read_csv(results_path)
```

## Example: Successfully Updated Script

See `batch_size/batch_size_analysis.py` for a complete example of a script that has been updated to use centralized paths.

## Testing

Run the path configuration test to verify everything is working:

```bash
poetry run python test_paths.py
```

This will validate that:
- All path imports work correctly
- Result directories are created properly
- Paths point to the correct locations
