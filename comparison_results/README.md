# Comparison Results Directory

This directory contains all experimental results from the data drift detection framework evaluation. The results are organized by dataset and include performance metrics, visualizations, and drift analysis data.

## Directory Structure

### Root Level Files
- **`consolidated_results.csv`** - Combined performance metrics for all experiments across datasets, batch sizes, and drift detection methods
- **`consolidated_drift_results.json`** - **Synthetic datasets only** - Consolidated drift detection analysis comparing ground truth synthetic drift points vs detected drift points
- **`drift_detection_results.csv`** - Additional drift detection metrics and statistics

### Dataset Directories
Each dataset has its own subdirectory containing:

```
[dataset_name]/
├── [dataset_name]_results.csv     # Performance metrics for this dataset
├── detected_drifts/               # Drift detection visualizations
│   └── *.png                      # Drift point plots for different batch sizes
├── feature_plots/                 # Individual feature distribution plots
│   └── *.png                      # Feature plots showing data distributions
├── heatmaps/                      # Drift detection heatmaps
│   └── *.png                      # P-value and distance heatmaps
├── metrics/                       # Additional metric visualizations
│   └── *.png                      # Performance comparison charts
└── drift_files/                   # Drift analysis files (synthetic datasets only)
    └── *.json                     # Detailed drift point analysis
```

## Dataset Types

### Real Datasets
- **`electricity/`** - Electricity market dataset
- **`magic/`** - Magic Gamma Telescope dataset
- **`MULTISEA/`** - Multi-concept SEA dataset
- **`MULTISTAGGER/`** - Multi-concept STAGGER dataset
- **`SEA/`** - SEA concept drift dataset
- **`STAGGER/`** - STAGGER concept drift dataset

### Synthetic Datasets
- **`synthetic_dataset_no_drifts/`** - Control dataset without drifts
- **`synthetic_dataset_with_parallel_drifts_abrupt/`** - Parallel abrupt drifts
- **`synthetic_dataset_with_parallel_drifts_incremental/`** - Parallel incremental drifts
- **`synthetic_dataset_with_switching_drifts_abrupt/`** - Switching abrupt drifts
- **`synthetic_dataset_with_switching_drifts_incremental/`** - Switching incremental drifts

### Balanced/Imbalanced Variants
- **`Abrupt (bal.)/`** and **`Abrupt (imbal.)/`** - Abrupt drift scenarios
- **`Incremental (bal.)/`** and **`Incremental (imbal.)/`** - Incremental drift scenarios
- **`Incremental-*-reoccurring (bal./imbal.)/`** - Reoccurring drift patterns
- **`Incremental-gradual (bal./imbal.)/`** - Gradual drift patterns

## File Formats

### CSV Files
- **Performance metrics**: accuracy, precision, recall, F1-score, AUC
- **Drift statistics**: number of detected drifts, batch information
- **Experimental parameters**: batch sizes, algorithms, drift scenarios

### JSON Files
- **Synthetic drift analysis**: comparison between true and detected drift points
- **Batch-level drift information**: detailed drift timing and feature analysis
- **Note**: JSON files are generated exclusively for synthetic datasets since they have known ground truth drift points. The ground truth for real world concept drift datasets was not considered in depth on the comparison for the current version of this work and stands as a future improvement.

### PNG Files
- **Feature plots**: Distribution visualizations for each dataset feature
- **Drift detection plots**: Timeline showing detected vs actual drift points
- **Heatmaps**: P-value and distance matrices for drift detection methods
- **Performance charts**: Comparative analysis across methods and batch sizes

## Drift Detection Methods

Results include comparisons of:
- **Base** - No drift detection (baseline)
- **KS95** - Kolmogorov-Smirnov test (95% confidence)
- **KS90** - Kolmogorov-Smirnov test (90% confidence)
- **HD** - Hellinger Distance Drift Detection Method
- **JS** - Jensen-Shannon Drift Detection Method

## Batch Sizes Evaluated
- 1000, 1500, 2000, 2500 samples per batch

## Usage

### For Analyzing Overall Performance
```bash
# View consolidated results
head comparison_results/consolidated_results.csv
```

### For Examining Specific Dataset Results
```bash
# Navigate to dataset directory
cd comparison_results/[dataset_name]/
ls -la  # View available files and subdirectories
```

### Viewing Visualizations
Open PNG files in the respective subdirectories to examine:
- Feature distributions and drift points
- Drift detection performance heatmaps
- Performance comparison charts

The results support the empirical analysis presented in the SBBD 2024 research paper on data drift detection techniques and the submission for JIDM 2025.
