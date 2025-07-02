# Experiment Timing Analysis Report

## Experiment Overview
- **Total Duration**: 2794.01 seconds
- **Datasets Processed**: 21
- **Start Time**: 2025-06-29T14:15:08.925271
- **End Time**: 2025-06-29T15:01:49.620274

## Performance Insights

### Batch Size Analysis
- **Most Efficient Batch Size**: 2500
- **Least Efficient Batch Size**: 1000

### Dataset Analysis
- **Slowest Dataset**: Incremental (imbal.)
- **Fastest Dataset**: magic

### Phase Analysis
- **Average Drift Detection Time**: 19.43 seconds
- **Average Evaluation Time**: 6.13 seconds
- **Drift Detection vs Evaluation Ratio**: 3.17:1

### Technique Performance
- **Fastest Technique**: JSDDM (1.879s avg)
- **Slowest Technique**: KSDDM_95 (8.098s avg)

## Files Generated
- `timing_analysis_overview.png`: Comprehensive timing visualizations
- `batch_size_performance.png`: Individual batch size performance chart
- `technique_performance.png`: Technique-specific performance charts
- `detailed_timing_data.csv`: Raw timing data for further analysis
- `timing_analysis_report.md`: This summary report