# Experiment Timing System Documentation

## Overview

The timing system provides comprehensive measurement and analysis of execution times for your data drift detection experiments. It captures timing data at three levels:

1. **Total Experiment Time** - Complete pipeline execution
2. **Per-Dataset Timing** - Time for each dataset across all batch sizes  
3. **Per-Phase Timing** - Individual drift detection and evaluation phases

## Key Features

### Hierarchical Timing Measurement
- **Experiment Level**: Total time for all datasets and configurations
- **Dataset Level**: Time per dataset across different batch sizes
- **Phase Level**: Separate timing for drift detection vs evaluation phases
- **Technique Level**: Individual timing for HDDDM, KSDDM_95, KSDDM_90, and JSDDM

### Automatic Data Collection
- **Context-Aware**: Automatically tracks current dataset and batch size
- **Phase Separation**: Distinguishes between drift detection and evaluation time
- **Technique Tracking**: Measures individual drift detection method performance
- **Granular Analysis**: Each technique timed separately for direct comparison

### Comprehensive Output
- **JSON Results**: Detailed timing data in `experiment_timing_results.json`
- **CSV Summary**: Enhanced tabular format with technique column in `timing_summary.csv`
- **Console Summary**: Real-time progress and final summary
- **Technique Analysis**: Dedicated analysis tools for technique comparison

## Files Created

After running your experiments with timing enabled, you'll find:

```
comparison_results/
├── experiment_timing_results.json    # Detailed timing data
└── timing_summary.csv                # Tabular summary

execution_time_analysis/              # Analysis reports
├── timing_analysis_overview.png
├── technique_performance.png
├── technique_detailed_analysis.png
├── detailed_timing_data.csv
├── timing_analysis_report.md
├── technique_analysis_report.md
└── technique_performance_summary.csv
```

## Usage

### Running Experiments with Timing

Your experiments now automatically include timing measurement. Simply run:

```bash
python codes/comparisor.py
```

The timing system will:
1. Start timing when the experiment begins
2. Track each dataset and batch size combination
3. Measure drift detection and evaluation phases separately
4. Save comprehensive timing results
5. Print a summary when complete

### Analyzing Timing Results

Use the timing analysis utility to generate detailed reports:

```bash
python codes/timing_analysis.py comparison_results/experiment_timing_results.json
```

This creates:
- **Visual charts** showing performance by batch size and dataset
- **Performance heatmaps** comparing all combinations
- **Technique comparisons** showing which methods are fastest
- **Summary report** with key insights and recommendations

## Timing Data Structure

### JSON Output Format
```json
{
  "experiment_metadata": {
    "start_time": "2025-01-15T10:30:00",
    "end_time": "2025-01-15T12:45:30", 
    "total_duration": 8130.45,
    "datasets_processed": 21
  },
  "dataset_timings": {
    "electricity": {
      "total_time": 245.67,
      "batch_timings": {
        "1000": {
          "drift_detection": 45.2,
          "evaluation": 89.3,
          "total": 134.5
        }
      }
    }
  },
  "technique_performance": {
    "HDDDM": {
      "total_time": 1245.6,
      "executions": 84,
      "avg_time": 14.8
    }
  }
}
```

### Enhanced CSV Output Format (with Technique-Level Granularity)
```csv
dataset,batch_size,technique,phase,duration_seconds,type
electricity,1000,HDDDM,drift_detection,12.5,technique_timing
electricity,1000,KSDDM_95,drift_detection,10.8,technique_timing
electricity,1000,KSDDM_90,drift_detection,11.1,technique_timing
electricity,1000,JSDDM,drift_detection,11.2,technique_timing
electricity,1000,ALL,evaluation,89.3,phase_timing
electricity,1000,ALL,total,134.5,batch_total
```

## Key Insights You Can Extract

### Batch Size Performance
- Which batch sizes are most efficient overall?
- How does batch size affect drift detection vs evaluation time?
- What's the optimal batch size for your hardware?

### Dataset Complexity Analysis  
- Which datasets take longest to process?
- Are synthetic datasets faster than real-world data?
- How does dataset size correlate with processing time?

### Technique Efficiency
- Which drift detection methods are fastest?
- How much overhead does each technique add?
- What's the performance vs accuracy trade-off?

### Phase Breakdown
- Is drift detection or evaluation the bottleneck?
- How much time is spent on visualization vs computation?
- Where should optimization efforts focus?

## Future Enhancements

Potential improvements to consider:

- **Memory usage tracking** alongside timing
- **Real-time timing dashboard** for long experiments
- **Comparative timing** across different hardware configurations
