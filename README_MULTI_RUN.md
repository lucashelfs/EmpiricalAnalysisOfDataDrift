# Multi-Run Experiment Framework

A comprehensive framework for running drift detection experiments multiple times to achieve statistical significance while maintaining reproducible datasets and efficient storage.

## 🚀 Quick Start

```python
from codes.multi_run_experiment import MultiRunExperimentRunner

# Quick test (3 datasets, 5 runs, ~10 minutes)
runner = MultiRunExperimentRunner(target_runs=5, dataset_filter="quick_test")
runner.run_multi_experiment()

# Full experiment (all datasets, 30 runs)
runner = MultiRunExperimentRunner(target_runs=30, dataset_filter="all")
runner.run_multi_experiment()
```

## ✨ Key Features

- ✅ **Reproducible synthetic datasets** (constant seed=42)
- ✅ **Variable algorithm randomness** per run
- ✅ **Incremental run extension** (30→50→100)
- ✅ **Smart storage optimization** (~2.5GB for 30 runs)
- ✅ **Dataset filtering** for testing
- ✅ **Statistical analysis suite**
- ✅ **Automatic recovery** from interruptions
- ✅ **Comprehensive documentation**

## 📊 Statistical Analysis

The framework automatically generates:

- **Mean ± Standard Deviation** for all metrics
- **95% Confidence Intervals**
- **Statistical significance tests**
- **Technique performance rankings**
- **Dataset difficulty analysis**
- **Batch size impact assessment**

## 🎯 Use Cases

### Quick Validation (10 minutes)
```python
runner = MultiRunExperimentRunner(target_runs=5, dataset_filter="quick_test")
runner.run_multi_experiment()
```

### Algorithm Testing (45 minutes)
```python
runner = MultiRunExperimentRunner(target_runs=10, dataset_filter="synthetic_only")
runner.run_multi_experiment()
```

### Full Statistical Analysis (6 hours)
```python
runner = MultiRunExperimentRunner(target_runs=30, dataset_filter="all")
runner.run_multi_experiment()
```

### Extending Experiments
```python
# Extend from 30 to 50 runs
runner = MultiRunExperimentRunner(
    target_runs=50,
    existing_experiment_id="multi_run_20250102_223000"
)
runner.run_multi_experiment()
```

## 📁 Directory Structure

```
comparison_results/
└── multi_run_20250102_223000/
    ├── shared_datasets/          # Generated once, symlinked
    ├── runs/
    │   ├── run_001/ ... run_030/ # Individual run results
    └── analysis/                 # Statistical summaries
        ├── all_runs_combined.csv
        ├── statistical_summary.csv
        ├── technique_comparison.csv
        └── confidence_intervals.csv
```

## 🔧 Configuration Options

### Dataset Filters

| Filter | Description | Use Case |
|--------|-------------|----------|
| `"quick_test"` | 3 synthetic datasets | Rapid testing |
| `"synthetic_only"` | All synthetic datasets | Algorithm validation |
| `"real_only"` | All real-world datasets | Real-world testing |
| `"all"` | Complete dataset suite | Full experiment |
| `["custom", "list"]` | Custom selection | Specific testing |

### Advanced Configuration

```python
runner = MultiRunExperimentRunner(
    target_runs=30,
    dataset_filter="synthetic_only",
    batch_sizes=[1000, 2500],  # Custom batch sizes
    algorithms=["NB", "HT"],   # Multiple algorithms
    base_seed=42               # Reproducibility seed
)
```

## 📈 Time Estimates

| Configuration | Datasets | Runs | Time |
|---------------|----------|------|------|
| Quick test | 3 synthetic | 5 | ~10 min |
| Synthetic only | 5 synthetic | 30 | ~2 hours |
| All datasets | 21 datasets | 30 | ~6 hours |

## 🛠️ Installation & Setup

1. **Ensure dependencies are installed**:
   ```bash
   pip install numpy pandas scipy scikit-learn
   ```

2. **Run the framework**:
   ```python
   from codes.multi_run_experiment import MultiRunExperimentRunner
   runner = MultiRunExperimentRunner(target_runs=5, dataset_filter="quick_test")
   runner.run_multi_experiment()
   ```

## 📚 Documentation

- **[Complete Guide](docs/multi_run_experiment_guide.md)** - Comprehensive documentation
- **[Quick Start](docs/multi_run_quick_start.md)** - 5-minute getting started
- **[Example Script](examples/multi_run_example.py)** - Demonstration script

## 🔄 Reproducibility & Verification

### Seed Management
- **Dataset Generation**: Always uses `seed=42` (constant across runs)
- **Algorithm Randomness**: Uses `seed=42 + run_id * 1000` (varies per run)

### Verification
```python
# Run same experiment twice - datasets should be identical, results should vary
runner1 = MultiRunExperimentRunner(target_runs=5, dataset_filter="quick_test")
runner1.run_multi_experiment()

runner2 = MultiRunExperimentRunner(target_runs=5, dataset_filter="quick_test")
runner2.run_multi_experiment()
```

## 🚨 Error Handling & Recovery

### Automatic Recovery
- **Run-level failures**: Continue with remaining runs
- **Dataset-level failures**: Skip failed dataset, continue with others
- **Checkpoint system**: Resume from last completed run

### Manual Recovery
```python
# Check experiment status
runner = MultiRunExperimentRunner(existing_experiment_id="multi_run_20250102_223000")
runner.print_experiment_status()

# Resume interrupted experiment
runner.run_multi_experiment()  # Continues from where it left off
```

## 📊 Analysis Tools

### Built-in Analyzer
```python
from codes.multi_run_experiment import MultiRunAnalyzer

analyzer = MultiRunAnalyzer("comparison_results/multi_run_20250102_223000")
analyzer.generate_complete_analysis()
```

### Generated Files
- `all_runs_combined.csv` - Raw results from all runs
- `statistical_summary.csv` - Mean, std, min, max for all metrics
- `technique_comparison.csv` - Performance comparison with confidence intervals
- `dataset_analysis.csv` - Dataset difficulty ranking
- `batch_size_analysis.csv` - Batch size impact analysis
- `confidence_intervals.csv` - 95% confidence intervals

## 🎯 Best Practices

### Development Workflow
1. **Start small**: Use `"quick_test"` filter first
2. **Validate results**: Check statistical analysis makes sense
3. **Scale gradually**: synthetic_only → fast_real → all
4. **Monitor progress**: Check experiment status regularly

### Production Workflow
1. **Plan storage**: Ensure sufficient disk space (~2.5GB for 30 runs)
2. **Set expectations**: Full experiments take 6-8 hours
3. **Use screen/tmux**: For long-running experiments
4. **Backup results**: Copy experiment directories after completion

## 🔍 Troubleshooting

### Common Issues

**"No results found to aggregate"**
- Check if runs completed successfully
- Verify `metadata.json` files have `"run_completed": true`

**Memory issues**
- Reduce batch sizes: `batch_sizes=[1000]`
- Use dataset filtering: `dataset_filter="quick_test"`

**Symlink errors (Windows)**
- Framework automatically falls back to copying
- May use more disk space

## 🤝 Integration

The framework integrates seamlessly with your existing codebase:

- **Minimal changes**: Wraps existing `run_full_experiment()` function
- **Backward compatibility**: Single-run functionality intact
- **Same output format**: Compatible with existing analysis scripts
- **Enhanced metadata**: Adds run tracking and statistical analysis

## 📝 Example Usage

```python
# Progressive testing workflow
from codes.multi_run_experiment import MultiRunExperimentRunner

# Step 1: Quick validation
quick = MultiRunExperimentRunner(target_runs=5, dataset_filter="quick_test")
quick.run_multi_experiment()

# Step 2: Synthetic validation
synthetic = MultiRunExperimentRunner(
    target_runs=10,
    dataset_filter="synthetic_only",
    existing_experiment_id=quick.experiment_id
)
synthetic.run_multi_experiment()

# Step 3: Full production
full = MultiRunExperimentRunner(
    target_runs=30,
    dataset_filter="all",
    existing_experiment_id=synthetic.experiment_id
)
full.run_multi_experiment()
```

## 🎉 Benefits

### For Research
- **Statistical rigor**: 30 runs enable robust confidence intervals
- **Reproducibility**: Consistent dataset generation with verification
- **Publication ready**: Comprehensive statistical analysis

### For Development
- **Quick validation**: Test framework changes rapidly
- **Incremental scaling**: Build confidence before full runs
- **Error resilience**: Automatic recovery from failures

### For Analysis
- **Complete granularity**: Every metric slice available
- **Built-in statistics**: No manual calculation needed
- **Flexible filtering**: Analyze any subset of results

## 🚀 Getting Started

1. **Try the quick test**:
   ```python
   from codes.multi_run_experiment import MultiRunExperimentRunner
   runner = MultiRunExperimentRunner(target_runs=5, dataset_filter="quick_test")
   runner.run_multi_experiment()
   ```

2. **Check the results**:
   ```python
   runner.print_experiment_status()
   # Results saved to: comparison_results/multi_run_TIMESTAMP/
   ```

3. **Scale up gradually**:
   - Move to `"synthetic_only"` for algorithm testing
   - Use `"all"` for full statistical analysis
   - Extend runs incrementally as needed

## 📞 Support

For questions or issues:
1. Check the [troubleshooting guide](docs/multi_run_experiment_guide.md#troubleshooting)
2. Review the [example script](examples/multi_run_example.py)
3. Examine the comprehensive [documentation](docs/multi_run_experiment_guide.md)

---

**Ready to achieve statistical significance in your drift detection experiments? Start with the quick test above!**
