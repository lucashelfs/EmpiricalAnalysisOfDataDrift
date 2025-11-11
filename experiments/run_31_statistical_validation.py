#!/usr/bin/env python3
"""
31-Run Statistical Validation Experiment

This script runs 31 independent experiments on all synthetic datasets 
using default comparisor configurations for statistical significance validation.

Configuration:
- 31 runs (for robust statistical significance)
- All synthetic scenarios (5 scenarios)
- Naive Bayes algorithm (default)
- Default comparisor parameters (80k entries, 2 drifts)
- Batch sizes: [1000, 1500, 2000, 2500]
- Custom folder: statistical_relevance_published_results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codes.synthetic_multi_run_experiment import SyntheticMultiRunExperiment

def main():
    print("31-Run Statistical Validation Experiment")
    print("=" * 60)
    
    # Create experiment with custom folder name and default comparisor parameters
    experiment = SyntheticMultiRunExperiment(
        target_runs=31,                    # Statistical significance (n=31)
        algorithms=["NB"],                 # Naive Bayes (default comparisor)
        synthetic_scenarios=["all"],       # All 5 synthetic scenarios
        batch_sizes=[1000, 1500, 2000, 2500],  # Default comparisor batch sizes
        
        # Default comparisor parameters (exactly matching comparisor.py)
        dataframe_size=80000,              # 80k entries (default)
        total_drift_length=20000,          # 20k drift length
        num_drifts=2,                      # 2 drift points (default)
        features_with_drifts=["feature1", "feature3", "feature5"],  # Default features
        
        # Efficiency settings
        enable_plots=False,                # Disable for speed
        
        # Custom experiment ID for publication-ready folder name
        existing_experiment_id="statistical_relevance_published_results"
    )
    
    print("Experiment Configuration:")
    print(f"- Target runs: 31 (statistical significance)")
    print(f"- Algorithm: Naive Bayes (default comparisor)")
    print(f"- Scenarios: All 5 synthetic scenarios")
    print(f"- Batch sizes: [1000, 1500, 2000, 2500] (default comparisor)")
    print(f"- Dataset size: 80,000 entries (default comparisor)")
    print(f"- Total drift length: 20,000 entries")
    print(f"- Number of drifts: 2 (default comparisor)")
    print(f"- Features with drifts: feature1, feature3, feature5 (default)")
    print(f"- Output folder: statistical_relevance_published_results")
    print(f"- Plotting: Disabled (for speed)")
    print(f"- Expected time: 6-8 hours")
    
    print("\nThis experiment will generate:")
    print("- 31 independent runs with different dataset seeds")
    print("- 3,100 total experimental records (31×5×4×5)")
    print("- Robust statistical analysis for publication")
    print("- Comprehensive performance validation")
    print("- Statistical significance testing capability")
    
    # Run the experiment
    print("\nStarting 31-run statistical validation experiment...")
    experiment.run_multi_experiment()
    
    # Analyze results
    print("\nAnalyzing experiment results...")
    experiment.analyze_results()
    
    print(f"\n31-run statistical validation experiment completed!")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Results saved to: {experiment.base_output_dir}")
    
    # Print summary of what was accomplished
    print(f"\nResults include:")
    print(f"- 31 independent experimental runs")
    print(f"- Statistical analysis with confidence intervals")
    print(f"- Performance comparison across all techniques")
    print(f"- Publication-ready statistical validation")
    print(f"- Comprehensive drift detection evaluation")

if __name__ == "__main__":
    main()
