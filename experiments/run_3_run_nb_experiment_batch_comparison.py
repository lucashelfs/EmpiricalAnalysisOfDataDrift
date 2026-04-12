#!/usr/bin/env python3
"""
3-Run Naive Bayes Experiment with Batch Size Comparison

This script runs a small experiment with 3 runs to compare different batch sizes
on all synthetic scenarios with 10 drift points.

Configuration:
- 3 runs (for quick testing)
- All synthetic scenarios
- Naive Bayes algorithm only
- Batch sizes: [1000, 1500, 2000, 2500]
- 10 drift points
- 80,000 dataframe size
- 20,000 total drift length
"""

from codes.synthetic_multi_run_experiment import SyntheticMultiRunExperiment

def main():
    print("3-Run Naive Bayes Experiment with Batch Size Comparison")
    print("=" * 60)
    
    # Configure experiment
    experiment = SyntheticMultiRunExperiment(
        target_runs=3,  # Small experiment with 3 runs
        algorithms=["NB"],  # Naive Bayes only
        synthetic_scenarios=["all"],  # All synthetic scenarios
        batch_sizes=[1000, 1500, 2000, 2500],  # Four different batch sizes
        
        # Dataset parameters
        dataframe_size=80000,  # 80k entries
        total_drift_length=20000,  # 20k drift length
        num_drifts=10,  # 10 drift points
        
        # Disable plotting for faster execution
        enable_plots=False
    )
    
    print("Experiment Configuration:")
    print(f"- Target runs: 3")
    print(f"- Algorithm: Naive Bayes only")
    print(f"- Scenarios: All synthetic scenarios")
    print(f"- Batch sizes: [1000, 1500, 2000, 2500]")
    print(f"- Dataset size: 80,000 entries")
    print(f"- Total drift length: 20,000 entries")
    print(f"- Number of drifts: 10")
    print(f"- Plotting: Disabled (for speed)")
    print(f"- Expected time: ~15-20 minutes")
    
    print("\nThis experiment will test:")
    print("- Impact of different batch sizes on drift detection")
    print("- Performance across all synthetic scenarios")
    print("- Statistical variance with 3 independent runs")
    print("- Efficiency of different batch size configurations")
    
    # Run the experiment
    print("\nStarting 3-run batch size comparison experiment...")
    experiment.run_multi_experiment()
    
    # Analyze results
    print("\nAnalyzing experiment results...")
    experiment.analyze_results()
    
    print(f"\n3-run batch size comparison experiment completed!")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Results saved to: {experiment.base_output_dir}")
    
    # Print summary of what to expect in results
    print(f"\nResults will include:")
    print(f"- Performance comparison across 4 batch sizes")
    print(f"- Analysis of all 5 synthetic scenarios")
    print(f"- Statistical summary with 3-run variance")
    print(f"- Batch size optimization recommendations")

if __name__ == "__main__":
    main()
