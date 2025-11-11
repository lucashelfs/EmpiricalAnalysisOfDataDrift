#!/usr/bin/env python3
"""
30-Run Naive Bayes Experiment

This script runs a comprehensive 30-run experiment focused on Naive Bayes
to get statistically robust results for drift detection effectiveness.
"""

from codes.synthetic_multi_run_experiment import SyntheticMultiRunExperiment

def main():
    print("Starting 30-Run Naive Bayes Experiment")
    print("=" * 50)
    
    # Configure experiment for 30 runs, NB only
    experiment = SyntheticMultiRunExperiment(
        target_runs=30,
        algorithms=["NB"],  # Only Naive Bayes
        synthetic_scenarios=["all"],  # All synthetic scenarios
        batch_sizes=[1000, 2000],
        dataframe_size=80000,  # Standard size
        total_drift_length=20000,
        num_drifts=2
    )
    
    print("Experiment Configuration:")
    print(f"- Target runs: 30")
    print(f"- Algorithm: Naive Bayes only")
    print(f"- Scenarios: All synthetic scenarios")
    print(f"- Batch sizes: [1000, 2000]")
    print(f"- Expected data points: ~1,500")
    print(f"- Estimated time: ~30 minutes")
    
    # Run the experiment
    print("\nStarting experiment...")
    experiment.run_multi_experiment()
    
    # Analyze results
    print("\nAnalyzing results...")
    experiment.analyze_results()
    
    print(f"\nExperiment completed!")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Results saved to: {experiment.base_output_dir}")

if __name__ == "__main__":
    main()
