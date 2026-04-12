#!/usr/bin/env python3
"""
Large Dataset Test with Visualization

This script tests the larger dataset configuration (160k entries, 40k drift length, 10 drifts)
with plotting enabled to verify drift patterns are correctly generated and visualized.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from codes.synthetic_multi_run_experiment import SyntheticMultiRunExperiment
from analysis.config import ensure_results_dirs

def main():
    print("Large Dataset Test with Visualization")
    print("=" * 50)

    # Ensure results directories exist
    ensure_results_dirs()

    # Configure experiment for large dataset with plotting
    experiment = SyntheticMultiRunExperiment(
        target_runs=2,  # Just 2 runs for quick testing
        algorithms=["NB"],  # Only Naive Bayes for speed
        synthetic_scenarios=["parallel_abrupt", "switching_incremental"],  # Test 2 scenarios
        batch_sizes=[1000, 2000],
        
        # NEW: Large dataset parameters
        dataframe_size=160000,  # 160k entries (double previous)
        total_drift_length=40000,  # 40k drift length (double previous)
        num_drifts=10,  # 10 drift points (5x previous)
        
        # NEW: Enable plotting for visualization
        enable_plots=True
    )
    
    print("Large Dataset Test Configuration:")
    print(f"- Target runs: 2 (quick test)")
    print(f"- Algorithm: Naive Bayes only")
    print(f"- Scenarios: parallel_abrupt, switching_incremental")
    print(f"- Batch sizes: [1000, 2000]")
    print(f"- Dataset size: 160,000 entries (vs 80,000 previously)")
    print(f"- Total drift length: 40,000 entries (vs 20,000 previously)")
    print(f"- Number of drifts: 10 (vs 2 previously)")
    print(f"- Plotting: ENABLED (will generate visualizations)")
    print(f"- Expected time: ~10-15 minutes")
    
    print("\nThis test will verify:")
    print("- Large dataset parameters work correctly")
    print("- 10 drift points are properly distributed")
    print("- 40k drift length creates smooth transitions")
    print("- Plotting generates clear visualizations")
    print("- System handles larger datasets efficiently")
    
    # Run the test experiment
    print("\nStarting large dataset test...")
    experiment.run_multi_experiment()
    
    # Analyze results
    print("\nAnalyzing test results...")
    experiment.analyze_results()
    
    print(f"\nLarge dataset test completed!")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Results saved to: {experiment.base_output_dir}")
    print(f"\nCheck the following for visualizations:")
    print(f"- Feature plots: {experiment.base_output_dir}/comparison_results/*/feature_plots/")
    print(f"- Heatmaps: {experiment.base_output_dir}/comparison_results/*/heatmaps/")
    print(f"- Drift plots: {experiment.base_output_dir}/comparison_results/*/detected_drifts/")

if __name__ == "__main__":
    main()
