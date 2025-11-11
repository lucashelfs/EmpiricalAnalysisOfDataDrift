#!/usr/bin/env python3
"""
30-Run Drift Sensitivity Experiment with Hoeffding Trees

This script runs a comprehensive drift sensitivity experiment using Hoeffding Trees
to investigate how the number of drift windows affects drift detection technique performance.

Research Question: Does distributing the same amount of drift across more windows
(10 vs 2) improve Hoeffding Tree drift detection performance?

Key Features:
- 30 independent runs with varying seeds
- Hoeffding Trees algorithm
- Same total drift amount distributed across more windows
- Self-contained analysis with HT baseline comparison

Usage:
    python run_30_drift_sensitivity_hoeffding_tree.py --num_drifts 10

Author: Drift Sensitivity Analysis
Date: 2025
"""

import sys
import argparse
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from codes.synthetic_multi_run_experiment import SyntheticMultiRunExperiment
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class HoeffdingTreeDriftSensitivityExperiment:
    """
    Manages Hoeffding Tree drift sensitivity experiments with configurable number of drift windows.
    """
    
    def __init__(self, num_drifts: int = 10, target_runs: int = 30, base_seed: int = 42):
        """
        Initialize the Hoeffding Tree drift sensitivity experiment.
        
        Args:
            num_drifts: Number of drift windows
            target_runs: Number of experimental runs
            base_seed: Base seed for reproducibility
        """
        self.num_drifts = num_drifts
        self.target_runs = target_runs
        self.base_seed = base_seed
        
        # Experiment configuration
        self.dataframe_size = 80000
        self.total_drift_length = 20000
        self.features_with_drifts = ["feature1", "feature3", "feature5"]
        self.synthetic_scenarios = ["all"]
        self.batch_sizes = [1000, 1500, 2000, 2500]
        self.algorithms = ["HT"]  # Hoeffding Trees
        
        # Create descriptive output directory
        self.output_dir = Path(f"comparison_results/drift_sensitivity_{num_drifts}_drifts_hoeffding_tree")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment ID
        self.experiment_id = f"drift_sensitivity_{num_drifts}_drifts_hoeffding_tree"
        
        print(f"Hoeffding Tree Drift Sensitivity Experiment Initialized")
        print(f"Output directory: {self.output_dir}")
        self.print_summary()
    
    def print_summary(self):
        """Print experiment configuration summary."""
        print(f"\nHoeffding Tree Drift Sensitivity Experiment Configuration")
        print(f"=" * 60)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Algorithm: Hoeffding Trees (HT)")
        print(f"Number of drift windows: {self.num_drifts}")
        print(f"Target runs: {self.target_runs}")
        print(f"Total drift length: {self.total_drift_length:,}")
        print(f"Drift length per window: {self.total_drift_length // self.num_drifts:,}")
        print(f"Dataset size: {self.dataframe_size:,}")
        print(f"Features with drifts: {self.features_with_drifts}")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Base seed: {self.base_seed}")
        print(f"Seed range: {self.base_seed} to {self.base_seed + (self.target_runs-1)*100}")
    
    def create_multi_run_experiment(self, enable_plots: bool = False, target_runs: int = 0) -> SyntheticMultiRunExperiment:
        """
        Create and configure the multi-run experiment instance.
        
        Args:
            enable_plots: Whether to enable plot generation
            target_runs: Override target runs (for partial experiments, 0 means use default)
        
        Returns:
            Configured SyntheticMultiRunExperiment instance
        """
        runs = target_runs if target_runs > 0 else self.target_runs
        
        # Create experiment with Hoeffding Tree parameters
        experiment = SyntheticMultiRunExperiment(
            target_runs=runs,
            base_seed=self.base_seed,
            dataframe_size=self.dataframe_size,
            total_drift_length=self.total_drift_length,
            num_drifts=self.num_drifts,
            features_with_drifts=self.features_with_drifts,
            synthetic_scenarios=self.synthetic_scenarios,
            batch_sizes=self.batch_sizes,
            algorithms=self.algorithms,
            enable_plots=enable_plots
        )
        
        return experiment
    
    def run_experiment(self) -> bool:
        """
        Execute the complete Hoeffding Tree drift sensitivity experiment.
        Two-phase approach: First run with plots, then remaining runs without plots.
        
        Returns:
            True if experiment completed successfully
        """
        print(f"\nStarting Hoeffding Tree Drift Sensitivity Experiment")
        print(f"=" * 80)
        print(f"Research Question: Does distributing drift across {self.num_drifts} windows")
        print(f"(vs 2 windows) improve Hoeffding Tree drift detection performance?")
        print(f"=" * 80)
        
        try:
            print(f"\nExperiment Configuration:")
            print(f"- Algorithm: Hoeffding Trees (HT)")
            print(f"- Number of drift windows: {self.num_drifts}")
            print(f"- Drift length per window: {self.total_drift_length // self.num_drifts:,}")
            print(f"- Total drift length: {self.total_drift_length:,}")
            print(f"- Target runs: {self.target_runs}")
            print(f"- Seed range: {self.base_seed} to {self.base_seed + (self.target_runs-1)*100}")
            print(f"- Output directory: {self.output_dir}")
            
            # Phase 1: Run first experiment with plots enabled
            print(f"\n" + "=" * 60)
            print(f"PHASE 1: Running first experiment with plots enabled")
            print(f"=" * 60)
            
            first_experiment = self.create_multi_run_experiment(enable_plots=True, target_runs=1)
            first_experiment.run_multi_experiment()
            
            # Copy first run results including plots
            self._copy_results_to_output_dir(first_experiment.base_output_dir)
            
            print(f"\nPhase 1 completed! Check plots in: {self.output_dir}")
            print(f"Please verify the drift splits are correct before continuing.")
            
            # Ask user to continue
            response = input(f"\nDo the drift splits look correct? Continue with remaining {self.target_runs-1} runs? (y/N): ")
            if response.lower() != 'y':
                print("Experiment stopped. Please check the plots and adjust parameters if needed.")
                return False
            
            # Phase 2: Run remaining experiments without plots
            print(f"\n" + "=" * 60)
            print(f"PHASE 2: Running remaining {self.target_runs-1} experiments without plots")
            print(f"=" * 60)
            
            # Create experiment for remaining runs (starting from run 2)
            remaining_experiment = SyntheticMultiRunExperiment(
                target_runs=self.target_runs,
                base_seed=self.base_seed,
                dataframe_size=self.dataframe_size,
                total_drift_length=self.total_drift_length,
                num_drifts=self.num_drifts,
                features_with_drifts=self.features_with_drifts,
                synthetic_scenarios=self.synthetic_scenarios,
                batch_sizes=self.batch_sizes,
                algorithms=self.algorithms,
                enable_plots=False,
                existing_experiment_id=first_experiment.experiment_id  # Continue from first experiment
            )
            
            remaining_experiment.run_multi_experiment()
            
            # Copy final results
            self._copy_results_to_output_dir(remaining_experiment.base_output_dir)
            
            print(f"\n" + "=" * 80)
            print(f"HOEFFDING TREE DRIFT SENSITIVITY EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"=" * 80)
            print(f"Results saved to: {self.output_dir}")
            print(f"Plots from first run available for verification")
            print(f"\nNext steps:")
            print(f"1. Run analysis for Hoeffding Tree results")
            print(f"2. Compare with Hoeffding Tree 2-drift baseline")
            print(f"3. Generate self-contained statistical reports")
            
            return True
                
        except Exception as e:
            print(f"\nError during experiment execution: {e}")
            print(f"Please check configuration and try again.")
            return False
    
    def _copy_results_to_output_dir(self, source_dir: str):
        """Copy results from the generated directory to our descriptive directory."""
        import shutil
        
        source_path = Path(source_dir)
        
        # Copy analysis results
        if (source_path / "analysis").exists():
            if (self.output_dir / "analysis").exists():
                shutil.rmtree(self.output_dir / "analysis")
            shutil.copytree(source_path / "analysis", self.output_dir / "analysis")
        
        # Copy configuration files
        for config_file in source_path.glob("*.json"):
            shutil.copy2(config_file, self.output_dir)
        
        print(f"Results copied to descriptive directory: {self.output_dir}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Hoeffding Tree drift sensitivity experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run experiment with 10 drift windows
    python run_30_drift_sensitivity_hoeffding_tree.py --num_drifts 10
    
    # Run with custom parameters
    python run_30_drift_sensitivity_hoeffding_tree.py --num_drifts 10 --target_runs 50 --base_seed 123
        """
    )
    
    parser.add_argument(
        "--num_drifts",
        type=int,
        default=10,
        help="Number of drift windows to distribute drift across (default: 10)"
    )
    
    parser.add_argument(
        "--target_runs",
        type=int,
        default=30,
        help="Number of experimental runs to execute (default: 30)"
    )
    
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for experiment reproducibility (default: 42)"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run Hoeffding Tree drift sensitivity experiment.
    """
    args = parse_arguments()
    
    print("Hoeffding Tree Drift Sensitivity Experiment")
    print("=" * 60)
    print(f"Research Question: Effect of drift window granularity on HT performance")
    print("=" * 60)
    
    # Create experiment
    experiment = HoeffdingTreeDriftSensitivityExperiment(
        num_drifts=args.num_drifts,
        target_runs=args.target_runs,
        base_seed=args.base_seed
    )
    
    # Run experiment
    success = experiment.run_experiment()
    
    if success:
        print(f"\nHoeffding Tree drift sensitivity experiment completed successfully!")
        print(f"Results available in: {experiment.output_dir}")
        
        # Show next steps
        print(f"\nRecommended next steps:")
        print(f"1. Analyze results with self-contained HT baseline comparison")
        print(f"2. Generate comprehensive HT-specific performance report")
        print(f"3. Compare findings with other algorithm results if needed")
        
        return 0
    else:
        print(f"\nExperiment failed. Please check logs and configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
