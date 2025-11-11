#!/usr/bin/env python3
"""
30-Run Drift Sensitivity Experiment

This script runs a comprehensive drift sensitivity experiment to investigate
how the number of drift windows affects drift detection technique performance.

Research Question: Does distributing the same amount of drift across more windows
(10 vs 2) improve drift detection technique performance?

Key Features:
- 30 independent runs with varying seeds
- Templated configuration for different num_drifts values
- Same total drift amount distributed across more windows
- Comprehensive statistical analysis and comparison

Usage:
    python run_30_drift_sensitivity_experiment.py --num_drifts 10
    python run_30_drift_sensitivity_experiment.py --num_drifts 20

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
    from experiments.drift_sensitivity_config import DriftSensitivityConfig
    from codes.synthetic_multi_run_experiment import SyntheticMultiRunExperiment
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class DriftSensitivityExperiment:
    """
    Manages drift sensitivity experiments with configurable number of drift windows.
    
    This class extends the existing multi-run experiment framework to support
    drift sensitivity analysis with varying numbers of drift windows while
    maintaining the same total amount of drift.
    """
    
    def __init__(self, config: DriftSensitivityConfig):
        """
        Initialize the drift sensitivity experiment.
        
        Args:
            config: DriftSensitivityConfig instance with experiment parameters
        """
        self.config = config
        self.config.validate_config()
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config_file = self.config.save_config(str(self.output_dir))
        
        print(f"Drift Sensitivity Experiment Initialized")
        print(f"Configuration saved to: {self.config_file}")
        self.config.print_summary()
    
    def create_multi_run_experiment(self) -> SyntheticMultiRunExperiment:
        """
        Create and configure the multi-run experiment instance.
        
        Returns:
            Configured SyntheticMultiRunExperiment instance
        """
        # Create experiment with modified parameters
        experiment = SyntheticMultiRunExperiment(
            target_runs=self.config.target_runs,
            base_seed=self.config.base_seed,
            dataframe_size=self.config.dataframe_size,
            total_drift_length=self.config.total_drift_length,
            num_drifts=self.config.num_drifts,  # Key parameter for sensitivity analysis
            features_with_drifts=self.config.features_with_drifts,
            synthetic_scenarios=self.config.synthetic_scenarios,
            batch_sizes=self.config.batch_sizes,
            algorithms=self.config.algorithms
        )
        
        return experiment
    
    def run_experiment(self) -> bool:
        """
        Execute the complete drift sensitivity experiment.
        
        Returns:
            True if experiment completed successfully
        """
        print(f"\nStarting Drift Sensitivity Experiment")
        print(f"=" * 80)
        print(f"Research Question: Does distributing drift across {self.config.num_drifts} windows")
        print(f"(vs 2 windows) improve drift detection performance?")
        print(f"=" * 80)
        
        try:
            # Create and run the multi-run experiment
            experiment = self.create_multi_run_experiment()
            
            print(f"\nExperiment Configuration:")
            print(f"- Number of drift windows: {self.config.num_drifts}")
            print(f"- Drift length per window: {self.config.drift_length_per_window:,}")
            print(f"- Total drift length: {self.config.total_drift_length:,}")
            print(f"- Target runs: {self.config.target_runs}")
            print(f"- Seed range: {self.config.base_seed} to {self.config.base_seed + (self.config.target_runs-1)*100}")
            print(f"- Output directory: {self.output_dir}")
            
            # Execute the experiment
            print(f"\nExecuting {self.config.target_runs} experimental runs...")
            experiment.run_multi_experiment()
            success = True
            
            if success:
                print(f"\n" + "=" * 80)
                print(f"DRIFT SENSITIVITY EXPERIMENT COMPLETED SUCCESSFULLY")
                print(f"=" * 80)
                print(f"Results saved to: {self.output_dir}")
                print(f"Configuration: {self.config_file}")
                print(f"\nNext steps:")
                print(f"1. Run drift_sensitivity_analysis.py to analyze results")
                print(f"2. Compare with 2-drift baseline experiment")
                print(f"3. Generate statistical significance reports")
                
                return True
            else:
                print(f"\nExperiment failed. Check logs for details.")
                return False
                
        except Exception as e:
            print(f"\nError during experiment execution: {e}")
            print(f"Please check configuration and try again.")
            return False
    
    def get_experiment_status(self) -> dict:
        """
        Get current experiment status and progress.
        
        Returns:
            Dictionary with experiment status information
        """
        status = {
            "experiment_id": self.config.experiment_id,
            "num_drifts": self.config.num_drifts,
            "target_runs": self.config.target_runs,
            "output_directory": str(self.output_dir),
            "config_file": self.config_file,
            "completed_runs": 0,
            "status": "not_started"
        }
        
        # Check for existing results to determine progress
        if self.output_dir.exists():
            # Look for completed run files
            run_files = list(self.output_dir.glob("run_*/"))
            status["completed_runs"] = len(run_files)
            
            if status["completed_runs"] == 0:
                status["status"] = "not_started"
            elif status["completed_runs"] < self.config.target_runs:
                status["status"] = "in_progress"
            else:
                status["status"] = "completed"
        
        return status

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run drift sensitivity experiment with configurable number of drift windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run experiment with 10 drift windows
    python run_30_drift_sensitivity_experiment.py --num_drifts 10
    
    # Run experiment with 20 drift windows
    python run_30_drift_sensitivity_experiment.py --num_drifts 20
    
    # Run with custom parameters
    python run_30_drift_sensitivity_experiment.py --num_drifts 10 --target_runs 50 --base_seed 123
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
    
    parser.add_argument(
        "--dataframe_size",
        type=int,
        default=80000,
        help="Size of synthetic datasets (default: 80000)"
    )
    
    parser.add_argument(
        "--total_drift_length",
        type=int,
        default=20000,
        help="Total amount of drift to distribute (default: 20000)"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check experiment status without running"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run drift sensitivity experiment.
    """
    args = parse_arguments()
    
    print("Drift Sensitivity Experiment")
    print("=" * 50)
    print(f"Research Question: Effect of drift window granularity on detection performance")
    print("=" * 50)
    
    # Create configuration
    config = DriftSensitivityConfig(
        num_drifts=args.num_drifts,
        target_runs=args.target_runs,
        base_seed=args.base_seed,
        dataframe_size=args.dataframe_size,
        total_drift_length=args.total_drift_length
    )
    
    # Create experiment
    experiment = DriftSensitivityExperiment(config)
    
    # Check status if requested
    if args.status:
        status = experiment.get_experiment_status()
        print(f"\nExperiment Status:")
        print(f"- ID: {status['experiment_id']}")
        print(f"- Status: {status['status']}")
        print(f"- Completed runs: {status['completed_runs']}/{status['target_runs']}")
        print(f"- Output directory: {status['output_directory']}")
        return
    
    # Run experiment
    success = experiment.run_experiment()
    
    if success:
        print(f"\n🎉 Drift sensitivity experiment completed successfully!")
        print(f"Results available in: {experiment.output_dir}")
        
        # Show next steps
        print(f"\nRecommended next steps:")
        print(f"1. Run analysis: python experiments/drift_sensitivity_analysis.py --num_drifts {args.num_drifts}")
        print(f"2. Compare with baseline: Compare against 2-drift experiment results")
        print(f"3. Generate report: Create comprehensive comparison report")
        
        return 0
    else:
        print(f"\n❌ Experiment failed. Please check logs and configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
