"""
Synthetic Multi-Run Experiment Framework

This module provides a specialized framework for running multiple drift detection experiments
on synthetic datasets with dataset variation to achieve statistical significance.

Key Features:
- Different synthetic dataset per run (creates variance)
- Flexible synthetic dataset parameters (size, drift length, num drifts)
- Focused on synthetic datasets only
- Built-in statistical analysis
- Easy-to-use interface

Author: Generated for Empirical Analysis of Data Drift Detection
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from codes.multi_run_experiment import MultiRunExperimentRunner, MultiRunAnalyzer
from codes.comparisor import (
    consolidate_csv_files,
    fetch_all_drifts,
    run_single_experiment,
    save_results_to_csv,
)
from codes.drift_generation import (
    generate_synthetic_dataset_with_drifts,
    save_synthetic_dataset,
)
from codes.plots import (
    plot_all_features,
    plot_drift_points,
    plot_feature_and_its_variations,
)
from codes.experiment_timer import timer


class SyntheticMultiRunExperiment(MultiRunExperimentRunner):
    """
    Specialized multi-run experiment framework for synthetic datasets.
    
    This class generates different synthetic dataset realizations per run while maintaining
    the same base structure. Each run uses a different seed for dataset generation,
    creating statistical variance needed for meaningful analysis.
    
    Key Features:
    - Different dataset seed per run (creates variance)
    - Flexible synthetic dataset parameters
    - All synthetic scenarios supported
    - Same comprehensive statistical analysis
    - Easy-to-use interface
    """
    
    def __init__(
        self,
        target_runs: int = 30,
        base_seed: int = 42,
        
        # Synthetic dataset parameters (flexible)
        dataframe_size: int = 80000,
        total_drift_length: int = 20000,
        num_drifts: int = 2,
        features_with_drifts: List[str] = None,
        
        # Scenario selection
        synthetic_scenarios: List[str] = None,
        
        # Standard parameters
        batch_sizes: List[int] = None,
        algorithms: List[str] = None,
        
        # Plotting control
        enable_plots: bool = False,
        
        # Existing experiment extension
        existing_experiment_id: Optional[str] = None,
    ):
        """
        Initialize the synthetic multi-run experiment.
        
        Args:
            target_runs: Total number of runs to execute
            base_seed: Base seed for reproducibility
            dataframe_size: Size of synthetic datasets (default: 80k)
            total_drift_length: Total amount of drift across all windows (default: 20k)
            num_drifts: Number of drift windows (default: 2)
            features_with_drifts: Features to apply drifts to
            synthetic_scenarios: Scenarios to test (default: all)
            batch_sizes: Batch sizes to test
            algorithms: Algorithms to test
            existing_experiment_id: ID of existing experiment to extend
        """
        
        # Set defaults for synthetic-specific parameters
        if features_with_drifts is None:
            features_with_drifts = ["feature1", "feature3", "feature5"]
        if synthetic_scenarios is None:
            synthetic_scenarios = ["all"]
        if batch_sizes is None:
            batch_sizes = [1000, 2000]
        if algorithms is None:
            algorithms = ["NB", "HT"]
        
        # Store synthetic-specific parameters
        self.dataframe_size = dataframe_size
        self.total_drift_length = total_drift_length
        self.num_drifts = num_drifts
        self.features_with_drifts = features_with_drifts
        self.synthetic_scenarios = synthetic_scenarios
        self.enable_plots = enable_plots
        
        # Determine dataset filter based on scenarios
        if "all" in synthetic_scenarios:
            dataset_filter = "synthetic_only"
        else:
            # Map scenario names to dataset names
            scenario_to_dataset = {
                "parallel_abrupt": "synthetic_dataset_with_parallel_drifts_abrupt",
                "parallel_incremental": "synthetic_dataset_with_parallel_drifts_incremental", 
                "switching_abrupt": "synthetic_dataset_with_switching_drifts_abrupt",
                "switching_incremental": "synthetic_dataset_with_switching_drifts_incremental",
                "no_drifts": "synthetic_dataset_no_drifts"
            }
            dataset_filter = [scenario_to_dataset[scenario] for scenario in synthetic_scenarios if scenario in scenario_to_dataset]
        
        # Initialize parent class
        super().__init__(
            target_runs=target_runs,
            base_seed=base_seed,
            existing_experiment_id=existing_experiment_id,
            dataset_filter=dataset_filter,
            batch_sizes=batch_sizes,
            algorithms=algorithms,
        )
        
        # Override parent's hardcoded values with our flexible parameters
        self.dataframe_size = dataframe_size
        self.num_drifts = num_drifts
        self.features_with_drifts = features_with_drifts
        
        # Update experiment configuration
        self._save_synthetic_experiment_config()
    
    def _save_synthetic_experiment_config(self):
        """Save synthetic-specific experiment configuration."""
        config = {
            "experiment_id": self.experiment_id,
            "experiment_type": "synthetic_multi_run",
            "target_runs": self.target_runs,
            "base_seed": self.base_seed,
            
            # Synthetic-specific parameters
            "dataframe_size": self.dataframe_size,
            "total_drift_length": self.total_drift_length,
            "num_drifts": self.num_drifts,
            "features_with_drifts": self.features_with_drifts,
            "synthetic_scenarios": self.synthetic_scenarios,
            
            # Standard parameters
            "dataset_filter": self.dataset_filter,
            "batch_sizes": self.batch_sizes,
            "algorithms": self.algorithms,
            
            "created_at": datetime.now().isoformat(),
            "existing_runs_at_start": len(self.existing_runs),
        }
        
        config_file = os.path.join(self.base_output_dir, "synthetic_experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def _set_run_seeds(self, run_id: int) -> Tuple[int, int]:
        """Set different seeds for dataset and algorithm per run."""
        # Different dataset per run (creates variance)
        dataset_seed = self.base_seed + run_id * 100
        
        # Algorithm seed (for future stochastic algorithms)
        algorithm_seed = self.base_seed + run_id * 1000
        
        return dataset_seed, algorithm_seed
    
    def _generate_synthetic_dataset_for_run(
        self, 
        run_id: int, 
        scenario: str, 
        dataset_name: str, 
        batch_size: int,
        drift_within_batch: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Generate a synthetic dataset for a specific run with varied seed."""
        
        dataset_seed, _ = self._set_run_seeds(run_id)
        
        # Set the seed for this specific dataset generation
        np.random.seed(dataset_seed)
        random.seed(dataset_seed)
        
        print(f"  Generating {dataset_name} for run {run_id} with seed {dataset_seed}")
        
        # Generate dataset with flexible parameters
        (
            synthetic_df,
            drift_points,
            drift_info,
            accumulated_differences,
            features_with_drifts,
        ) = generate_synthetic_dataset_with_drifts(
            dataframe_size=self.dataframe_size,
            features_with_drifts=self.features_with_drifts if scenario != "no_drifts" else [],
            batch_size=batch_size,
            drift_within_batch=drift_within_batch,
            num_features=5,
            loc=10,
            scale=1,
            seed=dataset_seed,  # Use varied seed
            scenario=scenario,
            num_drifts=self.num_drifts,
        )
        
        return synthetic_df, drift_points, accumulated_differences
    
    def _run_single_experiment_run(self, run_id: int) -> bool:
        """Execute a single experiment run with dataset variation."""
        print(f"\n--- Run {run_id}/{self.target_runs} ---")
        
        # Set up run directory
        run_dir = os.path.join(self.base_output_dir, "runs", f"run_{run_id:03d}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "timing"), exist_ok=True)
        
        # Set seeds for this run
        dataset_seed, algorithm_seed = self._set_run_seeds(run_id)
        
        # Initialize run metadata
        run_metadata = {
            "run_id": run_id,
            "dataset_seed": dataset_seed,
            "algorithm_seed": algorithm_seed,
            "start_time": datetime.now().isoformat(),
            "run_completed": False,
            "datasets_processed": [],
            "errors": []
        }
        
        try:
            # Start timing for this run
            timer.start_experiment()
            
            datasets = self.get_filtered_datasets()
            csv_file_paths = []
            
            for dataset in datasets:
                print(f"Processing {dataset}...")
                
                # Prepare output path (use run-specific directory)
                output_path = os.path.join(run_dir, "results", dataset)
                os.makedirs(output_path, exist_ok=True)
                csv_file_path = os.path.join(output_path, f"{dataset}_results.csv")
                
                # Remove existing CSV for this run
                if os.path.exists(csv_file_path):
                    os.remove(csv_file_path)
                
                for algorithm in self.algorithms:
                    for batch_size in self.batch_sizes:
                        timer.set_current_context(dataset, batch_size)
                        
                        try:
                            # Determine scenario from dataset name
                            if dataset == "synthetic_dataset_no_drifts":
                                scenario = "no_drifts"
                                drift_within_batch = None
                            elif "parallel_drifts_abrupt" in dataset:
                                scenario = "parallel_abrupt"
                                drift_within_batch = 1.0
                            elif "parallel_drifts_incremental" in dataset:
                                scenario = "parallel_incremental"
                                drift_within_batch = 1.0
                            elif "switching_drifts_abrupt" in dataset:
                                scenario = "switching_abrupt"
                                drift_within_batch = 1.0
                            elif "switching_drifts_incremental" in dataset:
                                scenario = "switching_incremental"
                                drift_within_batch = 1.0
                            else:
                                continue  # Skip non-synthetic datasets
                            
                            # Generate dataset for this specific run
                            synthetic_df, synthetic_drift_points, accumulated_differences = self._generate_synthetic_dataset_for_run(
                                run_id, scenario, dataset, batch_size, drift_within_batch or 1.0
                            )
                            
                            # Save dataset for this run
                            run_dataset_dir = os.path.join(run_dir, "datasets", dataset)
                            os.makedirs(run_dataset_dir, exist_ok=True)
                            dataset_file = os.path.join(run_dataset_dir, f"{dataset}.csv")
                            synthetic_df.to_csv(dataset_file, index=False)
                            
                            # CRITICAL: Copy dataset to expected global location for load_and_prepare_dataset()
                            from codes.config import comparisons_output_dir
                            global_dataset_dir = os.path.join(comparisons_output_dir, dataset)
                            os.makedirs(global_dataset_dir, exist_ok=True)
                            global_dataset_file = os.path.join(global_dataset_dir, f"{dataset}.csv")
                            synthetic_df.to_csv(global_dataset_file, index=False)
                            
                            # Create custom output directory for plots
                            plots_output_dir = os.path.join(self.base_output_dir, "plots") if self.enable_plots else None
                            
                            # Time drift detection phase
                            with timer.time_phase("drift_detection", dataset, batch_size):
                                detected_drifts_dict = fetch_all_drifts(
                                    batch_size,
                                    dataset,
                                    drift_alignment_within_batch=drift_within_batch,
                                    plot_heatmaps=self.enable_plots,  # Use plotting parameter
                                    custom_output_dir=plots_output_dir,
                                )
                            
                            # Time evaluation phase
                            with timer.time_phase("evaluation", dataset, batch_size):
                                test_results, drift_results, num_batches = run_single_experiment(
                                    dataset,
                                    batch_size,
                                    algorithm,
                                    drift_alignment_within_batch=drift_within_batch,
                                    detected_drifts_dict=detected_drifts_dict,
                                )
                            
                            # Generate additional plots if enabled
                            if self.enable_plots:
                                try:
                                    # Plot all features with drift points (only if drift points exist)
                                    if synthetic_drift_points and scenario != "no_drifts":
                                        plot_all_features(
                                            synthetic_df,
                                            dataset,
                                            synthetic_drift_points,
                                            suffix=f"_{scenario}_drifts_{batch_size}_{drift_within_batch or 1.0}_run_{run_id}",
                                            custom_output_dir=plots_output_dir,
                                        )
                                        print(f"✅ Feature plots generated for {dataset}")
                                    
                                    # Plot drift points comparison (only if we have detected drifts)
                                    if drift_results and any(len(v) > 0 for v in drift_results.values() if isinstance(v, list)):
                                        plot_drift_points(
                                            drift_results,
                                            dataset,
                                            batch_size,
                                            synthetic_drift_points=synthetic_drift_points,
                                            drift_alignment_within_batch=drift_within_batch or 1.0,
                                            max_index=int(self.dataframe_size / batch_size),
                                            custom_output_dir=plots_output_dir,
                                        )
                                        print(f"✅ Drift point plots generated for {dataset}")
                                    
                                    # Skip the problematic feature variations plot for now
                                    # This function has issues with empty image lists
                                    print(f"✅ Basic plots generated successfully for {dataset}")
                                    
                                except Exception as plot_error:
                                    print(f"⚠️ Plotting failed for {dataset}: {str(plot_error)}")
                                    print("Continuing experiment without plots...")
                            
                            # Save results
                            save_results_to_csv(
                                dataset,
                                batch_size,
                                drift_results,
                                test_results,
                                num_batches,
                                csv_file_path,
                                drift_alignment_with_batch=drift_within_batch if drift_within_batch is not None else 1.0,
                                scenario=scenario,
                                type_of_dataset="synthetic",
                                algorithm=algorithm,
                                run_id=run_id,
                                algorithm_seed=algorithm_seed,
                            )
                        
                        except Exception as e:
                            error_msg = f"Error processing {dataset} with batch_size {batch_size}: {str(e)}"
                            print(f"❌ {error_msg}")
                            run_metadata["errors"].append(error_msg)
                            continue
                
                csv_file_paths.append(csv_file_path)
                run_metadata["datasets_processed"].append(dataset)
            
            # Consolidate results for this run
            if csv_file_paths:
                consolidated_csv = os.path.join(run_dir, "results", "consolidated_results.csv")
                consolidate_csv_files(csv_file_paths, consolidated_csv)
            
            # Save timing results for this run
            timer.save_timing_results(os.path.join(run_dir, "timing"))
            
            # Mark run as completed
            run_metadata["run_completed"] = True
            run_metadata["end_time"] = datetime.now().isoformat()
            run_metadata["total_datasets"] = len(datasets)
            
            print(f"✅ Run {run_id} completed successfully!")
            return True
            
        except Exception as e:
            error_msg = f"Critical error in run {run_id}: {str(e)}"
            print(f"❌ {error_msg}")
            run_metadata["errors"].append(error_msg)
            run_metadata["run_completed"] = False
            run_metadata["end_time"] = datetime.now().isoformat()
            return False
        
        finally:
            # Always save run metadata
            metadata_file = os.path.join(run_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(run_metadata, f, indent=4)
    
    def print_experiment_status(self):
        """Print current experiment status with synthetic-specific information."""
        datasets = self.get_filtered_datasets()
        estimates = self.estimate_execution_time()
        
        print(f"\n=== Synthetic Multi-Run Experiment Status ===")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Target runs: {self.target_runs}")
        print(f"Completed runs: {len(self.existing_runs)}")
        print(f"Remaining runs: {estimates['runs_needed']}")
        
        print(f"\n=== Synthetic Dataset Configuration ===")
        print(f"Dataset size: {self.dataframe_size:,} entries")
        print(f"Total drift length: {self.total_drift_length:,} entries")
        print(f"Number of drift windows: {self.num_drifts}")
        print(f"Features with drifts: {self.features_with_drifts}")
        print(f"Scenarios: {len(datasets)} ({self.synthetic_scenarios})")
        
        print(f"\n=== Experiment Parameters ===")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Algorithms: {self.algorithms}")
        
        if estimates['runs_needed'] > 0:
            print(f"\n=== Time Estimates ===")
            print(f"Estimated time: {estimates['total_hours']:.1f} hours ({estimates['total_minutes']:.0f} minutes)")
            print(f"Dataset breakdown: {estimates['dataset_breakdown']}")
        else:
            print(f"\n✅ All runs completed! Ready for analysis.")
    
    def run_multi_experiment(self):
        """Execute the complete synthetic multi-run experiment."""
        print(f"\n🧬 Starting Synthetic Multi-Run Experiment")
        self.print_experiment_status()
        
        # Confirm execution if significant time required
        estimates = self.estimate_execution_time()
        if estimates['runs_needed'] > 0 and estimates['total_hours'] > 0.5:
            response = input(f"\nThis will take approximately {estimates['total_hours']:.1f} hours. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Experiment cancelled.")
                return
        
        start_time = time.time()
        
        try:
            # Note: No shared dataset generation - each run generates its own datasets
            print("✅ Using per-run dataset generation (no shared datasets needed)")
            
            # Execute runs
            print(f"\n=== Executing Runs {self.start_run}-{self.target_runs} ===")
            successful_runs = 0
            failed_runs = 0
            
            for run_id in range(self.start_run, self.target_runs + 1):
                if self._run_single_experiment_run(run_id):
                    successful_runs += 1
                else:
                    failed_runs += 1
            
            # Aggregate results
            print(f"\n=== Aggregating Results ===")
            self._aggregate_all_results()
            
            # Final summary
            total_time = time.time() - start_time
            print(f"\n🎉 Synthetic Multi-Run Experiment Completed!")
            print(f"Total time: {total_time/3600:.2f} hours")
            print(f"Successful runs: {successful_runs}")
            print(f"Failed runs: {failed_runs}")
            print(f"Results saved to: {self.base_output_dir}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Experiment interrupted by user")
            print(f"Partial results saved to: {self.base_output_dir}")
            print(f"You can resume by extending this experiment ID: {self.experiment_id}")
        except Exception as e:
            print(f"\n❌ Experiment failed with error: {str(e)}")
            print(f"Partial results saved to: {self.base_output_dir}")
    
    def analyze_results(self):
        """Generate comprehensive statistical analysis of the synthetic multi-run results."""
        print(f"\n📊 Analyzing Synthetic Multi-Run Results...")
        
        analyzer = MultiRunAnalyzer(self.base_output_dir)
        
        if analyzer.df is not None:
            print(f"✅ Loaded {len(analyzer.df)} result records")
            print(f"📊 Runs: {analyzer.df['run_id'].nunique()}")
            print(f"🔬 Techniques: {analyzer.df['technique'].unique().tolist()}")
            print(f"📁 Datasets: {analyzer.df['dataset'].unique().tolist()}")
            
            # Generate statistical analysis
            analyzer.generate_complete_analysis()
            
            # Print summary
            print(f"\n📈 STATISTICAL SUMMARY:")
            print(f"Overall Accuracy: {analyzer.df['accuracy'].mean():.3f} ± {analyzer.df['accuracy'].std():.3f}")
            print(f"Overall F1-Score: {analyzer.df['f1'].mean():.3f} ± {analyzer.df['f1'].std():.3f}")
            
            # Technique comparison
            print(f"\n🔬 TECHNIQUE COMPARISON:")
            for technique in analyzer.df['technique'].unique():
                technique_data = analyzer.df[analyzer.df['technique'] == technique]
                acc_mean = technique_data['accuracy'].mean()
                acc_std = technique_data['accuracy'].std()
                f1_mean = technique_data['f1'].mean()
                f1_std = technique_data['f1'].std()
                print(f"    {technique}: Acc={acc_mean:.3f}±{acc_std:.3f}, F1={f1_mean:.3f}±{f1_std:.3f}")
            
            print(f"\n📂 Analysis files saved to: {self.base_output_dir}/analysis")
            
        else:
            print("❌ No results found to analyze")


if __name__ == "__main__":
    # Example usage
    print("Synthetic Multi-Run Experiment Framework")
    print("Usage examples:")
    print("1. Quick test: experiment = SyntheticMultiRunExperiment(target_runs=5)")
    print("2. Custom parameters: experiment = SyntheticMultiRunExperiment(target_runs=10, dataframe_size=100000)")
    print("3. Specific scenarios: experiment = SyntheticMultiRunExperiment(synthetic_scenarios=['parallel_abrupt', 'no_drifts'])")
    print("4. Run experiment: experiment.run_multi_experiment()")
    print("5. Analyze results: experiment.analyze_results()")
