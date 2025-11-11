"""
Multi-Run Experiment Framework for Data Drift Detection

This module provides a framework for running drift detection experiments multiple times
to achieve statistical significance while maintaining reproducible datasets and efficient storage.

Key Features:
- Reproducible synthetic datasets (constant seed=42)
- Variable algorithm randomness per run
- Incremental run extension (30→50→100)
- Smart storage optimization (~2.5GB for 30 runs)
- Dataset filtering for testing
- Statistical analysis suite

Author: Generated for Empirical Analysis of Data Drift Detection
"""

import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from codes.comparisor import (
    consolidate_csv_files,
    fetch_all_drifts,
    handle_synthetic_dataset,
    prepare_datasets,
    prepare_output_path,
    run_single_experiment,
    save_drift_points_to_file,
    save_results_to_csv,
)
from codes.config import comparisons_output_dir as base_output_dir
from codes.experiment_timer import timer


class MultiRunExperimentRunner:
    """
    Main class for running multiple experiment iterations with statistical analysis.
    
    This class manages the execution of N runs of the drift detection experiment,
    ensuring reproducible datasets while varying algorithm randomness for statistical
    significance testing.
    """
    
    def __init__(
        self,
        target_runs: int = 30,
        base_seed: int = 42,
        existing_experiment_id: Optional[str] = None,
        dataset_filter: Union[str, List[str]] = "all",
        batch_sizes: Optional[List[int]] = None,
        algorithms: Optional[List[str]] = None,
    ):
        """
        Initialize the multi-run experiment runner.
        
        Args:
            target_runs: Total number of runs to execute
            base_seed: Base seed for dataset generation (kept constant)
            existing_experiment_id: ID of existing experiment to extend
            dataset_filter: Filter for datasets ("all", "synthetic_only", "quick_test", etc.)
            batch_sizes: List of batch sizes to test (default: [1000, 1500, 2000, 2500])
            algorithms: List of algorithms to test (default: ["NB"])
        """
        self.target_runs = target_runs
        self.base_seed = base_seed
        self.dataset_filter = dataset_filter
        self.batch_sizes = batch_sizes or [1000, 1500, 2000, 2500]
        self.algorithms = algorithms or ["NB"]
        
        # Experiment management
        if existing_experiment_id:
            self.experiment_id = existing_experiment_id
            self.base_output_dir = os.path.join(base_output_dir, self.experiment_id)
            self.existing_runs = self._detect_completed_runs()
            self.start_run = len(self.existing_runs) + 1
            print(f"Extending experiment {self.experiment_id}")
            print(f"Found {len(self.existing_runs)} completed runs")
        else:
            self.experiment_id = f"multi_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.base_output_dir = os.path.join(base_output_dir, self.experiment_id)
            self.existing_runs = []
            self.start_run = 1
            print(f"Starting new experiment {self.experiment_id}")
        
        # Create directory structure
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "shared_datasets"), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "analysis"), exist_ok=True)
        
        # Dataset configuration
        self.dataframe_size = 80000
        self.num_drifts = 2
        self.features_with_drifts = ["feature1", "feature3", "feature5"]
        self.drift_alignment_batch_percentages = [1.0]
        
        # Save experiment configuration
        self._save_experiment_config()
    
    def _detect_completed_runs(self) -> List[int]:
        """Detect which runs are already completed and valid."""
        completed_runs = []
        runs_dir = os.path.join(self.base_output_dir, "runs")
        
        if not os.path.exists(runs_dir):
            return completed_runs
            
        for run_dir in os.listdir(runs_dir):
            if run_dir.startswith("run_"):
                try:
                    run_id = int(run_dir.split("_")[1])
                    if self._validate_run_completeness(run_id):
                        completed_runs.append(run_id)
                except (ValueError, IndexError):
                    continue
        
        return sorted(completed_runs)
    
    def _validate_run_completeness(self, run_id: int) -> bool:
        """Check if a run completed successfully."""
        run_dir = os.path.join(self.base_output_dir, "runs", f"run_{run_id:03d}")
        metadata_file = os.path.join(run_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('run_completed', False)
            except (json.JSONDecodeError, IOError):
                return False
        return False
    
    def _save_experiment_config(self):
        """Save experiment configuration for reproducibility."""
        config = {
            "experiment_id": self.experiment_id,
            "target_runs": self.target_runs,
            "base_seed": self.base_seed,
            "dataset_filter": self.dataset_filter,
            "batch_sizes": self.batch_sizes,
            "algorithms": self.algorithms,
            "dataframe_size": self.dataframe_size,
            "num_drifts": self.num_drifts,
            "features_with_drifts": self.features_with_drifts,
            "created_at": datetime.now().isoformat(),
            "existing_runs_at_start": len(self.existing_runs),
        }
        
        config_file = os.path.join(self.base_output_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def get_filtered_datasets(self) -> List[str]:
        """Get datasets based on filter criteria."""
        all_datasets = prepare_datasets()
        
        # Handle list input first
        if isinstance(self.dataset_filter, list):
            return self.dataset_filter
        
        dataset_filters = {
            "quick_test": [
                "synthetic_dataset_with_parallel_drifts_abrupt",
                "synthetic_dataset_with_switching_drifts_incremental",
                "synthetic_dataset_no_drifts"
            ],
            "synthetic_only": [
                d for d in all_datasets if d.startswith("synthetic_")
            ],
            "real_only": [
                d for d in all_datasets if not d.startswith("synthetic_")
            ],
            "fast_real": [
                "electricity", "magic"
            ]
        }
        
        if self.dataset_filter in dataset_filters:
            return dataset_filters[self.dataset_filter]
        else:  # "all"
            return all_datasets
    
    def estimate_execution_time(self) -> Dict[str, Any]:
        """Provide time estimates based on dataset filter and run count."""
        datasets = self.get_filtered_datasets()
        
        # Time estimates in minutes per dataset per run
        time_estimates = {
            "synthetic": 2,
            "small_real": 5,
            "large_real": 15
        }
        
        total_minutes = 0
        dataset_breakdown = {"synthetic": 0, "small_real": 0, "large_real": 0}
        
        for dataset in datasets:
            if dataset.startswith("synthetic"):
                total_minutes += time_estimates["synthetic"]
                dataset_breakdown["synthetic"] += 1
            elif dataset in ["electricity", "magic"]:
                total_minutes += time_estimates["small_real"]
                dataset_breakdown["small_real"] += 1
            else:
                total_minutes += time_estimates["large_real"]
                dataset_breakdown["large_real"] += 1
        
        # Multiply by number of runs needed
        runs_needed = self.target_runs - len(self.existing_runs)
        total_minutes *= runs_needed
        
        return {
            "total_minutes": total_minutes,
            "total_hours": total_minutes / 60,
            "runs_needed": runs_needed,
            "datasets_count": len(datasets),
            "dataset_breakdown": dataset_breakdown
        }
    
    def print_experiment_status(self):
        """Print current experiment status and estimates."""
        datasets = self.get_filtered_datasets()
        estimates = self.estimate_execution_time()
        
        print(f"\n=== Experiment Status ===")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Target runs: {self.target_runs}")
        print(f"Completed runs: {len(self.existing_runs)}")
        print(f"Remaining runs: {estimates['runs_needed']}")
        print(f"Datasets: {estimates['datasets_count']} ({self.dataset_filter})")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Algorithms: {self.algorithms}")
        
        if estimates['runs_needed'] > 0:
            print(f"\n=== Time Estimates ===")
            print(f"Estimated time: {estimates['total_hours']:.1f} hours ({estimates['total_minutes']:.0f} minutes)")
            print(f"Dataset breakdown: {estimates['dataset_breakdown']}")
        else:
            print(f"\n✅ All runs completed! Ready for analysis.")
    
    def _generate_shared_datasets(self):
        """Generate all synthetic datasets once with consistent seed."""
        print("\n=== Phase 1: Generating Shared Datasets ===")
        datasets = self.get_filtered_datasets()
        synthetic_datasets = [d for d in datasets if d.startswith("synthetic_")]
        
        if not synthetic_datasets:
            print("No synthetic datasets to generate.")
            return
        
        # Set consistent seed for dataset generation
        np.random.seed(self.base_seed)
        random.seed(self.base_seed)
        
        for dataset in synthetic_datasets:
            dataset_path = os.path.join(self.base_output_dir, "shared_datasets", dataset)
            csv_path = os.path.join(dataset_path, f"{dataset}.csv")
            
            if os.path.exists(csv_path):
                print(f"Dataset {dataset} already exists, skipping...")
                continue
            
            print(f"Generating {dataset}...")
            
            # Create dataset directory
            os.makedirs(dataset_path, exist_ok=True)
            
            # Generate dataset based on type
            for batch_size in self.batch_sizes:
                if dataset == "synthetic_dataset_no_drifts":
                    scenario = "no_drifts"
                    synthetic_df, accumulated_differences, synthetic_drift_points = handle_synthetic_dataset(
                        scenario,
                        dataset,
                        self.dataframe_size,
                        batch_size,
                        features_with_drifts=[],
                    )
                else:
                    for drift_within_batch in self.drift_alignment_batch_percentages:
                        if "parallel_drifts" in dataset:
                            scenario = "parallel_abrupt" if "abrupt" in dataset else "parallel_incremental"
                        elif "switching_drifts" in dataset:
                            scenario = "switching_abrupt" if "abrupt" in dataset else "switching_incremental"
                        else:
                            continue
                        
                        synthetic_df, accumulated_differences, synthetic_drift_points = handle_synthetic_dataset(
                            scenario,
                            dataset,
                            self.dataframe_size,
                            batch_size,
                            drift_within_batch,
                            self.features_with_drifts,
                            num_drifts=self.num_drifts,
                        )
                        break  # Only generate once per dataset
                break  # Only generate once per dataset
        
        print("✅ Shared dataset generation completed!")
    
    def _create_run_symlinks(self, run_id: int):
        """Create symlinks to shared datasets for a specific run."""
        run_dir = os.path.join(self.base_output_dir, "runs", f"run_{run_id:03d}")
        datasets_dir = os.path.join(run_dir, "datasets")
        shared_datasets_dir = os.path.join(self.base_output_dir, "shared_datasets")
        
        os.makedirs(datasets_dir, exist_ok=True)
        
        # Create symlinks to shared datasets
        if os.path.exists(shared_datasets_dir):
            for dataset_name in os.listdir(shared_datasets_dir):
                shared_path = os.path.join(shared_datasets_dir, dataset_name)
                link_path = os.path.join(datasets_dir, dataset_name)
                
                if os.path.exists(shared_path) and not os.path.exists(link_path):
                    try:
                        os.symlink(shared_path, link_path)
                    except OSError:
                        # Fallback to copying if symlinks not supported
                        shutil.copytree(shared_path, link_path)
    
    def _set_run_seed(self, run_id: int):
        """Set algorithm-specific randomness for this run."""
        algorithm_seed = self.base_seed + run_id * 1000
        np.random.seed(algorithm_seed)
        random.seed(algorithm_seed)
        return algorithm_seed
    
    def _run_single_experiment_run(self, run_id: int) -> bool:
        """Execute a single experiment run."""
        print(f"\n--- Run {run_id}/{self.target_runs} ---")
        
        # Set up run directory
        run_dir = os.path.join(self.base_output_dir, "runs", f"run_{run_id:03d}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "timing"), exist_ok=True)
        
        # Create symlinks to shared datasets
        self._create_run_symlinks(run_id)
        
        # Set algorithm seed for this run
        algorithm_seed = self._set_run_seed(run_id)
        
        # Initialize run metadata
        run_metadata = {
            "run_id": run_id,
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
            json_file_paths = []
            
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
                            if dataset.startswith("synthetic_"):
                                # Handle synthetic datasets
                                if dataset == "synthetic_dataset_no_drifts":
                                    with timer.time_phase("drift_detection", dataset, batch_size):
                                        detected_drifts_dict = fetch_all_drifts(
                                            batch_size,
                                            dataset,
                                            drift_alignment_within_batch=None,
                                            plot_heatmaps=False,  # Skip plots for individual runs
                                        )
                                    
                                    with timer.time_phase("evaluation", dataset, batch_size):
                                        test_results, drift_results, num_batches = run_single_experiment(
                                            dataset,
                                            batch_size,
                                            algorithm,
                                            detected_drifts_dict=detected_drifts_dict,
                                        )
                                    
                                    save_results_to_csv(
                                        dataset,
                                        batch_size,
                                        drift_results,
                                        test_results,
                                        num_batches,
                                        csv_file_path,
                                        scenario="no_drifts",
                                        type_of_dataset="synthetic",
                                        algorithm=algorithm,
                                        run_id=run_id,
                                        algorithm_seed=algorithm_seed,
                                    )
                                else:
                                    for drift_within_batch in self.drift_alignment_batch_percentages:
                                        with timer.time_phase("drift_detection", dataset, batch_size):
                                            detected_drifts_dict = fetch_all_drifts(
                                                batch_size,
                                                dataset,
                                                drift_alignment_within_batch=drift_within_batch,
                                                plot_heatmaps=False,  # Skip plots for individual runs
                                            )
                                        
                                        # Determine scenario
                                        if "parallel_drifts" in dataset:
                                            scenario = "parallel_abrupt" if "abrupt" in dataset else "parallel_incremental"
                                        elif "switching_drifts" in dataset:
                                            scenario = "switching_abrupt" if "abrupt" in dataset else "switching_incremental"
                                        else:
                                            scenario = "unknown"
                                        
                                        with timer.time_phase("evaluation", dataset, batch_size):
                                            test_results, drift_results, num_batches = run_single_experiment(
                                                dataset,
                                                batch_size,
                                                algorithm,
                                                drift_alignment_within_batch=drift_within_batch,
                                                detected_drifts_dict=detected_drifts_dict,
                                            )
                                        
                                        save_results_to_csv(
                                            dataset,
                                            batch_size,
                                            drift_results,
                                            test_results,
                                            num_batches,
                                            csv_file_path,
                                            drift_alignment_with_batch=drift_within_batch,
                                            scenario=scenario,
                                            type_of_dataset="synthetic",
                                            algorithm=algorithm,
                                            run_id=run_id,
                                            algorithm_seed=algorithm_seed,
                                        )
                            else:
                                # Handle real datasets
                                with timer.time_phase("drift_detection", dataset, batch_size):
                                    detected_drifts_dict = fetch_all_drifts(
                                        batch_size,
                                        dataset,
                                        drift_alignment_within_batch=None,
                                        plot_heatmaps=False,  # Skip plots for individual runs
                                    )
                                
                                with timer.time_phase("evaluation", dataset, batch_size):
                                    test_results, drift_results, num_batches = run_single_experiment(
                                        dataset,
                                        batch_size,
                                        algorithm,
                                        detected_drifts_dict=detected_drifts_dict,
                                    )
                                
                                save_results_to_csv(
                                    dataset,
                                    batch_size,
                                    drift_results,
                                    test_results,
                                    num_batches,
                                    csv_file_path,
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
            run_timing_file = os.path.join(run_dir, "timing", "timing_results.json")
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
    
    def run_multi_experiment(self):
        """Execute the complete multi-run experiment."""
        print(f"\n🚀 Starting Multi-Run Experiment")
        self.print_experiment_status()
        
        # Confirm execution if significant time required
        estimates = self.estimate_execution_time()
        if estimates['runs_needed'] > 0 and estimates['total_hours'] > 1:
            response = input(f"\nThis will take approximately {estimates['total_hours']:.1f} hours. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Experiment cancelled.")
                return
        
        start_time = time.time()
        
        try:
            # Phase 1: Generate shared datasets (if needed)
            if not os.path.exists(os.path.join(self.base_output_dir, "shared_datasets")):
                self._generate_shared_datasets()
            else:
                print("✅ Shared datasets already exist, skipping generation...")
            
            # Phase 2: Execute runs
            print(f"\n=== Phase 2: Executing Runs {self.start_run}-{self.target_runs} ===")
            successful_runs = 0
            failed_runs = 0
            
            for run_id in range(self.start_run, self.target_runs + 1):
                if self._run_single_experiment_run(run_id):
                    successful_runs += 1
                else:
                    failed_runs += 1
            
            # Phase 3: Aggregate results
            print(f"\n=== Phase 3: Aggregating Results ===")
            self._aggregate_all_results()
            
            # Final summary
            total_time = time.time() - start_time
            print(f"\n🎉 Multi-Run Experiment Completed!")
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
    
    def _aggregate_all_results(self):
        """Aggregate results from all completed runs."""
        print("Aggregating results from all runs...")
        
        # Collect all consolidated results
        all_results = []
        runs_dir = os.path.join(self.base_output_dir, "runs")
        
        for run_dir in os.listdir(runs_dir):
            if run_dir.startswith("run_"):
                consolidated_file = os.path.join(runs_dir, run_dir, "results", "consolidated_results.csv")
                if os.path.exists(consolidated_file):
                    df = pd.read_csv(consolidated_file)
                    all_results.append(df)
        
        if not all_results:
            print("❌ No results found to aggregate")
            return
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        analysis_dir = os.path.join(self.base_output_dir, "analysis")
        combined_file = os.path.join(analysis_dir, "all_runs_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        
        # Generate statistical analysis
        analyzer = MultiRunAnalyzer(self.base_output_dir)
        analyzer.generate_complete_analysis()
        
        print(f"✅ Results aggregated and saved to {analysis_dir}")


class MultiRunAnalyzer:
    """Statistical analysis tools for multi-run experiment results."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.analysis_dir = os.path.join(experiment_dir, "analysis")
        self.results_file = os.path.join(self.analysis_dir, "all_runs_combined.csv")
        
        if os.path.exists(self.results_file):
            self.df = pd.read_csv(self.results_file)
        else:
            self.df = None
            print(f"❌ Results file not found: {self.results_file}")
    
    def generate_complete_analysis(self):
        """Generate complete statistical analysis suite."""
        if self.df is None:
            return
        
        print("Generating statistical analysis...")
        
        # Basic statistics
        self._generate_statistical_summary()
        
        # Technique comparison
        self._generate_technique_comparison()
        
        # Dataset analysis
        self._generate_dataset_analysis()
        
        # Batch size analysis
        self._generate_batch_size_analysis()
        
        # Confidence intervals
        self._generate_confidence_intervals()
        
        print("✅ Statistical analysis completed!")
    
    def _generate_statistical_summary(self):
        """Generate basic statistical summary."""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        summary_stats = []
        for metric in metrics:
            if metric in self.df.columns:
                stats_dict = {
                    'metric': metric,
                    'mean': self.df[metric].mean(),
                    'std': self.df[metric].std(),
                    'min': self.df[metric].min(),
                    'max': self.df[metric].max(),
                    'median': self.df[metric].median(),
                    'count': self.df[metric].count()
                }
                summary_stats.append(stats_dict)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(self.analysis_dir, "statistical_summary.csv")
        summary_df.to_csv(summary_file, index=False)
    
    def _generate_technique_comparison(self):
        """Generate technique performance comparison."""
        if 'technique' not in self.df.columns:
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        technique_stats = []
        
        for technique in self.df['technique'].unique():
            technique_data = self.df[self.df['technique'] == technique]
            
            for metric in metrics:
                if metric in self.df.columns:
                    values = technique_data[metric].dropna()
                    if len(values) > 0:
                        # Calculate confidence interval
                        confidence_level = 0.95
                        degrees_freedom = len(values) - 1
                        sample_mean = values.mean()
                        sample_standard_error = stats.sem(values)
                        confidence_interval = stats.t.interval(
                            confidence_level, degrees_freedom, sample_mean, sample_standard_error
                        )
                        
                        stats_dict = {
                            'technique': technique,
                            'metric': metric,
                            'mean': sample_mean,
                            'std': values.std(),
                            'count': len(values),
                            'ci_lower': confidence_interval[0],
                            'ci_upper': confidence_interval[1],
                            'ci_width': confidence_interval[1] - confidence_interval[0]
                        }
                        technique_stats.append(stats_dict)
        
        technique_df = pd.DataFrame(technique_stats)
        technique_file = os.path.join(self.analysis_dir, "technique_comparison.csv")
        technique_df.to_csv(technique_file, index=False)
    
    def _generate_dataset_analysis(self):
        """Generate dataset difficulty analysis."""
        if 'dataset' not in self.df.columns:
            return
        
        dataset_stats = []
        for dataset in self.df['dataset'].unique():
            dataset_data = self.df[self.df['dataset'] == dataset]
            
            stats_dict = {
                'dataset': dataset,
                'mean_accuracy': dataset_data['accuracy'].mean(),
                'std_accuracy': dataset_data['accuracy'].std(),
                'mean_f1': dataset_data['f1'].mean(),
                'std_f1': dataset_data['f1'].std(),
                'count': len(dataset_data)
            }
            dataset_stats.append(stats_dict)
        
        dataset_df = pd.DataFrame(dataset_stats)
        dataset_df = dataset_df.sort_values('mean_accuracy')  # Sort by difficulty
        
        dataset_file = os.path.join(self.analysis_dir, "dataset_analysis.csv")
        dataset_df.to_csv(dataset_file, index=False)
    
    def _generate_batch_size_analysis(self):
        """Generate batch size impact analysis."""
        if 'batch_size' not in self.df.columns:
            return
        
        batch_stats = []
        for batch_size in self.df['batch_size'].unique():
            batch_data = self.df[self.df['batch_size'] == batch_size]
            
            stats_dict = {
                'batch_size': batch_size,
                'mean_accuracy': batch_data['accuracy'].mean(),
                'std_accuracy': batch_data['accuracy'].std(),
                'mean_f1': batch_data['f1'].mean(),
                'std_f1': batch_data['f1'].std(),
                'count': len(batch_data)
            }
            batch_stats.append(stats_dict)
        
        batch_df = pd.DataFrame(batch_stats)
        batch_df = batch_df.sort_values('batch_size')
        
        batch_file = os.path.join(self.analysis_dir, "batch_size_analysis.csv")
        batch_df.to_csv(batch_file, index=False)
    
    def _generate_confidence_intervals(self):
        """Generate confidence intervals for all metrics."""
        if self.df is None:
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        ci_stats = []
        
        for metric in metrics:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                if len(values) > 0:
                    confidence_level = 0.95
                    degrees_freedom = len(values) - 1
                    sample_mean = values.mean()
                    sample_standard_error = stats.sem(values)
                    confidence_interval = stats.t.interval(
                        confidence_level, degrees_freedom, sample_mean, sample_standard_error
                    )
                    
                    ci_dict = {
                        'metric': metric,
                        'mean': sample_mean,
                        'std': values.std(),
                        'count': len(values),
                        'ci_lower': confidence_interval[0],
                        'ci_upper': confidence_interval[1],
                        'ci_width': confidence_interval[1] - confidence_interval[0],
                        'confidence_level': confidence_level
                    }
                    ci_stats.append(ci_dict)
        
        ci_df = pd.DataFrame(ci_stats)
        ci_file = os.path.join(self.analysis_dir, "confidence_intervals.csv")
        ci_df.to_csv(ci_file, index=False)


if __name__ == "__main__":
    # Example usage
    print("Multi-Run Experiment Framework")
    print("Usage examples:")
    print("1. Quick test: runner = MultiRunExperimentRunner(target_runs=5, dataset_filter='quick_test')")
    print("2. Synthetic only: runner = MultiRunExperimentRunner(target_runs=10, dataset_filter='synthetic_only')")
    print("3. Full experiment: runner = MultiRunExperimentRunner(target_runs=30, dataset_filter='all')")
    print("4. Extend experiment: runner = MultiRunExperimentRunner(target_runs=50, existing_experiment_id='multi_run_20250102_223000')")
