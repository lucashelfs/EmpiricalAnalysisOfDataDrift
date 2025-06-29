import time
import json
import os
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from datetime import datetime
import pandas as pd


class ExperimentTimer:
    """
    Hierarchical timing system for data drift detection experiments.
    Provides three levels of timing granularity:
    1. Total experiment time
    2. Per-dataset timing
    3. Per-technique/phase timing
    """
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.experiment_start = None
        self.current_dataset = None
        self.current_batch_size = None
        self.technique_timings = {}  # New: Store individual technique timings
        
    def start_experiment(self):
        """Start timing the entire experiment."""
        self.experiment_start = time.perf_counter()
        self.timings = {
            "experiment_metadata": {
                "start_time": datetime.now().isoformat(),
                "datasets_processed": 0,
                "total_duration": 0.0
            },
            "dataset_timings": {},
            "technique_performance": {},
            "batch_size_analysis": {},
            "phase_timings": {}
        }
        
    def end_experiment(self):
        """End timing the entire experiment."""
        if self.experiment_start:
            total_duration = time.perf_counter() - self.experiment_start
            self.timings["experiment_metadata"]["total_duration"] = total_duration
            self.timings["experiment_metadata"]["end_time"] = datetime.now().isoformat()
            return total_duration
        return 0.0
    
    def set_current_context(self, dataset: str, batch_size: int):
        """Set the current dataset and batch size context."""
        self.current_dataset = dataset
        self.current_batch_size = batch_size
        
        # Initialize dataset timing structure if not exists
        if dataset not in self.timings["dataset_timings"]:
            self.timings["dataset_timings"][dataset] = {
                "total_time": 0.0,
                "batch_timings": {},
                "phases": {}
            }
            
        # Initialize batch timing structure if not exists
        if batch_size not in self.timings["dataset_timings"][dataset]["batch_timings"]:
            self.timings["dataset_timings"][dataset]["batch_timings"][batch_size] = {
                "drift_detection": 0.0,
                "evaluation": 0.0,
                "visualization": 0.0,
                "total": 0.0
            }
    
    @contextmanager
    def time_phase(self, phase_name: str, dataset: Optional[str] = None, batch_size: Optional[int] = None):
        """Context manager for timing specific phases."""
        dataset = dataset or self.current_dataset
        batch_size = batch_size or self.current_batch_size
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._record_phase_timing(phase_name, duration, dataset, batch_size)
    
    @contextmanager
    def time_technique(self, technique_name: str, dataset: Optional[str] = None, batch_size: Optional[int] = None):
        """Context manager for timing individual drift detection techniques."""
        dataset = dataset or self.current_dataset
        batch_size = batch_size or self.current_batch_size
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._record_technique_timing(technique_name, duration, dataset, batch_size)
    
    def _record_technique_timing(self, technique_name: str, duration: float, dataset: Optional[str], batch_size: Optional[int]):
        """Record timing for a specific technique with detailed tracking."""
        # Store in technique_timings for CSV output
        key = f"{dataset}_{batch_size}_{technique_name}"
        self.technique_timings[key] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "technique": technique_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        # Also record in the existing technique_performance structure
        self.record_technique_timing(technique_name, duration, dataset, batch_size)
    
    def _record_phase_timing(self, phase_name: str, duration: float, dataset: Optional[str], batch_size: Optional[int]):
        """Record timing for a specific phase."""
        # Record in dataset timings
        if dataset and batch_size:
            if dataset in self.timings["dataset_timings"]:
                if batch_size in self.timings["dataset_timings"][dataset]["batch_timings"]:
                    self.timings["dataset_timings"][dataset]["batch_timings"][batch_size][phase_name] = duration
                    
                    # Update total for this batch
                    batch_data = self.timings["dataset_timings"][dataset]["batch_timings"][batch_size]
                    batch_data["total"] = sum(v for k, v in batch_data.items() if k != "total")
        
        # Record in phase timings for global analysis
        if phase_name not in self.timings["phase_timings"]:
            self.timings["phase_timings"][phase_name] = []
        
        self.timings["phase_timings"][phase_name].append({
            "dataset": dataset,
            "batch_size": batch_size,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_technique_timing(self, technique: str, duration: float, dataset: Optional[str] = None, batch_size: Optional[int] = None):
        """Record timing for a specific drift detection technique."""
        if technique not in self.timings["technique_performance"]:
            self.timings["technique_performance"][technique] = {
                "total_time": 0.0,
                "executions": 0,
                "avg_time": 0.0,
                "details": []
            }
        
        self.timings["technique_performance"][technique]["total_time"] += duration
        self.timings["technique_performance"][technique]["executions"] += 1
        self.timings["technique_performance"][technique]["avg_time"] = (
            self.timings["technique_performance"][technique]["total_time"] / 
            self.timings["technique_performance"][technique]["executions"]
        )
        
        self.timings["technique_performance"][technique]["details"].append({
            "dataset": dataset or self.current_dataset,
            "batch_size": batch_size or self.current_batch_size,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def update_dataset_total(self, dataset: str):
        """Update total time for a dataset after all batch sizes are processed."""
        if dataset in self.timings["dataset_timings"]:
            total_time = 0.0
            for batch_size, batch_data in self.timings["dataset_timings"][dataset]["batch_timings"].items():
                total_time += batch_data.get("total", 0.0)
            self.timings["dataset_timings"][dataset]["total_time"] = total_time
            self.timings["experiment_metadata"]["datasets_processed"] += 1
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of all timing data."""
        return self.timings.copy()
    
    def save_timing_results(self, output_dir: str):
        """Save timing results to JSON and CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed JSON
        json_path = os.path.join(output_dir, "experiment_timing_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.timings, f, indent=4)
        
        # Save CSV summary
        csv_path = os.path.join(output_dir, "timing_summary.csv")
        self._save_timing_csv(csv_path)
        
        print(f"Timing results saved to {json_path} and {csv_path}")
        return json_path, csv_path
    
    def _save_timing_csv(self, csv_path: str):
        """Save timing data as CSV for easy analysis with technique-level granularity."""
        rows = []
        
        # Individual technique timings (most granular)
        for key, timing_data in self.technique_timings.items():
            rows.append({
                "dataset": timing_data["dataset"],
                "batch_size": timing_data["batch_size"],
                "technique": timing_data["technique"],
                "phase": "drift_detection",
                "duration_seconds": timing_data["duration"],
                "type": "technique_timing"
            })
        
        # Phase timings (evaluation, visualization, etc.)
        for dataset, dataset_data in self.timings["dataset_timings"].items():
            for batch_size, batch_data in dataset_data["batch_timings"].items():
                for phase, duration in batch_data.items():
                    if phase not in ["total", "drift_detection"]:  # Skip drift_detection as it's covered by techniques
                        rows.append({
                            "dataset": dataset,
                            "batch_size": batch_size,
                            "technique": "ALL",
                            "phase": phase,
                            "duration_seconds": duration,
                            "type": "phase_timing"
                        })
                
                # Add total row
                rows.append({
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "technique": "ALL",
                    "phase": "total",
                    "duration_seconds": batch_data["total"],
                    "type": "batch_total"
                })
        
        # Technique performance averages
        for technique, tech_data in self.timings["technique_performance"].items():
            rows.append({
                "dataset": "ALL",
                "batch_size": "ALL",
                "technique": technique,
                "phase": "drift_detection",
                "duration_seconds": tech_data["avg_time"],
                "type": "technique_avg"
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    
    def print_timing_summary(self):
        """Print a formatted timing summary."""
        if not self.timings:
            print("No timing data available.")
            return
        
        print("\n" + "="*60)
        print("EXPERIMENT TIMING SUMMARY")
        print("="*60)
        
        # Experiment metadata
        metadata = self.timings["experiment_metadata"]
        print(f"Total Experiment Duration: {metadata['total_duration']:.2f} seconds")
        print(f"Datasets Processed: {metadata['datasets_processed']}")
        print(f"Start Time: {metadata.get('start_time', 'N/A')}")
        
        # Top 5 slowest datasets
        print("\nSLOWEST DATASETS:")
        dataset_times = [(name, data["total_time"]) for name, data in self.timings["dataset_timings"].items()]
        dataset_times.sort(key=lambda x: x[1], reverse=True)
        
        for i, (dataset, duration) in enumerate(dataset_times[:5], 1):
            print(f"{i}. {dataset}: {duration:.2f} seconds")
        
        # Technique performance
        print("\nTECHNIQUE PERFORMANCE:")
        for technique, data in self.timings["technique_performance"].items():
            print(f"{technique}: {data['avg_time']:.3f}s avg ({data['executions']} executions)")
        
        # Batch size analysis
        batch_analysis = {}
        for dataset_data in self.timings["dataset_timings"].values():
            for batch_size, batch_data in dataset_data["batch_timings"].items():
                if batch_size not in batch_analysis:
                    batch_analysis[batch_size] = []
                batch_analysis[batch_size].append(batch_data["total"])
        
        print("\nBATCH SIZE PERFORMANCE:")
        for batch_size in sorted(batch_analysis.keys()):
            times = batch_analysis[batch_size]
            avg_time = sum(times) / len(times) if times else 0
            print(f"Batch Size {batch_size}: {avg_time:.2f}s average")
        
        print("="*60)


# Global timer instance
timer = ExperimentTimer()


def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start_time
            func_name = func.__name__
            timer.record_technique_timing(func_name, duration)
    return wrapper
