#!/usr/bin/env python3
"""
Drift Sensitivity Experiment Configuration

This module provides templated configuration for drift sensitivity experiments
that investigate how the number of drift windows affects technique performance.

Research Question: Does distributing the same amount of drift across more windows
improve drift detection technique performance?

Usage:
    from drift_sensitivity_config import DriftSensitivityConfig
    config = DriftSensitivityConfig(num_drifts=10)
"""

from dataclasses import dataclass
from typing import List
import json
from pathlib import Path
from datetime import datetime

@dataclass
class DriftSensitivityConfig:
    """
    Configuration class for drift sensitivity experiments.
    
    This class encapsulates all experimental parameters and provides
    methods for generating experiment configurations with different
    numbers of drift windows.
    """
    
    # Core experimental parameters
    num_drifts: int = 10
    target_runs: int = 30
    base_seed: int = 42
    
    # Dataset parameters (keep total drift amount constant)
    dataframe_size: int = 80000
    total_drift_length: int = 20000  # Same total as original 2-drift experiment
    
    # Synthetic dataset configuration
    features_with_drifts: List[str] = None
    synthetic_scenarios: List[str] = None
    dataset_filter: str = "synthetic_only"
    
    # Experimental design parameters
    batch_sizes: List[int] = None
    algorithms: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.features_with_drifts is None:
            self.features_with_drifts = ["feature1", "feature3", "feature5"]
        
        if self.synthetic_scenarios is None:
            self.synthetic_scenarios = ["all"]
        
        if self.batch_sizes is None:
            self.batch_sizes = [1000, 1500, 2000, 2500]
        
        if self.algorithms is None:
            self.algorithms = ["HT"]
    
    @property
    def experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        return f"drift_sensitivity_{self.num_drifts}_drifts"
    
    @property
    def experiment_type(self) -> str:
        """Return experiment type identifier."""
        return "drift_sensitivity_multi_run"
    
    @property
    def drift_length_per_window(self) -> int:
        """Calculate drift length per window to maintain total drift amount."""
        return self.total_drift_length // self.num_drifts
    
    @property
    def output_directory(self) -> str:
        """Generate output directory name."""
        return f"comparison_results/drift_sensitivity_{self.num_drifts}_drifts"
    
    def generate_seeds(self) -> List[int]:
        """
        Generate list of seeds for all experimental runs.
        
        Returns:
            List of unique seeds for each run
        """
        return [self.base_seed + i * 100 for i in range(self.target_runs)]
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "num_drifts": self.num_drifts,
            "target_runs": self.target_runs,
            "base_seed": self.base_seed,
            "dataframe_size": self.dataframe_size,
            "total_drift_length": self.total_drift_length,
            "drift_length_per_window": self.drift_length_per_window,
            "features_with_drifts": self.features_with_drifts,
            "synthetic_scenarios": self.synthetic_scenarios,
            "dataset_filter": self.dataset_filter,
            "batch_sizes": self.batch_sizes,
            "algorithms": self.algorithms,
            "output_directory": self.output_directory,
            "created_at": datetime.now().isoformat(),
            "seeds": self.generate_seeds()
        }
    
    def save_config(self, output_dir: str) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            output_dir: Directory to save configuration
            
        Returns:
            Path to saved configuration file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config_file = output_path / f"{self.experiment_id}_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        return str(config_file)
    
    @classmethod
    def load_config(cls, config_file: str) -> 'DriftSensitivityConfig':
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            DriftSensitivityConfig instance
        """
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            num_drifts=config_dict["num_drifts"],
            target_runs=config_dict["target_runs"],
            base_seed=config_dict["base_seed"],
            dataframe_size=config_dict["dataframe_size"],
            total_drift_length=config_dict["total_drift_length"],
            features_with_drifts=config_dict["features_with_drifts"],
            synthetic_scenarios=config_dict["synthetic_scenarios"],
            dataset_filter=config_dict["dataset_filter"],
            batch_sizes=config_dict["batch_sizes"],
            algorithms=config_dict["algorithms"]
        )
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.num_drifts <= 0:
            raise ValueError("num_drifts must be positive")
        
        if self.target_runs <= 0:
            raise ValueError("target_runs must be positive")
        
        if self.total_drift_length <= 0:
            raise ValueError("total_drift_length must be positive")
        
        if self.dataframe_size <= self.total_drift_length:
            raise ValueError("dataframe_size must be larger than total_drift_length")
        
        if self.drift_length_per_window <= 0:
            raise ValueError("drift_length_per_window must be positive")
        
        if not self.features_with_drifts:
            raise ValueError("features_with_drifts cannot be empty")
        
        if not self.batch_sizes:
            raise ValueError("batch_sizes cannot be empty")
        
        return True
    
    def print_summary(self):
        """Print human-readable configuration summary."""
        print(f"Drift Sensitivity Experiment Configuration")
        print(f"=" * 50)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Number of drift windows: {self.num_drifts}")
        print(f"Target runs: {self.target_runs}")
        print(f"Total drift length: {self.total_drift_length:,}")
        print(f"Drift length per window: {self.drift_length_per_window:,}")
        print(f"Dataset size: {self.dataframe_size:,}")
        print(f"Features with drifts: {self.features_with_drifts}")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Algorithms: {self.algorithms}")
        print(f"Output directory: {self.output_directory}")
        print(f"Base seed: {self.base_seed}")
        print(f"Seed range: {self.base_seed} to {self.base_seed + (self.target_runs-1)*100}")

def create_drift_sensitivity_configs() -> dict:
    """
    Create standard drift sensitivity configurations.
    
    Returns:
        Dictionary of configurations for different num_drifts values
    """
    configs = {}
    
    # Configuration for 10 drifts
    configs["10_drifts"] = DriftSensitivityConfig(num_drifts=10)
    
    # Configuration for 20 drifts (for future use)
    configs["20_drifts"] = DriftSensitivityConfig(num_drifts=20)
    
    return configs

def main():
    """
    Main function to demonstrate configuration usage.
    """
    print("Drift Sensitivity Experiment Configuration Demo")
    print("=" * 60)
    
    # Create configuration for 10 drifts
    config_10 = DriftSensitivityConfig(num_drifts=10)
    config_10.validate_config()
    config_10.print_summary()
    
    print("\n" + "=" * 60)
    
    # Create configuration for 20 drifts
    config_20 = DriftSensitivityConfig(num_drifts=20)
    config_20.validate_config()
    config_20.print_summary()
    
    print("\n" + "=" * 60)
    print("Configuration validation successful!")

if __name__ == "__main__":
    main()
