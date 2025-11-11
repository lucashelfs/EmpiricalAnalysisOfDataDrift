#!/usr/bin/env python3
"""
Drift Sensitivity Analysis

This script analyzes the results of drift sensitivity experiments to determine
how the number of drift windows affects drift detection technique performance.

Research Question: Does distributing the same amount of drift across more windows
improve drift detection technique performance?

Key Features:
- Statistical comparison between different num_drifts configurations
- Comparison with 2-drift baseline experiment
- Comprehensive technique performance analysis
- Statistical significance testing

Usage:
    python drift_sensitivity_analysis.py --num_drifts 10
    python drift_sensitivity_analysis.py --num_drifts 10 --compare_baseline

Author: Drift Sensitivity Analysis
Date: 2025
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    from experiments.drift_sensitivity_config import DriftSensitivityConfig
    from codes.multi_run_experiment import MultiRunAnalyzer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages: pip install pandas numpy scipy")
    sys.exit(1)

class DriftSensitivityAnalyzer:
    """
    Analyzes drift sensitivity experiment results to understand the impact
    of drift window granularity on detection performance.
    """
    
    def __init__(self, num_drifts: int, results_dir: Optional[str] = None):
        """
        Initialize the drift sensitivity analyzer.
        
        Args:
            num_drifts: Number of drift windows in the experiment
            results_dir: Custom results directory (optional)
        """
        self.num_drifts = num_drifts
        
        # Set results directory
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = Path(f"comparison_results/drift_sensitivity_{num_drifts}_drifts")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize analyzer
        self.analyzer = None
        self.df = None
        
        # Storage for analysis results
        self.technique_stats = {}
        self.baseline_comparison = {}
        self.statistical_results = []
        
    def _load_config(self) -> Optional[DriftSensitivityConfig]:
        """Load experiment configuration."""
        config_file = self.results_dir / f"drift_sensitivity_{self.num_drifts}_drifts_config.json"
        
        if config_file.exists():
            try:
                return DriftSensitivityConfig.load_config(str(config_file))
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
                return None
        else:
            print(f"Warning: Config file not found at {config_file}")
            return None
    
    def load_results(self) -> bool:
        """
        Load experiment results for analysis.
        
        Returns:
            True if results loaded successfully
        """
        print(f"Loading drift sensitivity results from: {self.results_dir}")
        
        if not self.results_dir.exists():
            print(f"Error: Results directory not found: {self.results_dir}")
            return False
        
        try:
            # Use MultiRunAnalyzer to load results
            self.analyzer = MultiRunAnalyzer(str(self.results_dir))
            
            if self.analyzer.df is not None:
                self.df = self.analyzer.df
                print(f"Loaded {len(self.df)} result records")
                print(f"Runs: {self.df['run_id'].nunique()}")
                print(f"Techniques: {self.df['technique'].unique().tolist()}")
                print(f"Datasets: {self.df['dataset'].unique().tolist()}")
                return True
            else:
                print("No results found in the specified directory")
                return False
                
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def calculate_technique_statistics(self) -> None:
        """Calculate comprehensive statistics for each technique."""
        if self.df is None:
            return
        
        print("Calculating technique statistics...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        techniques = self.df['technique'].unique()
        
        for technique in techniques:
            technique_data = self.df[self.df['technique'] == technique]
            
            self.technique_stats[technique] = {}
            
            for metric in metrics:
                if metric in technique_data.columns:
                    values = technique_data[metric].dropna()
                    
                    if len(values) > 0:
                        self.technique_stats[technique][metric] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'count': len(values),
                            'min': values.min(),
                            'max': values.max(),
                            'median': values.median(),
                            'ci_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
                            'ci_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values))
                        }
    
    def compare_with_baseline(self, baseline_dir: str = "comparison_results/statistical_relevance_published_results") -> None:
        """
        Compare results with 2-drift baseline experiment.
        
        Args:
            baseline_dir: Directory containing baseline results
        """
        print(f"Comparing with 2-drift baseline from: {baseline_dir}")
        
        baseline_file = Path(baseline_dir) / "analysis" / "technique_comparison.csv"
        
        if not baseline_file.exists():
            print(f"Warning: Baseline file not found at {baseline_file}")
            return
        
        try:
            baseline_df = pd.read_csv(baseline_file)
            
            # Convert baseline to same format as our technique_stats
            baseline_stats = {}
            for _, row in baseline_df.iterrows():
                technique = row['technique']
                metric = row['metric']
                
                if technique not in baseline_stats:
                    baseline_stats[technique] = {}
                
                baseline_stats[technique][metric] = {
                    'mean': row['mean'],
                    'std': row['std'],
                    'count': row['count'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper']
                }
            
            # Compare each technique and metric
            for technique in self.technique_stats:
                if technique in baseline_stats:
                    self.baseline_comparison[technique] = {}
                    
                    for metric in self.technique_stats[technique]:
                        if metric in baseline_stats[technique]:
                            current_stats = self.technique_stats[technique][metric]
                            baseline_stat = baseline_stats[technique][metric]
                            
                            # Calculate statistical comparison
                            comparison = self._statistical_comparison(
                                current_stats, baseline_stat, 
                                f"{self.num_drifts}_drifts", "2_drifts"
                            )
                            
                            self.baseline_comparison[technique][metric] = comparison
            
            print(f"Baseline comparison completed for {len(self.baseline_comparison)} techniques")
            
        except Exception as e:
            print(f"Error comparing with baseline: {e}")
    
    def _statistical_comparison(self, stats1: dict, stats2: dict, label1: str, label2: str) -> dict:
        """
        Perform statistical comparison between two sets of statistics.
        
        Args:
            stats1: Statistics for first group
            stats2: Statistics for second group
            label1: Label for first group
            label2: Label for second group
            
        Returns:
            Dictionary containing comparison results
        """
        mean1, std1, n1 = stats1['mean'], stats1['std'], stats1['count']
        mean2, std2, n2 = stats2['mean'], stats2['std'], stats2['count']
        
        # Calculate pooled standard error
        pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        # Calculate difference and effect size
        mean_difference = mean1 - mean2
        percent_difference = (mean_difference / mean2) * 100
        
        # Calculate t-statistic and p-value
        t_statistic = mean_difference / pooled_se
        df = n1 + n2 - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        
        # Calculate Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / df)
        cohens_d = mean_difference / pooled_std
        
        # Determine significance
        is_significant = p_value < 0.05
        is_highly_significant = p_value < 0.01
        is_practically_significant = abs(cohens_d) > 0.2
        
        # Determine improvement
        is_improvement = mean_difference > 0 and is_significant
        is_substantial_improvement = is_improvement and is_practically_significant
        
        return {
            f'{label1}_mean': mean1,
            f'{label2}_mean': mean2,
            'mean_difference': mean_difference,
            'percent_difference': percent_difference,
            'pooled_se': pooled_se,
            't_statistic': t_statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': is_significant,
            'is_highly_significant': is_highly_significant,
            'is_practically_significant': is_practically_significant,
            'is_improvement': is_improvement,
            'is_substantial_improvement': is_substantial_improvement
        }
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("DRIFT SENSITIVITY ANALYSIS REPORT")
        print("="*80)
        print(f"Experiment: {self.num_drifts} drift windows vs 2 drift windows (baseline)")
        print(f"Research Question: Does distributing drift across more windows improve performance?")
        print("="*80)
        
        if self.config:
            print(f"\nEXPERIMENT CONFIGURATION:")
            print(f"Number of drift windows: {self.config.num_drifts}")
            print(f"Drift length per window: {self.config.drift_length_per_window:,}")
            print(f"Total drift length: {self.config.total_drift_length:,}")
            print(f"Target runs: {self.config.target_runs}")
            print(f"Dataset size: {self.config.dataframe_size:,}")
        
        if self.df is not None:
            print(f"\nDATA SUMMARY:")
            print(f"Total records: {len(self.df):,}")
            print(f"Completed runs: {self.df['run_id'].nunique()}")
            print(f"Techniques analyzed: {len(self.df['technique'].unique())}")
            print(f"Datasets processed: {len(self.df['dataset'].unique())}")
        
        # Technique performance summary
        if self.technique_stats:
            print(f"\nTECHNIQUE PERFORMANCE SUMMARY:")
            print(f"{'Technique':<12} {'Accuracy':<12} {'F1-Score':<12} {'AUC':<12}")
            print("-" * 60)
            
            for technique in sorted(self.technique_stats.keys()):
                stats = self.technique_stats[technique]
                acc = stats.get('accuracy', {}).get('mean', 0)
                f1 = stats.get('f1', {}).get('mean', 0)
                auc = stats.get('auc', {}).get('mean', 0)
                print(f"{technique:<12} {acc:<12.4f} {f1:<12.4f} {auc:<12.4f}")
        
        # Baseline comparison results
        if self.baseline_comparison:
            print(f"\nBASELINE COMPARISON RESULTS:")
            print(f"Comparing {self.num_drifts}-drift vs 2-drift performance")
            print(f"{'Technique':<12} {'Metric':<12} {'Improvement':<12} {'P-value':<12} {'Effect Size':<12}")
            print("-" * 72)
            
            improvements = 0
            substantial_improvements = 0
            total_comparisons = 0
            
            for technique in sorted(self.baseline_comparison.keys()):
                for metric in sorted(self.baseline_comparison[technique].keys()):
                    comp = self.baseline_comparison[technique][metric]
                    total_comparisons += 1
                    
                    if comp['is_improvement']:
                        improvements += 1
                    if comp['is_substantial_improvement']:
                        substantial_improvements += 1
                    
                    improvement_str = f"{comp['percent_difference']:+.2f}%"
                    p_val_str = f"{comp['p_value']:.4f}"
                    effect_str = f"{comp['cohens_d']:+.3f}"
                    
                    print(f"{technique:<12} {metric:<12} {improvement_str:<12} {p_val_str:<12} {effect_str:<12}")
            
            print(f"\nSUMMARY STATISTICS:")
            print(f"Total comparisons: {total_comparisons}")
            print(f"Significant improvements: {improvements} ({improvements/total_comparisons*100:.1f}%)")
            print(f"Substantial improvements: {substantial_improvements} ({substantial_improvements/total_comparisons*100:.1f}%)")
        
        # Research conclusions
        print(f"\nRESEARCH CONCLUSIONS:")
        print("="*50)
        
        if self.baseline_comparison:
            # Calculate overall improvement metrics
            all_improvements = []
            all_substantial = []
            
            for technique in self.baseline_comparison:
                for metric in self.baseline_comparison[technique]:
                    comp = self.baseline_comparison[technique][metric]
                    all_improvements.append(comp['is_improvement'])
                    all_substantial.append(comp['is_substantial_improvement'])
            
            improvement_rate = sum(all_improvements) / len(all_improvements) * 100
            substantial_rate = sum(all_substantial) / len(all_substantial) * 100
            
            if substantial_rate == 0:
                print(f"FINDING: No substantial improvements found")
                print(f"Distributing drift across {self.num_drifts} windows does not substantially")
                print(f"improve drift detection performance compared to 2 windows.")
            elif substantial_rate < 25:
                print(f"FINDING: Limited improvements ({substantial_rate:.1f}%)")
                print(f"Distributing drift across {self.num_drifts} windows provides minimal")
                print(f"performance benefits over 2 windows.")
            else:
                print(f"FINDING: Notable improvements ({substantial_rate:.1f}%)")
                print(f"Distributing drift across {self.num_drifts} windows shows measurable")
                print(f"performance benefits over 2 windows.")
            
            print(f"\nIMPLICATIONS:")
            if substantial_rate == 0:
                print("- Drift window granularity may not be a critical factor")
                print("- Current drift detection methods may be insensitive to drift distribution")
                print("- Focus should be on other performance improvement strategies")
            else:
                print("- Drift window granularity affects detection performance")
                print("- Smaller, more frequent drift windows may be beneficial")
                print("- Consider optimizing drift window size for specific scenarios")
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save analysis results to files."""
        if output_dir is None:
            output_dir = self.results_dir / "analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save technique statistics
        if self.technique_stats:
            technique_file = output_dir / f"drift_sensitivity_{self.num_drifts}_technique_stats.json"
            with open(technique_file, 'w') as f:
                json.dump(self.technique_stats, f, indent=4, default=str)
            print(f"Technique statistics saved to: {technique_file}")
        
        # Save baseline comparison
        if self.baseline_comparison:
            comparison_file = output_dir / f"drift_sensitivity_{self.num_drifts}_baseline_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(self.baseline_comparison, f, indent=4, default=str)
            print(f"Baseline comparison saved to: {comparison_file}")
            
            # Save as CSV for easier analysis
            comparison_data = []
            for technique in self.baseline_comparison:
                for metric in self.baseline_comparison[technique]:
                    comp = self.baseline_comparison[technique][metric]
                    row = {
                        'technique': technique,
                        'metric': metric,
                        'num_drifts': self.num_drifts,
                        **comp
                    }
                    comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                csv_file = output_dir / f"drift_sensitivity_{self.num_drifts}_comparison.csv"
                comparison_df.to_csv(csv_file, index=False)
                print(f"Comparison CSV saved to: {csv_file}")
    
    def run_complete_analysis(self, compare_baseline: bool = True) -> bool:
        """
        Run the complete drift sensitivity analysis.
        
        Args:
            compare_baseline: Whether to compare with baseline experiment
            
        Returns:
            True if analysis completed successfully
        """
        print(f"Starting Drift Sensitivity Analysis for {self.num_drifts} drift windows")
        print("="*80)
        
        # Load results
        if not self.load_results():
            return False
        
        # Calculate statistics
        self.calculate_technique_statistics()
        
        # Compare with baseline if requested
        if compare_baseline:
            self.compare_with_baseline()
        
        # Generate report
        self.generate_comprehensive_report()
        
        # Save results
        self.save_results()
        
        print(f"\nDrift sensitivity analysis completed successfully")
        return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze drift sensitivity experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze 10-drift experiment
    python drift_sensitivity_analysis.py --num_drifts 10
    
    # Analyze with baseline comparison
    python drift_sensitivity_analysis.py --num_drifts 10 --compare_baseline
    
    # Analyze with custom results directory
    python drift_sensitivity_analysis.py --num_drifts 10 --results_dir /path/to/results
        """
    )
    
    parser.add_argument(
        "--num_drifts",
        type=int,
        required=True,
        help="Number of drift windows in the experiment to analyze"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Custom results directory (optional)"
    )
    
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Compare with 2-drift baseline experiment"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Custom output directory for analysis results"
    )
    
    return parser.parse_args()

def main():
    """Main function to run drift sensitivity analysis."""
    args = parse_arguments()
    
    print("Drift Sensitivity Analysis")
    print("="*50)
    print(f"Analyzing {args.num_drifts}-drift experiment results")
    print("="*50)
    
    # Create analyzer
    analyzer = DriftSensitivityAnalyzer(
        num_drifts=args.num_drifts,
        results_dir=args.results_dir
    )
    
    # Run analysis
    success = analyzer.run_complete_analysis(compare_baseline=args.compare_baseline)
    
    if success:
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {analyzer.results_dir}/analysis")
        return 0
    else:
        print(f"\nAnalysis failed. Please check the results directory and try again.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
