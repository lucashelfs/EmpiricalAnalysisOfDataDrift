#!/usr/bin/env python3
"""
Statistical Significance Analysis for Data Drift Detection Results

This script compares single-run experimental results against 31-run statistical baselines
to determine which results are statistically significant.

Author: Data Drift Analysis Framework
Date: December 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StatisticalSignificanceAnalyzer:
    """
    Analyzes statistical significance of single-run results against multi-run baselines.
    """
    
    def __init__(self, comparison_results_dir: str = "comparison_results"):
        """
        Initialize the analyzer with paths to result directories.
        
        Args:
            comparison_results_dir: Path to the comparison results directory
        """
        self.comparison_results_dir = Path(comparison_results_dir)
        self.baseline_dir = self.comparison_results_dir / "statistical_relevance_published_results"
        self.baseline_analysis_dir = self.baseline_dir / "analysis"
        
        # Define synthetic datasets to analyze
        self.synthetic_datasets = [
            "synthetic_dataset_with_parallel_drifts_abrupt",
            "synthetic_dataset_with_parallel_drifts_incremental", 
            "synthetic_dataset_with_switching_drifts_abrupt",
            "synthetic_dataset_with_switching_drifts_incremental",
            "synthetic_dataset_no_drifts"
        ]
        
        # Define techniques and metrics
        self.techniques = ["Base", "KS95", "KS90", "HD", "JS"]
        self.batch_sizes = [1000, 1500, 2000, 2500]
        self.metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        
        # Storage for results
        self.baseline_stats = {}
        self.single_run_results = {}
        self.significance_results = []
        
    def load_baseline_statistics(self) -> None:
        """Load the 31-run baseline statistics from analysis files."""
        print("Loading 31-run baseline statistics...")
        
        # Load technique-specific baselines
        technique_file = self.baseline_analysis_dir / "technique_comparison.csv"
        if technique_file.exists():
            technique_df = pd.read_csv(technique_file)
            for _, row in technique_df.iterrows():
                key = f"{row['technique']}_{row['metric']}"
                self.baseline_stats[key] = {
                    'mean': row['mean'],
                    'std': row['std'],
                    'count': row['count'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper']
                }
        
        # Load batch size specific baselines
        batch_file = self.baseline_analysis_dir / "batch_size_analysis.csv"
        if batch_file.exists():
            batch_df = pd.read_csv(batch_file)
            for _, row in batch_df.iterrows():
                key = f"batch_{row['batch_size']}_accuracy"
                self.baseline_stats[key] = {
                    'mean': row['mean_accuracy'],
                    'std': row['std_accuracy'],
                    'count': row['count']
                }
                key = f"batch_{row['batch_size']}_f1"
                self.baseline_stats[key] = {
                    'mean': row['mean_f1'],
                    'std': row['std_f1'],
                    'count': row['count']
                }
        
        # Load dataset-specific baselines
        dataset_file = self.baseline_analysis_dir / "dataset_analysis.csv"
        if dataset_file.exists():
            dataset_df = pd.read_csv(dataset_file)
            for _, row in dataset_df.iterrows():
                dataset_name = row['dataset']
                key = f"{dataset_name}_accuracy"
                self.baseline_stats[key] = {
                    'mean': row['mean_accuracy'],
                    'std': row['std_accuracy'],
                    'count': row['count']
                }
                key = f"{dataset_name}_f1"
                self.baseline_stats[key] = {
                    'mean': row['mean_f1'],
                    'std': row['std_f1'],
                    'count': row['count']
                }
        
        print(f"Loaded {len(self.baseline_stats)} baseline statistics")
    
    def load_single_run_results(self) -> None:
        """Load single-run results from individual dataset folders."""
        print("Loading single-run results from dataset folders...")
        
        for dataset in self.synthetic_datasets:
            dataset_dir = self.comparison_results_dir / dataset
            results_file = dataset_dir / f"{dataset}_results.csv"
            
            if results_file.exists():
                df = pd.read_csv(results_file)
                self.single_run_results[dataset] = df
                print(f"  Loaded {len(df)} results for {dataset}")
            else:
                print(f"  Warning: Results file not found for {dataset}")
    
    def calculate_statistical_significance(self, 
                                         observed_value: float,
                                         baseline_mean: float,
                                         baseline_std: float,
                                         baseline_count: int) -> Dict:
        """
        Calculate statistical significance metrics for an observed value.
        
        Args:
            observed_value: The single observed result
            baseline_mean: Mean from 31-run baseline
            baseline_std: Standard deviation from 31-run baseline
            baseline_count: Sample size of baseline
            
        Returns:
            Dictionary with statistical metrics
        """
        # Calculate standard error
        standard_error = baseline_std / np.sqrt(baseline_count)
        
        # Calculate z-score
        z_score = (observed_value - baseline_mean) / standard_error
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Calculate effect size (Cohen's d)
        cohens_d = (observed_value - baseline_mean) / baseline_std
        
        # Determine significance levels
        is_significant_05 = p_value < 0.05
        is_significant_01 = p_value < 0.01
        is_significant_001 = p_value < 0.001
        
        # Calculate confidence intervals
        ci_95_lower = baseline_mean - 1.96 * standard_error
        ci_95_upper = baseline_mean + 1.96 * standard_error
        ci_99_lower = baseline_mean - 2.576 * standard_error
        ci_99_upper = baseline_mean + 2.576 * standard_error
        
        # Check if observed value is outside confidence intervals
        outside_95_ci = observed_value < ci_95_lower or observed_value > ci_95_upper
        outside_99_ci = observed_value < ci_99_lower or observed_value > ci_99_upper
        
        return {
            'observed_value': observed_value,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'standard_error': standard_error,
            'z_score': z_score,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant_05': is_significant_05,
            'is_significant_01': is_significant_01,
            'is_significant_001': is_significant_001,
            'outside_95_ci': outside_95_ci,
            'outside_99_ci': outside_99_ci,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'ci_99_lower': ci_99_lower,
            'ci_99_upper': ci_99_upper,
            'deviation_from_mean': observed_value - baseline_mean,
            'percent_deviation': ((observed_value - baseline_mean) / baseline_mean) * 100
        }
    
    def analyze_all_results(self) -> None:
        """Perform comprehensive statistical analysis on all results."""
        print("Performing statistical significance analysis...")
        
        for dataset in self.synthetic_datasets:
            if dataset not in self.single_run_results:
                continue
                
            df = self.single_run_results[dataset]
            
            for _, row in df.iterrows():
                technique = row['technique']
                batch_size = int(row['batch_size'])
                
                for metric in self.metrics:
                    if metric not in row or pd.isna(row[metric]):
                        continue
                    
                    observed_value = float(row[metric])
                    
                    # Get baseline statistics for this technique and metric
                    baseline_key = f"{technique}_{metric}"
                    if baseline_key not in self.baseline_stats:
                        continue
                    
                    baseline = self.baseline_stats[baseline_key]
                    
                    # Calculate significance
                    significance = self.calculate_statistical_significance(
                        observed_value=observed_value,
                        baseline_mean=baseline['mean'],
                        baseline_std=baseline['std'],
                        baseline_count=baseline['count']
                    )
                    
                    # Store result
                    result = {
                        'dataset': dataset,
                        'technique': technique,
                        'batch_size': batch_size,
                        'metric': metric,
                        **significance
                    }
                    
                    self.significance_results.append(result)
        
        print(f"Analyzed {len(self.significance_results)} individual comparisons")
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of significant results."""
        if not self.significance_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.significance_results)
        
        # Create summary statistics
        summary_stats = {
            'total_comparisons': len(df),
            'significant_05': len(df[df['is_significant_05']]),
            'significant_01': len(df[df['is_significant_01']]),
            'significant_001': len(df[df['is_significant_001']]),
            'outside_95_ci': len(df[df['outside_95_ci']]),
            'outside_99_ci': len(df[df['outside_99_ci']]),
            'mean_absolute_z_score': df['z_score'].abs().mean(),
            'mean_absolute_cohens_d': df['cohens_d'].abs().mean(),
            'mean_percent_deviation': df['percent_deviation'].abs().mean()
        }
        
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total comparisons analyzed: {summary_stats['total_comparisons']:,}")
        print(f"Significant at p < 0.05: {summary_stats['significant_05']:,} ({summary_stats['significant_05']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Significant at p < 0.01: {summary_stats['significant_01']:,} ({summary_stats['significant_01']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Significant at p < 0.001: {summary_stats['significant_001']:,} ({summary_stats['significant_001']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Outside 95% CI: {summary_stats['outside_95_ci']:,} ({summary_stats['outside_95_ci']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Outside 99% CI: {summary_stats['outside_99_ci']:,} ({summary_stats['outside_99_ci']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Mean absolute Z-score: {summary_stats['mean_absolute_z_score']:.3f}")
        print(f"Mean absolute Cohen's d: {summary_stats['mean_absolute_cohens_d']:.3f}")
        print(f"Mean absolute % deviation: {summary_stats['mean_percent_deviation']:.1f}%")
        
        return df
    
    def save_detailed_results(self, output_file: str = "statistical_significance_detailed_results.csv") -> None:
        """Save detailed results to CSV file."""
        if not self.significance_results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.significance_results)
        
        # Round numerical columns for readability
        numerical_cols = ['observed_value', 'baseline_mean', 'baseline_std', 'standard_error',
                         'z_score', 'p_value', 'cohens_d', 'ci_95_lower', 'ci_95_upper',
                         'ci_99_lower', 'ci_99_upper', 'deviation_from_mean', 'percent_deviation']
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        # Sort by significance and effect size
        df = df.sort_values(['is_significant_05', 'p_value', 'cohens_d'], 
                           ascending=[False, True, False])
        
        df.to_csv(output_file, index=False)
        print(f"Detailed results saved to: {output_file}")
    
    def create_significance_matrix(self) -> pd.DataFrame:
        """Create a matrix showing significance by dataset, technique, and metric."""
        if not self.significance_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.significance_results)
        
        # Create pivot table for significance at p < 0.05
        significance_matrix = df.pivot_table(
            index=['dataset', 'technique'],
            columns='metric',
            values='is_significant_05',
            aggfunc='any',  # True if any batch size is significant
            fill_value=False
        )
        
        return significance_matrix
    
    def print_most_significant_results(self, top_n: int = 20) -> None:
        """Print the most statistically significant results."""
        if not self.significance_results:
            return
        
        df = pd.DataFrame(self.significance_results)
        
        # Filter for significant results and sort by p-value
        significant_df = df[df['is_significant_05']].copy()
        significant_df = significant_df.sort_values('p_value')
        
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, len(significant_df))} MOST SIGNIFICANT RESULTS")
        print(f"{'='*80}")
        
        for i, (_, row) in enumerate(significant_df.head(top_n).iterrows()):
            print(f"\n{i+1}. {row['dataset']} | {row['technique']} | Batch {row['batch_size']} | {row['metric']}")
            print(f"   Observed: {row['observed_value']:.4f} | Expected: {row['baseline_mean']:.4f}")
            print(f"   Z-score: {row['z_score']:.3f} | p-value: {row['p_value']:.2e}")
            print(f"   Cohen's d: {row['cohens_d']:.3f} | Deviation: {row['percent_deviation']:.1f}%")
    
    def run_complete_analysis(self) -> pd.DataFrame:
        """Run the complete statistical significance analysis."""
        print("Starting Statistical Significance Analysis")
        print("="*80)
        
        # Load data
        self.load_baseline_statistics()
        self.load_single_run_results()
        
        # Perform analysis
        self.analyze_all_results()
        
        # Generate reports
        summary_df = self.generate_summary_report()
        
        # Save results
        self.save_detailed_results()
        
        # Print significant results
        self.print_most_significant_results()
        
        # Create and save significance matrix
        significance_matrix = self.create_significance_matrix()
        if not significance_matrix.empty:
            significance_matrix.to_csv("significance_matrix.csv")
            print(f"\nSignificance matrix saved to: significance_matrix.csv")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        return summary_df

def main():
    """Main function to run the statistical significance analysis."""
    analyzer = StatisticalSignificanceAnalyzer()
    results_df = analyzer.run_complete_analysis()
    
    return analyzer, results_df

if __name__ == "__main__":
    analyzer, results = main()
