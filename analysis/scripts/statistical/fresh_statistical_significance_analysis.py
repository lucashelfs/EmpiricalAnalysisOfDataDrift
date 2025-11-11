#!/usr/bin/env python3
"""
Fresh Statistical Significance Analysis

Compares the fresh single-run results against the 31-run statistical baseline
to validate our findings about single-run unreliability.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class FreshStatisticalSignificanceAnalyzer:
    """
    Analyzes statistical significance of fresh single-run results against multi-run baselines.
    """
    
    def __init__(self):
        """Initialize the analyzer with paths to result directories."""
        self.baseline_dir = Path("comparison_results/statistical_relevance_published_results/analysis")
        self.fresh_results_dir = Path("comparison_results_fresh_validation")
        
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
        self.fresh_results = {}
        self.significance_results = []
        
    def load_baseline_statistics(self) -> None:
        """Load the 31-run baseline statistics from analysis files."""
        print("Loading 31-run baseline statistics...")
        
        # Load technique-specific baselines
        technique_file = self.baseline_dir / "technique_comparison.csv"
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
        
        print(f"Loaded {len(self.baseline_stats)} baseline statistics")
    
    def load_fresh_results(self) -> None:
        """Load fresh single-run results from dataset folders."""
        print("Loading fresh single-run results...")
        
        for dataset in self.synthetic_datasets:
            dataset_dir = self.fresh_results_dir / dataset
            results_file = dataset_dir / f"{dataset}_results.csv"
            
            if results_file.exists():
                df = pd.read_csv(results_file)
                self.fresh_results[dataset] = df
                print(f"  Loaded {len(df)} fresh results for {dataset}")
            else:
                print(f"  Warning: Fresh results file not found for {dataset}")
    
    def calculate_statistical_significance(self, 
                                         observed_value: float,
                                         baseline_mean: float,
                                         baseline_std: float,
                                         baseline_count: int) -> Dict:
        """
        Calculate statistical significance metrics for an observed value.
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
    
    def analyze_fresh_results(self) -> None:
        """Perform comprehensive statistical analysis on fresh results."""
        print("Performing statistical significance analysis on fresh results...")
        
        for dataset in self.synthetic_datasets:
            if dataset not in self.fresh_results:
                continue
                
            df = self.fresh_results[dataset]
            
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
        
        print(f"Analyzed {len(self.significance_results)} fresh comparisons")
    
    def generate_comparison_report(self) -> None:
        """Generate a comparison report between fresh and original analysis."""
        if not self.significance_results:
            return
        
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
        print("FRESH STATISTICAL SIGNIFICANCE ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total fresh comparisons analyzed: {summary_stats['total_comparisons']:,}")
        print(f"Significant at p < 0.05: {summary_stats['significant_05']:,} ({summary_stats['significant_05']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Significant at p < 0.01: {summary_stats['significant_01']:,} ({summary_stats['significant_01']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Significant at p < 0.001: {summary_stats['significant_001']:,} ({summary_stats['significant_001']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Outside 95% CI: {summary_stats['outside_95_ci']:,} ({summary_stats['outside_95_ci']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Outside 99% CI: {summary_stats['outside_99_ci']:,} ({summary_stats['outside_99_ci']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Mean absolute Z-score: {summary_stats['mean_absolute_z_score']:.3f}")
        print(f"Mean absolute Cohen's d: {summary_stats['mean_absolute_cohens_d']:.3f}")
        print(f"Mean absolute % deviation: {summary_stats['mean_percent_deviation']:.1f}%")
        
        # Compare with original analysis
        print("\n" + "="*80)
        print("COMPARISON WITH ORIGINAL ANALYSIS")
        print("="*80)
        
        # Load original results for comparison
        try:
            original_df = pd.read_csv('statistical_significance_detailed_results.csv')
            original_summary = {
                'total_comparisons': len(original_df),
                'significant_05': len(original_df[original_df['is_significant_05']]),
                'mean_percent_deviation': original_df['percent_deviation'].abs().mean(),
                'mean_absolute_cohens_d': original_df['cohens_d'].abs().mean()
            }
            
            print("ORIGINAL vs FRESH Results:")
            print(f"Significant results: {original_summary['significant_05']}/{original_summary['total_comparisons']} vs {summary_stats['significant_05']}/{summary_stats['total_comparisons']}")
            print(f"Mean % deviation: {original_summary['mean_percent_deviation']:.1f}% vs {summary_stats['mean_percent_deviation']:.1f}%")
            print(f"Mean effect size: {original_summary['mean_absolute_cohens_d']:.3f} vs {summary_stats['mean_absolute_cohens_d']:.3f}")
            
            # Check consistency
            if (summary_stats['significant_05'] == summary_stats['total_comparisons'] and 
                original_summary['significant_05'] == original_summary['total_comparisons']):
                print("\n✅ CONSISTENCY CONFIRMED: Both analyses show 100% significant results")
            else:
                print("\n⚠️  INCONSISTENCY DETECTED: Different significance patterns")
                
            deviation_diff = abs(summary_stats['mean_percent_deviation'] - original_summary['mean_percent_deviation'])
            if deviation_diff < 2.0:  # Within 2% is considered consistent
                print(f"✅ DEVIATION CONSISTENCY: Difference of {deviation_diff:.1f}% is within expected range")
            else:
                print(f"⚠️  DEVIATION INCONSISTENCY: Difference of {deviation_diff:.1f}% is larger than expected")
                
        except FileNotFoundError:
            print("Original analysis results not found for comparison")
    
    def save_fresh_results(self, output_file: str = "fresh_statistical_significance_results.csv") -> None:
        """Save fresh results to CSV file."""
        if not self.significance_results:
            print("No fresh results to save")
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
        print(f"Fresh detailed results saved to: {output_file}")
    
    def print_sample_comparisons(self, n_samples: int = 10) -> None:
        """Print sample comparisons to show the pattern."""
        if not self.significance_results:
            return
        
        df = pd.DataFrame(self.significance_results)
        sample_df = df.sample(min(n_samples, len(df)))
        
        print(f"\n{'='*80}")
        print(f"SAMPLE FRESH COMPARISONS (Random {len(sample_df)} results)")
        print(f"{'='*80}")
        
        for i, (_, row) in enumerate(sample_df.iterrows()):
            print(f"\n{i+1}. {row['dataset']} | {row['technique']} | Batch {row['batch_size']} | {row['metric']}")
            print(f"   Fresh: {row['observed_value']:.4f} | Expected: {row['baseline_mean']:.4f}")
            print(f"   Z-score: {row['z_score']:.3f} | p-value: {row['p_value']:.2e}")
            print(f"   Cohen's d: {row['cohens_d']:.3f} | Deviation: {row['percent_deviation']:.1f}%")
            print(f"   Significant: {row['is_significant_05']}")
    
    def run_fresh_analysis(self) -> None:
        """Run the complete fresh statistical significance analysis."""
        print("Starting Fresh Statistical Significance Analysis")
        print("="*80)
        
        # Load data
        self.load_baseline_statistics()
        self.load_fresh_results()
        
        # Perform analysis
        self.analyze_fresh_results()
        
        # Generate reports
        self.generate_comparison_report()
        
        # Save results
        self.save_fresh_results()
        
        # Print sample comparisons
        self.print_sample_comparisons()
        
        print(f"\n{'='*80}")
        print("FRESH ANALYSIS COMPLETE")
        print(f"{'='*80}")

def main():
    """Main function to run the fresh statistical significance analysis."""
    analyzer = FreshStatisticalSignificanceAnalyzer()
    analyzer.run_fresh_analysis()

if __name__ == "__main__":
    main()
