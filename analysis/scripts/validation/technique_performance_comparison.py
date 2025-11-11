#!/usr/bin/env python3
"""
Technique Performance Comparison Analysis

Analyzes the 31-run multi-run experiment results to determine if drift detection
techniques (KS95, KS90, HD, JS) outperform the base classifier in any scenarios.
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from analysis.config import (
    VALIDATION_RESULTS,
    COMPARISON_RESULTS,
    ensure_results_dirs
)

class TechniquePerformanceAnalyzer:
    """
    Analyzes technique performance against base classifier using 31-run statistical data.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.results_dir = COMPARISON_RESULTS / "statistical_relevance_published_results"
        self.analysis_dir = self.results_dir / "analysis"
        
        # Define techniques and metrics
        self.drift_techniques = ["KS95", "KS90", "HD", "JS"]
        self.all_techniques = ["Base"] + self.drift_techniques
        self.metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        
        # Storage for results
        self.technique_stats = {}
        self.dataset_results = {}
        self.comparison_results = []
        self.summary_stats = {}
        
    def load_technique_statistics(self) -> None:
        """Load technique-level statistics from the 31-run analysis."""
        print("Loading 31-run technique statistics...")
        
        technique_file = self.analysis_dir / "technique_comparison.csv"
        if technique_file.exists():
            df = pd.read_csv(technique_file)
            
            for _, row in df.iterrows():
                key = f"{row['technique']}_{row['metric']}"
                self.technique_stats[key] = {
                    'technique': row['technique'],
                    'metric': row['metric'],
                    'mean': row['mean'],
                    'std': row['std'],
                    'count': row['count'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper']
                }
            
            print(f"Loaded statistics for {len(df)} technique-metric combinations")
        else:
            print(f"Warning: Technique statistics file not found at {technique_file}")
    
    def load_dataset_results(self) -> None:
        """Load dataset-specific results from the 31-run analysis."""
        print("Loading dataset-specific results...")
        
        dataset_file = self.analysis_dir / "dataset_comparison.csv"
        if dataset_file.exists():
            df = pd.read_csv(dataset_file)
            
            for _, row in df.iterrows():
                dataset = row['dataset']
                if dataset not in self.dataset_results:
                    self.dataset_results[dataset] = {}
                
                key = f"{row['technique']}_{row['metric']}"
                self.dataset_results[dataset][key] = {
                    'technique': row['technique'],
                    'metric': row['metric'],
                    'mean': row['mean'],
                    'std': row['std'],
                    'count': row['count'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper']
                }
            
            print(f"Loaded results for {len(self.dataset_results)} datasets")
        else:
            print(f"Warning: Dataset results file not found at {dataset_file}")
    
    def compare_techniques_vs_base(self) -> None:
        """Compare each drift detection technique against the base classifier."""
        print("Comparing drift detection techniques vs base classifier...")
        
        for metric in self.metrics:
            base_key = f"Base_{metric}"
            if base_key not in self.technique_stats:
                continue
            
            base_stats = self.technique_stats[base_key]
            
            for technique in self.drift_techniques:
                technique_key = f"{technique}_{metric}"
                if technique_key not in self.technique_stats:
                    continue
                
                tech_stats = self.technique_stats[technique_key]
                
                # Perform statistical comparison
                comparison = self.statistical_comparison(base_stats, tech_stats)
                comparison.update({
                    'metric': metric,
                    'base_technique': 'Base',
                    'drift_technique': technique,
                    'comparison_type': 'overall'
                })
                
                self.comparison_results.append(comparison)
    
    def compare_by_dataset(self) -> None:
        """Compare techniques by individual dataset."""
        print("Comparing techniques by dataset...")
        
        for dataset in self.dataset_results:
            dataset_data = self.dataset_results[dataset]
            
            for metric in self.metrics:
                base_key = f"Base_{metric}"
                if base_key not in dataset_data:
                    continue
                
                base_stats = dataset_data[base_key]
                
                for technique in self.drift_techniques:
                    technique_key = f"{technique}_{metric}"
                    if technique_key not in dataset_data:
                        continue
                    
                    tech_stats = dataset_data[technique_key]
                    
                    # Perform statistical comparison
                    comparison = self.statistical_comparison(base_stats, tech_stats)
                    comparison.update({
                        'metric': metric,
                        'base_technique': 'Base',
                        'drift_technique': technique,
                        'comparison_type': 'dataset',
                        'dataset': dataset
                    })
                    
                    self.comparison_results.append(comparison)
    
    def statistical_comparison(self, base_stats: dict, tech_stats: dict) -> dict:
        """
        Perform statistical comparison between base and technique.
        """
        base_mean = base_stats['mean']
        base_std = base_stats['std']
        base_count = base_stats['count']
        
        tech_mean = tech_stats['mean']
        tech_std = tech_stats['std']
        tech_count = tech_stats['count']
        
        # Calculate pooled standard error for difference of means
        pooled_se = np.sqrt((base_std**2 / base_count) + (tech_std**2 / tech_count))
        
        # Calculate difference and effect size
        mean_difference = tech_mean - base_mean
        percent_difference = (mean_difference / base_mean) * 100
        
        # Calculate t-statistic and p-value for two-sample t-test
        t_statistic = mean_difference / pooled_se
        df = base_count + tech_count - 2  # degrees of freedom
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        
        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((base_count - 1) * base_std**2 + (tech_count - 1) * tech_std**2) / df)
        cohens_d = mean_difference / pooled_std
        
        # Determine significance and practical significance
        is_significant = p_value < 0.05
        is_highly_significant = p_value < 0.01
        is_practically_significant = abs(cohens_d) > 0.2  # Small effect size threshold
        
        # Determine if technique is better
        technique_better = mean_difference > 0 and is_significant
        technique_substantially_better = technique_better and is_practically_significant
        
        return {
            'base_mean': base_mean,
            'tech_mean': tech_mean,
            'mean_difference': mean_difference,
            'percent_difference': percent_difference,
            'pooled_se': pooled_se,
            't_statistic': t_statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': is_significant,
            'is_highly_significant': is_highly_significant,
            'is_practically_significant': is_practically_significant,
            'technique_better': technique_better,
            'technique_substantially_better': technique_substantially_better,
            'base_ci_lower': base_stats['ci_lower'],
            'base_ci_upper': base_stats['ci_upper'],
            'tech_ci_lower': tech_stats['ci_lower'],
            'tech_ci_upper': tech_stats['ci_upper']
        }
    
    def generate_summary_statistics(self) -> None:
        """Generate summary statistics from all comparisons."""
        if not self.comparison_results:
            return
        
        df = pd.DataFrame(self.comparison_results)
        
        # Overall summary
        total_comparisons = len(df)
        significant_improvements = len(df[df['technique_better']])
        substantial_improvements = len(df[df['technique_substantially_better']])
        
        # By technique
        technique_summary = df.groupby('drift_technique').agg({
            'technique_better': 'sum',
            'technique_substantially_better': 'sum',
            'mean_difference': 'mean',
            'percent_difference': 'mean',
            'cohens_d': 'mean'
        }).round(4)
        
        # By metric
        metric_summary = df.groupby('metric').agg({
            'technique_better': 'sum',
            'technique_substantially_better': 'sum',
            'mean_difference': 'mean',
            'percent_difference': 'mean',
            'cohens_d': 'mean'
        }).round(4)
        
        # Best cases (where techniques significantly outperform base)
        best_cases = df[df['technique_substantially_better']].sort_values(
            'percent_difference', ascending=False
        )
        
        self.summary_stats = {
            'total_comparisons': total_comparisons,
            'significant_improvements': significant_improvements,
            'substantial_improvements': substantial_improvements,
            'improvement_rate': significant_improvements / total_comparisons * 100,
            'substantial_improvement_rate': substantial_improvements / total_comparisons * 100,
            'technique_summary': technique_summary,
            'metric_summary': metric_summary,
            'best_cases': best_cases
        }
    
    def print_analysis_results(self) -> None:
        """Print comprehensive analysis results."""
        print("\n" + "="*80)
        print("TECHNIQUE PERFORMANCE COMPARISON ANALYSIS")
        print("="*80)
        
        if not self.summary_stats:
            print("No analysis results available.")
            return
        
        # Overall summary
        print(f"\nOVERALL SUMMARY:")
        print(f"Total comparisons: {self.summary_stats['total_comparisons']}")
        print(f"Cases where drift techniques beat base: {self.summary_stats['significant_improvements']} ({self.summary_stats['improvement_rate']:.1f}%)")
        print(f"Cases with substantial improvement: {self.summary_stats['substantial_improvements']} ({self.summary_stats['substantial_improvement_rate']:.1f}%)")
        
        # Technique-specific results
        print(f"\nRESULTS BY TECHNIQUE:")
        print(f"{'Technique':<10} {'Better':<8} {'Substantial':<12} {'Avg Diff':<10} {'Avg %':<8} {'Avg Effect':<10}")
        print("-" * 70)
        
        for technique, row in self.summary_stats['technique_summary'].iterrows():
            print(f"{technique:<10} {int(row['technique_better']):<8} {int(row['technique_substantially_better']):<12} "
                  f"{row['mean_difference']:<10.4f} {row['percent_difference']:<8.2f}% {row['cohens_d']:<10.3f}")
        
        # Metric-specific results
        print(f"\nRESULTS BY METRIC:")
        print(f"{'Metric':<12} {'Better':<8} {'Substantial':<12} {'Avg Diff':<10} {'Avg %':<8} {'Avg Effect':<10}")
        print("-" * 72)
        
        for metric, row in self.summary_stats['metric_summary'].iterrows():
            print(f"{metric:<12} {int(row['technique_better']):<8} {int(row['technique_substantially_better']):<12} "
                  f"{row['mean_difference']:<10.4f} {row['percent_difference']:<8.2f}% {row['cohens_d']:<10.3f}")
        
        # Best cases
        print(f"\nBEST CASES (Substantial Improvements):")
        if len(self.summary_stats['best_cases']) > 0:
            print(f"{'Technique':<8} {'Metric':<10} {'Dataset':<30} {'Improvement':<12} {'Effect Size':<12}")
            print("-" * 80)
            
            for _, case in self.summary_stats['best_cases'].head(10).iterrows():
                dataset = case.get('dataset', 'Overall')
                print(f"{case['drift_technique']:<8} {case['metric']:<10} {dataset:<30} "
                      f"{case['percent_difference']:<12.2f}% {case['cohens_d']:<12.3f}")
        else:
            print("No substantial improvements found.")
        
        # Worst cases (where techniques significantly underperform)
        df = pd.DataFrame(self.comparison_results)
        worst_cases = df[df['technique_better'] == False].sort_values(
            'percent_difference', ascending=True
        )
        
        print(f"\nWORST CASES (Significant Underperformance):")
        if len(worst_cases) > 0:
            print(f"{'Technique':<8} {'Metric':<10} {'Dataset':<30} {'Decline':<12} {'Effect Size':<12}")
            print("-" * 80)
            
            for _, case in worst_cases.head(10).iterrows():
                dataset = case.get('dataset', 'Overall')
                print(f"{case['drift_technique']:<8} {case['metric']:<10} {dataset:<30} "
                      f"{case['percent_difference']:<12.2f}% {case['cohens_d']:<12.3f}")
        
        # Key insights
        print(f"\nKEY INSIGHTS:")
        
        if self.summary_stats['substantial_improvement_rate'] == 0:
            print("❌ NO SUBSTANTIAL IMPROVEMENTS: Drift detection techniques do not substantially outperform the base classifier in any scenario.")
        elif self.summary_stats['substantial_improvement_rate'] < 10:
            print(f"⚠️  MINIMAL IMPROVEMENTS: Only {self.summary_stats['substantial_improvement_rate']:.1f}% of cases show substantial improvement.")
        else:
            print(f"✅ SOME IMPROVEMENTS: {self.summary_stats['substantial_improvement_rate']:.1f}% of cases show substantial improvement.")
        
        # Determine best technique
        best_technique = self.summary_stats['technique_summary']['technique_substantially_better'].idxmax()
        best_count = self.summary_stats['technique_summary'].loc[best_technique, 'technique_substantially_better']
        
        if best_count > 0:
            print(f"🏆 BEST TECHNIQUE: {best_technique} with {int(best_count)} substantial improvements")
        else:
            print("🤷 NO CLEAR WINNER: No technique shows consistent substantial improvements")
    
    def save_detailed_results(self, filename: str = "technique_comparison_detailed.csv") -> None:
        """Save detailed comparison results to CSV."""
        if not self.comparison_results:
            print("No results to save.")
            return

        ensure_results_dirs()
        df = pd.DataFrame(self.comparison_results)

        # Round numerical columns
        numerical_cols = ['base_mean', 'tech_mean', 'mean_difference', 'percent_difference',
                         'pooled_se', 't_statistic', 'p_value', 'cohens_d',
                         'base_ci_lower', 'base_ci_upper', 'tech_ci_lower', 'tech_ci_upper']

        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].round(6)

        # Sort by improvement potential
        df = df.sort_values(['technique_substantially_better', 'percent_difference'],
                           ascending=[False, False])

        output_path = VALIDATION_RESULTS / filename
        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")
    
    def run_analysis(self) -> None:
        """Run the complete technique performance analysis."""
        print("Starting Technique Performance Comparison Analysis")
        print("="*80)
        
        # Load data
        self.load_technique_statistics()
        self.load_dataset_results()
        
        # Perform comparisons
        self.compare_techniques_vs_base()
        self.compare_by_dataset()
        
        # Generate summary
        self.generate_summary_statistics()
        
        # Print results
        self.print_analysis_results()
        
        # Save results
        self.save_detailed_results()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")

def main():
    """Main function to run the technique performance analysis."""
    analyzer = TechniquePerformanceAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
