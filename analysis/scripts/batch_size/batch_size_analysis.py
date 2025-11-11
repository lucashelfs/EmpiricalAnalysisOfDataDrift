#!/usr/bin/env python3
"""
Batch Size Analysis for 3-Run Experiment

This script analyzes the impact of different batch sizes (1000, 1500, 2000, 2500)
on drift detection performance across all synthetic scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_batch_size_results():
    """Load the 3-run batch size experiment results."""
    results_path = "comparison_results/multi_run_20250712_132621/analysis/all_runs_combined.csv"
    
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        print(f"✅ Loaded {len(df)} records from batch size experiment")
        return df
    else:
        print(f"❌ No results found at {results_path}")
        return None

def analyze_batch_size_impact():
    """Analyze the impact of batch sizes on drift detection performance."""
    
    df = load_batch_size_results()
    if df is None:
        return
    
    print("\n" + "="*80)
    print("BATCH SIZE IMPACT ANALYSIS")
    print("="*80)
    
    # Basic experiment info
    print(f"\n📊 EXPERIMENT CONFIGURATION:")
    print(f"Runs: {df['run_id'].nunique()}")
    print(f"Batch Sizes: {sorted(df['batch_size'].unique())}")
    print(f"Techniques: {df['technique'].unique().tolist()}")
    print(f"Scenarios: {df['dataset'].unique().tolist()}")
    print(f"Total Records: {len(df)}")
    
    # Overall batch size performance
    print(f"\n📏 OVERALL BATCH SIZE PERFORMANCE:")
    batch_performance = df.groupby('batch_size')['accuracy'].agg(['mean', 'std', 'count']).round(4)
    for batch_size, row in batch_performance.iterrows():
        print(f"  Batch {batch_size}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})")
    
    # Technique-wise batch size analysis
    print(f"\n🔬 TECHNIQUE-WISE BATCH SIZE ANALYSIS:")
    for technique in sorted(df['technique'].unique()):
        print(f"\n  {technique}:")
        technique_data = df[df['technique'] == technique]
        technique_batch_perf = technique_data.groupby('batch_size')['accuracy'].agg(['mean', 'std']).round(4)
        
        for batch_size, row in technique_batch_perf.iterrows():
            print(f"    Batch {batch_size}: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Scenario-wise batch size analysis
    print(f"\n📁 SCENARIO-WISE BATCH SIZE ANALYSIS:")
    for dataset in df['dataset'].unique():
        scenario_name = get_scenario_name(dataset)
        print(f"\n  {scenario_name}:")
        scenario_data = df[df['dataset'] == dataset]
        scenario_batch_perf = scenario_data.groupby('batch_size')['accuracy'].agg(['mean', 'std']).round(4)
        
        for batch_size, row in scenario_batch_perf.iterrows():
            print(f"    Batch {batch_size}: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Statistical significance testing
    print(f"\n📈 STATISTICAL SIGNIFICANCE TESTING:")
    batch_sizes = sorted(df['batch_size'].unique())
    
    print(f"Pairwise t-tests between batch sizes:")
    for i, batch1 in enumerate(batch_sizes):
        for batch2 in batch_sizes[i+1:]:
            data1 = df[df['batch_size'] == batch1]['accuracy']
            data2 = df[df['batch_size'] == batch2]['accuracy']
            
            t_stat, p_value = stats.ttest_ind(data1, data2)
            significant = "Yes" if p_value < 0.05 else "No"
            
            print(f"  Batch {batch1} vs {batch2}: p={p_value:.4f}, significant={significant}")
    
    # Best batch size per technique
    print(f"\n🏆 BEST BATCH SIZE PER TECHNIQUE:")
    for technique in sorted(df['technique'].unique()):
        technique_data = df[df['technique'] == technique]
        best_batch = technique_data.groupby('batch_size')['accuracy'].mean().idxmax()
        best_performance = technique_data.groupby('batch_size')['accuracy'].mean().max()
        
        print(f"  {technique}: Batch {best_batch} ({best_performance:.4f} accuracy)")
    
    # Drift detection effectiveness by batch size
    print(f"\n🎯 DRIFT DETECTION EFFECTIVENESS BY BATCH SIZE:")
    for batch_size in sorted(df['batch_size'].unique()):
        batch_data = df[df['batch_size'] == batch_size]
        base_perf = batch_data[batch_data['technique'] == 'Base']['accuracy'].mean()
        
        print(f"\n  Batch Size {batch_size} (Base: {base_perf:.4f}):")
        for technique in ['KS95', 'KS90', 'HD', 'JS']:
            if technique in batch_data['technique'].values:
                tech_perf = batch_data[batch_data['technique'] == technique]['accuracy'].mean()
                benefit = tech_perf - base_perf
                print(f"    {technique}: {tech_perf:.4f} (benefit: {benefit:+.4f})")
    
    # Generate visualizations
    create_batch_size_visualizations(df)
    
    # Generate detailed report
    generate_batch_size_report(df)
    
    print(f"\n✅ Batch size analysis completed!")

def get_scenario_name(dataset):
    """Convert dataset name to readable scenario name."""
    if "parallel_drifts_abrupt" in dataset:
        return "Parallel Abrupt"
    elif "parallel_drifts_incremental" in dataset:
        return "Parallel Incremental"
    elif "switching_drifts_abrupt" in dataset:
        return "Switching Abrupt"
    elif "switching_drifts_incremental" in dataset:
        return "Switching Incremental"
    elif "no_drifts" in dataset:
        return "No Drifts"
    else:
        return dataset

def create_batch_size_visualizations(df):
    """Create visualizations for batch size analysis."""
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = "batch_size_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall Batch Size Performance
    plt.figure(figsize=(15, 10))
    
    # Box plot by batch size
    plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x='batch_size', y='accuracy')
    plt.title('Overall Performance by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    
    # Technique comparison across batch sizes
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x='batch_size', y='accuracy', hue='technique')
    plt.title('Technique Performance by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Scenario comparison across batch sizes
    plt.subplot(2, 3, 3)
    df_scenarios = df.copy()
    df_scenarios['scenario'] = df_scenarios['dataset'].apply(get_scenario_name)
    sns.boxplot(data=df_scenarios, x='batch_size', y='accuracy', hue='scenario')
    plt.title('Scenario Performance by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Mean performance by batch size
    plt.subplot(2, 3, 4)
    batch_means = df.groupby('batch_size')['accuracy'].mean()
    batch_stds = df.groupby('batch_size')['accuracy'].std()
    
    plt.errorbar(batch_means.index, batch_means.values, yerr=batch_stds.values, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.title('Mean Performance by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Technique-wise mean performance
    plt.subplot(2, 3, 5)
    technique_batch_means = df.groupby(['batch_size', 'technique'])['accuracy'].mean().unstack()
    technique_batch_means.plot(kind='line', marker='o')
    plt.title('Technique Performance Trends')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.legend(title='Technique')
    plt.grid(True, alpha=0.3)
    
    # Drift detection benefit by batch size
    plt.subplot(2, 3, 6)
    benefits = []
    batch_sizes = []
    techniques = []
    
    for batch_size in sorted(df['batch_size'].unique()):
        batch_data = df[df['batch_size'] == batch_size]
        base_mean = batch_data[batch_data['technique'] == 'Base']['accuracy'].mean()
        
        for technique in ['KS95', 'KS90', 'HD', 'JS']:
            if technique in batch_data['technique'].values:
                tech_mean = batch_data[batch_data['technique'] == technique]['accuracy'].mean()
                benefit = tech_mean - base_mean
                benefits.append(benefit)
                batch_sizes.append(batch_size)
                techniques.append(technique)
    
    benefit_df = pd.DataFrame({
        'batch_size': batch_sizes,
        'technique': techniques,
        'benefit': benefits
    })
    
    sns.barplot(data=benefit_df, x='batch_size', y='benefit', hue='technique')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Drift Detection Benefit by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy Benefit vs Base')
    plt.legend(title='Technique')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed heatmap
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    heatmap_data = df.groupby(['technique', 'batch_size'])['accuracy'].mean().unstack()
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Accuracy'})
    plt.title('Technique Performance Heatmap by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Technique')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_size_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}/")

def generate_batch_size_report(df):
    """Generate a comprehensive batch size analysis report."""
    
    output_dir = "batch_size_analysis"
    report_path = f"{output_dir}/batch_size_analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Batch Size Impact Analysis Report\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report analyzes the impact of different batch sizes (1000, 1500, 2000, 2500) ")
        f.write("on drift detection performance across all synthetic scenarios with 10 drift points.\n\n")
        
        f.write("## Methodology\n\n")
        f.write("### Experimental Setup\n")
        f.write(f"- Independent Runs: {df['run_id'].nunique()}\n")
        f.write(f"- Batch Sizes Tested: {', '.join(map(str, sorted(df['batch_size'].unique())))}\n")
        f.write(f"- Drift Detection Techniques: {', '.join([t for t in df['technique'].unique() if t != 'Base'])}\n")
        f.write(f"- Scenarios: {len(df['dataset'].unique())} synthetic scenarios\n")
        f.write(f"- Algorithm: {', '.join(df['algorithm'].unique())}\n")
        f.write(f"- Dataset Size: 80,000 entries per scenario\n")
        f.write(f"- Number of Drifts: 10 per scenario\n\n")
        
        f.write("## Results\n\n")
        
        f.write("### 1. Overall Batch Size Performance\n\n")
        batch_performance = df.groupby('batch_size')['accuracy'].agg(['mean', 'std', 'count']).round(4)
        
        f.write("| Batch Size | Mean Accuracy | Std Dev | Sample Size |\n")
        f.write("|------------|---------------|---------|-------------|\n")
        for batch_size, row in batch_performance.iterrows():
            f.write(f"| {batch_size} | {row['mean']:.4f} | {row['std']:.4f} | {row['count']} |\n")
        
        f.write("\n### 2. Best Performing Batch Size per Technique\n\n")
        f.write("| Technique | Best Batch Size | Best Performance |\n")
        f.write("|-----------|-----------------|------------------|\n")
        
        for technique in sorted(df['technique'].unique()):
            technique_data = df[df['technique'] == technique]
            best_batch = technique_data.groupby('batch_size')['accuracy'].mean().idxmax()
            best_performance = technique_data.groupby('batch_size')['accuracy'].mean().max()
            f.write(f"| {technique} | {best_batch} | {best_performance:.4f} |\n")
        
        f.write("\n### 3. Drift Detection Effectiveness by Batch Size\n\n")
        for batch_size in sorted(df['batch_size'].unique()):
            batch_data = df[df['batch_size'] == batch_size]
            base_perf = batch_data[batch_data['technique'] == 'Base']['accuracy'].mean()
            
            f.write(f"**Batch Size {batch_size}** (Base Performance: {base_perf:.4f})\n\n")
            f.write("| Technique | Performance | Benefit vs Base |\n")
            f.write("|-----------|-------------|------------------|\n")
            
            for technique in ['KS95', 'KS90', 'HD', 'JS']:
                if technique in batch_data['technique'].values:
                    tech_perf = batch_data[batch_data['technique'] == technique]['accuracy'].mean()
                    benefit = tech_perf - base_perf
                    f.write(f"| {technique} | {tech_perf:.4f} | {benefit:+.4f} |\n")
            f.write("\n")
        
        f.write("### 4. Statistical Significance Testing\n\n")
        f.write("Pairwise t-tests between batch sizes:\n\n")
        f.write("| Comparison | P-value | Significant (p<0.05) |\n")
        f.write("|------------|---------|----------------------|\n")
        
        batch_sizes = sorted(df['batch_size'].unique())
        for i, batch1 in enumerate(batch_sizes):
            for batch2 in batch_sizes[i+1:]:
                data1 = df[df['batch_size'] == batch1]['accuracy']
                data2 = df[df['batch_size'] == batch2]['accuracy']
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                significant = "Yes" if p_value < 0.05 else "No"
                
                f.write(f"| Batch {batch1} vs {batch2} | {p_value:.4f} | {significant} |\n")
        
        f.write("\n### 5. Key Findings\n\n")
        
        # Find best overall batch size
        best_batch_overall = df.groupby('batch_size')['accuracy'].mean().idxmax()
        best_performance_overall = df.groupby('batch_size')['accuracy'].mean().max()
        
        f.write(f"- **Best Overall Batch Size**: {best_batch_overall} ({best_performance_overall:.4f} accuracy)\n")
        
        # Count how many techniques favor each batch size
        technique_preferences = {}
        for technique in df['technique'].unique():
            technique_data = df[df['technique'] == technique]
            best_batch = technique_data.groupby('batch_size')['accuracy'].mean().idxmax()
            if best_batch not in technique_preferences:
                technique_preferences[best_batch] = []
            technique_preferences[best_batch].append(technique)
        
        f.write(f"- **Technique Preferences**:\n")
        for batch_size, techniques in technique_preferences.items():
            f.write(f"  - Batch {batch_size}: {', '.join(techniques)}\n")
        
        # Performance range
        min_perf = df.groupby('batch_size')['accuracy'].mean().min()
        max_perf = df.groupby('batch_size')['accuracy'].mean().max()
        perf_range = max_perf - min_perf
        
        f.write(f"- **Performance Range**: {perf_range:.4f} ({min_perf:.4f} to {max_perf:.4f})\n")
        
        f.write("\n## Conclusions\n\n")
        
        f.write("### Primary Research Questions\n\n")
        
        f.write("**1. Which batch size provides the best overall performance?**\n")
        f.write(f"- Batch size {best_batch_overall} achieves the highest mean accuracy ({best_performance_overall:.4f})\n")
        if perf_range < 0.01:
            f.write("- Performance differences between batch sizes are minimal (< 1%)\n")
        else:
            f.write(f"- Performance differences are moderate ({perf_range:.1%})\n")
        
        f.write("\n**2. Do different techniques favor different batch sizes?**\n")
        unique_preferences = len(set(technique_preferences.keys()))
        if unique_preferences == 1:
            f.write("- All techniques favor the same batch size\n")
        else:
            f.write(f"- Techniques show varied preferences across {unique_preferences} different batch sizes\n")
        
        f.write("\n**3. How does batch size affect drift detection effectiveness?**\n")
        # Calculate average benefit across all techniques for each batch size
        avg_benefits = {}
        for batch_size in sorted(df['batch_size'].unique()):
            batch_data = df[df['batch_size'] == batch_size]
            base_perf = batch_data[batch_data['technique'] == 'Base']['accuracy'].mean()
            
            benefits = []
            for technique in ['KS95', 'KS90', 'HD', 'JS']:
                if technique in batch_data['technique'].values:
                    tech_perf = batch_data[batch_data['technique'] == technique]['accuracy'].mean()
                    benefit = tech_perf - base_perf
                    benefits.append(benefit)
            
            avg_benefits[batch_size] = np.mean(benefits) if benefits else 0
        
        best_benefit_batch = max(avg_benefits.keys(), key=lambda k: avg_benefits[k])
        best_benefit_value = avg_benefits[best_benefit_batch]
        
        if best_benefit_value > 0:
            f.write(f"- Batch size {best_benefit_batch} provides the best drift detection benefit ({best_benefit_value:+.4f})\n")
        else:
            f.write("- All batch sizes show negative drift detection benefits (Base algorithm dominates)\n")
        
        f.write("\n### Practical Recommendations\n\n")
        
        f.write("1. **For General Use**:\n")
        f.write(f"   - Use batch size {best_batch_overall} for optimal overall performance\n")
        f.write("   - Performance differences between batch sizes are relatively small\n")
        
        f.write("\n2. **For Specific Techniques**:\n")
        for batch_size, techniques in technique_preferences.items():
            if len(techniques) > 1:
                f.write(f"   - Batch size {batch_size} recommended for: {', '.join(techniques)}\n")
        
        f.write("\n3. **For Computational Efficiency**:\n")
        f.write("   - Smaller batch sizes (1000) provide more frequent drift detection opportunities\n")
        f.write("   - Larger batch sizes (2500) reduce computational overhead\n")
        f.write("   - Choose based on latency vs. throughput requirements\n")
        
        f.write("\n### Limitations\n\n")
        f.write(f"- Analysis limited to {df['run_id'].nunique()} independent runs\n")
        f.write("- Results specific to synthetic datasets with 10 drift points\n")
        f.write("- Performance may vary with different drift characteristics\n")
        f.write("- Computational cost analysis not included\n\n")
        
        f.write("---\n\n")
        f.write("*This analysis was conducted using the 3-run batch size comparison experimental framework.*\n")
    
    print(f"📄 Detailed report saved to {report_path}")

if __name__ == "__main__":
    analyze_batch_size_impact()
