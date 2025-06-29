import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np


def load_timing_data(timing_json_path: str) -> dict:
    """Load timing data from JSON file."""
    with open(timing_json_path, 'r') as f:
        return json.load(f)


def create_timing_analysis_report(timing_data: dict, output_dir: str = "timing_analysis"):
    """Create comprehensive timing analysis report with visualizations."""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract data for analysis
    dataset_timings = timing_data.get("dataset_timings", {})
    technique_performance = timing_data.get("technique_performance", {})
    experiment_metadata = timing_data.get("experiment_metadata", {})
    
    # 1. Dataset Performance Analysis
    dataset_performance_data = []
    for dataset, data in dataset_timings.items():
        for batch_size, batch_data in data.get("batch_timings", {}).items():
            dataset_performance_data.append({
                "dataset": dataset,
                "batch_size": batch_size,
                "drift_detection_time": batch_data.get("drift_detection", 0),
                "evaluation_time": batch_data.get("evaluation", 0),
                "total_time": batch_data.get("total", 0)
            })
    
    df_performance = pd.DataFrame(dataset_performance_data)
    
    # 2. Batch Size Analysis
    if not df_performance.empty:
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Total time by batch size
        plt.subplot(2, 2, 1)
        batch_size_analysis = df_performance.groupby('batch_size')['total_time'].agg(['mean', 'std']).reset_index()
        plt.bar(batch_size_analysis['batch_size'], batch_size_analysis['mean'], 
                yerr=batch_size_analysis['std'], capsize=5)
        plt.title('Average Execution Time by Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (seconds)')
        
        # Subplot 2: Phase breakdown by batch size
        plt.subplot(2, 2, 2)
        phase_data = df_performance.groupby('batch_size')[['drift_detection_time', 'evaluation_time']].mean()
        phase_data.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Phase Breakdown by Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (seconds)')
        plt.legend(['Drift Detection', 'Evaluation'])
        plt.xticks(rotation=45)
        
        # Subplot 3: Dataset comparison heatmap
        plt.subplot(2, 2, 3)
        pivot_data = df_performance.pivot_table(values='total_time', index='dataset', columns='batch_size', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Execution Time Heatmap (Dataset vs Batch Size)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Subplot 4: Top 10 slowest dataset-batch combinations
        plt.subplot(2, 2, 4)
        top_slow = df_performance.nlargest(10, 'total_time')
        top_slow['dataset_batch'] = top_slow['dataset'] + '\n(' + top_slow['batch_size'].astype(str) + ')'
        plt.barh(range(len(top_slow)), top_slow['total_time'])
        plt.yticks(range(len(top_slow)), top_slow['dataset_batch'])
        plt.title('Top 10 Slowest Combinations')
        plt.xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_analysis_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Technique Performance Analysis
    if technique_performance:
        plt.figure(figsize=(12, 6))
        
        techniques = list(technique_performance.keys())
        avg_times = [technique_performance[tech]['avg_time'] for tech in techniques]
        executions = [technique_performance[tech]['executions'] for tech in techniques]
        
        plt.subplot(1, 2, 1)
        plt.bar(techniques, avg_times)
        plt.title('Average Execution Time by Technique')
        plt.xlabel('Technique')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(techniques, executions)
        plt.title('Number of Executions by Technique')
        plt.xlabel('Technique')
        plt.ylabel('Number of Executions')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/technique_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Generate summary report
    report_lines = [
        "# Experiment Timing Analysis Report",
        "",
        "## Experiment Overview",
        f"- **Total Duration**: {experiment_metadata.get('total_duration', 0):.2f} seconds",
        f"- **Datasets Processed**: {experiment_metadata.get('datasets_processed', 0)}",
        f"- **Start Time**: {experiment_metadata.get('start_time', 'N/A')}",
        f"- **End Time**: {experiment_metadata.get('end_time', 'N/A')}",
        "",
        "## Performance Insights",
        ""
    ]
    
    if not df_performance.empty:
        # Batch size insights
        best_batch_size = df_performance.groupby('batch_size')['total_time'].mean().idxmin()
        worst_batch_size = df_performance.groupby('batch_size')['total_time'].mean().idxmax()
        
        report_lines.extend([
            "### Batch Size Analysis",
            f"- **Most Efficient Batch Size**: {best_batch_size}",
            f"- **Least Efficient Batch Size**: {worst_batch_size}",
            ""
        ])
        
        # Dataset insights
        slowest_dataset = df_performance.groupby('dataset')['total_time'].mean().idxmax()
        fastest_dataset = df_performance.groupby('dataset')['total_time'].mean().idxmin()
        
        report_lines.extend([
            "### Dataset Analysis",
            f"- **Slowest Dataset**: {slowest_dataset}",
            f"- **Fastest Dataset**: {fastest_dataset}",
            ""
        ])
        
        # Phase analysis
        avg_drift_time = df_performance['drift_detection_time'].mean()
        avg_eval_time = df_performance['evaluation_time'].mean()
        
        report_lines.extend([
            "### Phase Analysis",
            f"- **Average Drift Detection Time**: {avg_drift_time:.2f} seconds",
            f"- **Average Evaluation Time**: {avg_eval_time:.2f} seconds",
            f"- **Drift Detection vs Evaluation Ratio**: {avg_drift_time/avg_eval_time:.2f}:1" if avg_eval_time > 0 else "- **Drift Detection vs Evaluation Ratio**: N/A",
            ""
        ])
    
    # Technique insights
    if technique_performance:
        fastest_technique = min(technique_performance.keys(), key=lambda x: technique_performance[x]['avg_time'])
        slowest_technique = max(technique_performance.keys(), key=lambda x: technique_performance[x]['avg_time'])
        
        report_lines.extend([
            "### Technique Performance",
            f"- **Fastest Technique**: {fastest_technique} ({technique_performance[fastest_technique]['avg_time']:.3f}s avg)",
            f"- **Slowest Technique**: {slowest_technique} ({technique_performance[slowest_technique]['avg_time']:.3f}s avg)",
            ""
        ])
    
    # Save detailed CSV for further analysis
    if not df_performance.empty:
        df_performance.to_csv(f"{output_dir}/detailed_timing_data.csv", index=False)
        report_lines.extend([
            "## Files Generated",
            "- `timing_analysis_overview.png`: Comprehensive timing visualizations",
            "- `technique_performance.png`: Technique-specific performance charts",
            "- `detailed_timing_data.csv`: Raw timing data for further analysis",
            "- `timing_analysis_report.md`: This summary report",
        ])
    
    # Save report
    with open(f"{output_dir}/timing_analysis_report.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Timing analysis complete! Results saved to {output_dir}/")
    return df_performance


def analyze_technique_performance_from_csv(csv_path: str, output_dir: str = "timing_analysis"):
    """Analyze technique-level performance from the enhanced CSV format."""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter technique-level data
    technique_data = df[df['type'] == 'technique_timing'].copy()
    
    if technique_data.empty:
        print("No technique-level timing data found in CSV.")
        return
    
    # Create comprehensive technique analysis
    plt.figure(figsize=(20, 15))
    
    # 1. Technique comparison by average time
    plt.subplot(3, 3, 1)
    technique_avg = technique_data.groupby('technique')['duration_seconds'].mean().sort_values()
    technique_avg.plot(kind='bar')
    plt.title('Average Execution Time by Technique')
    plt.xlabel('Technique')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    # 2. Technique performance by batch size
    plt.subplot(3, 3, 2)
    technique_batch = technique_data.pivot_table(values='duration_seconds', index='technique', columns='batch_size', aggfunc='mean')
    sns.heatmap(technique_batch, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Technique Performance by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Technique')
    
    # 3. Technique consistency (std deviation)
    plt.subplot(3, 3, 3)
    technique_std = technique_data.groupby('technique')['duration_seconds'].std().sort_values()
    technique_std.plot(kind='bar', color='orange')
    plt.title('Technique Performance Consistency (Lower = More Consistent)')
    plt.xlabel('Technique')
    plt.ylabel('Standard Deviation (seconds)')
    plt.xticks(rotation=45)
    
    # 4. Top 10 fastest technique-dataset combinations
    plt.subplot(3, 3, 4)
    top_fast = technique_data.nsmallest(10, 'duration_seconds')
    top_fast['combo'] = top_fast['technique'] + '\n' + top_fast['dataset'].str[:15]
    plt.barh(range(len(top_fast)), top_fast['duration_seconds'])
    plt.yticks(range(len(top_fast)), top_fast['combo'])
    plt.title('Top 10 Fastest Technique-Dataset Combinations')
    plt.xlabel('Time (seconds)')
    
    # 5. Top 10 slowest technique-dataset combinations
    plt.subplot(3, 3, 5)
    top_slow = technique_data.nlargest(10, 'duration_seconds')
    top_slow['combo'] = top_slow['technique'] + '\n' + top_slow['dataset'].str[:15]
    plt.barh(range(len(top_slow)), top_slow['duration_seconds'])
    plt.yticks(range(len(top_slow)), top_slow['combo'])
    plt.title('Top 10 Slowest Technique-Dataset Combinations')
    plt.xlabel('Time (seconds)')
    
    # 6. Technique performance distribution
    plt.subplot(3, 3, 6)
    for technique in technique_data['technique'].unique():
        tech_times = technique_data[technique_data['technique'] == technique]['duration_seconds']
        plt.hist(tech_times, alpha=0.7, label=technique, bins=20)
    plt.title('Technique Performance Distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 7. Batch size scaling for each technique
    plt.subplot(3, 3, 7)
    for technique in technique_data['technique'].unique():
        tech_batch_data = technique_data[technique_data['technique'] == technique]
        batch_avg = tech_batch_data.groupby('batch_size')['duration_seconds'].mean()
        plt.plot(batch_avg.index, batch_avg.values, marker='o', label=technique)
    plt.title('Technique Scaling by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    
    # 8. Dataset complexity ranking by technique
    plt.subplot(3, 3, 8)
    dataset_complexity = technique_data.groupby('dataset')['duration_seconds'].mean().sort_values(ascending=False)
    dataset_complexity.head(10).plot(kind='barh')
    plt.title('Top 10 Most Complex Datasets (by avg technique time)')
    plt.xlabel('Average Time (seconds)')
    
    # 9. Technique efficiency ratio (fastest vs slowest)
    plt.subplot(3, 3, 9)
    technique_ratios = []
    for technique in technique_data['technique'].unique():
        tech_times = technique_data[technique_data['technique'] == technique]['duration_seconds']
        ratio = tech_times.max() / tech_times.min() if tech_times.min() > 0 else 0
        technique_ratios.append((technique, ratio))
    
    technique_ratios.sort(key=lambda x: x[1])
    techniques, ratios = zip(*technique_ratios)
    plt.bar(techniques, ratios)
    plt.title('Technique Performance Variability (Lower = More Stable)')
    plt.xlabel('Technique')
    plt.ylabel('Max/Min Time Ratio')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/technique_detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate technique comparison report
    report_lines = [
        "# Technique Performance Analysis Report",
        "",
        "## Technique Rankings",
        ""
    ]
    
    # Add technique rankings
    report_lines.append("### By Average Speed (Fastest to Slowest)")
    for i, (technique, avg_time) in enumerate(technique_avg.items(), 1):
        report_lines.append(f"{i}. **{technique}**: {avg_time:.3f}s average")
    
    report_lines.extend(["", "### By Consistency (Most to Least Consistent)"])
    for i, (technique, std_time) in enumerate(technique_std.items(), 1):
        report_lines.append(f"{i}. **{technique}**: {std_time:.3f}s std deviation")
    
    # Add batch size recommendations
    report_lines.extend(["", "## Batch Size Recommendations", ""])
    for technique in technique_data['technique'].unique():
        tech_batch_data = technique_data[technique_data['technique'] == technique]
        best_batch = tech_batch_data.groupby('batch_size')['duration_seconds'].mean().idxmin()
        worst_batch = tech_batch_data.groupby('batch_size')['duration_seconds'].mean().idxmax()
        report_lines.append(f"- **{technique}**: Best batch size = {best_batch}, Worst = {worst_batch}")
    
    # Save technique-specific CSV
    technique_summary = technique_data.groupby(['technique', 'batch_size']).agg({
        'duration_seconds': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    technique_summary.to_csv(f"{output_dir}/technique_performance_summary.csv")
    
    # Save report
    with open(f"{output_dir}/technique_analysis_report.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Technique analysis complete! Results saved to {output_dir}/")
    print(f"Key files: technique_detailed_analysis.png, technique_performance_summary.csv")
    
    return technique_data


def compare_batch_sizes(timing_data: dict):
    """Generate specific batch size comparison insights."""
    dataset_timings = timing_data.get("dataset_timings", {})
    
    batch_size_stats = {}
    for dataset, data in dataset_timings.items():
        for batch_size, batch_data in data.get("batch_timings", {}).items():
            if batch_size not in batch_size_stats:
                batch_size_stats[batch_size] = []
            batch_size_stats[batch_size].append(batch_data.get("total", 0))
    
    print("\n" + "="*50)
    print("BATCH SIZE PERFORMANCE COMPARISON")
    print("="*50)
    
    for batch_size in sorted(batch_size_stats.keys(), key=int):
        times = batch_size_stats[batch_size]
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Batch Size {int(batch_size):4d}: {avg_time:6.2f}s ± {std_time:5.2f}s (n={len(times)})")
    
    return batch_size_stats


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        timing_file = sys.argv[1]
        
        # Analyze JSON data
        timing_data = load_timing_data(timing_file)
        create_timing_analysis_report(timing_data)
        compare_batch_sizes(timing_data)
        
        # Analyze CSV data for technique-level insights
        csv_file = timing_file.replace('experiment_timing_results.json', 'timing_summary.csv')
        if Path(csv_file).exists():
            print(f"\nAnalyzing technique-level data from {csv_file}...")
            analyze_technique_performance_from_csv(csv_file)
        else:
            print(f"CSV file not found: {csv_file}")
            print("Technique-level analysis requires the timing_summary.csv file.")
    else:
        print("Usage: python timing_analysis.py <timing_json_file>")
        print("Example: python timing_analysis.py comparison_results/experiment_timing_results.json")
        print("\nThis will analyze both the JSON file and the corresponding CSV file for complete insights.")
