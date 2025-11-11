#!/usr/bin/env python3
"""
Simple Fresh Validation

Simplified version without timer to get fresh single-run results quickly.
"""

import os
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from analysis.config import (
    VALIDATION_RESULTS,
    ensure_results_dirs
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.naive_bayes import MultinomialNB

from codes.common import (
    define_batches,
    find_indexes,
    load_and_prepare_dataset,
)
from codes.ddm import fetch_hdddm_drifts, fetch_jsddm_drifts, fetch_ksddm_drifts
from codes.drift_generation import (
    generate_synthetic_dataset_with_drifts,
    save_synthetic_dataset,
)

# Use centralized validation results directory
FRESH_OUTPUT_DIR = VALIDATION_RESULTS / "fresh_validation"


def run_prequential_naive_bayes(
    dataset: str,
    batch_size: int = 1000,
    batches_with_drift_list: Optional[List[str]] = None,
):
    """Run the prequential Naive Bayes algorithm on the dataset with specific drifts."""

    X, Y, _ = load_and_prepare_dataset(dataset)
    X = define_batches(X=X, batch_size=batch_size)
    Y = pd.DataFrame(Y, columns=["class"])
    Y = define_batches(X=Y, batch_size=batch_size)

    reference_batch = 1
    reference = X[X.Batch == reference_batch].iloc[:, :-1]
    y_reference = Y[Y.Batch == reference_batch].iloc[:, :-1].values.ravel()

    base_classifier = MultinomialNB()
    base_classifier.partial_fit(reference, y_reference, np.unique(Y["class"]))

    batches = list(set(X.Batch) - {reference_batch})
    batch_predictions = []
    drift_indexes = []

    y_pred = base_classifier.predict(reference)
    batch_predictions.append(y_pred)

    if batches_with_drift_list is not None:
        drift_indexes = [
            index + 1
            for index, value in enumerate(batches_with_drift_list)
            if value == "drift"
        ]

    for batch in batches:
        X_batch = X[X.Batch == batch].iloc[:, :-1]
        Y_batch = Y[Y.Batch == batch].iloc[:, :-1].values.ravel()

        # Test
        y_pred = base_classifier.predict(X_batch)
        batch_predictions.append(y_pred)

        # Train
        if batches_with_drift_list is not None:
            if batch in drift_indexes:
                reference = X_batch
                y_reference = Y_batch
                base_classifier = MultinomialNB()
                base_classifier.partial_fit(
                    reference, y_reference, np.unique(Y["class"])
                )
            else:
                base_classifier.partial_fit(X_batch, Y_batch)
        else:
            base_classifier.partial_fit(X_batch, Y_batch)

    batch_predictions = [item for sublist in batch_predictions for item in sublist]
    return X, Y, batch_predictions


def fetch_all_drifts_simple(
    batch_size,
    dataset,
    drift_alignment_within_batch: Optional[float] = None,
    custom_output_dir=None,
):
    """Fetch drift detection results without timing."""
    results = {}

    results["hd_drifts"] = fetch_hdddm_drifts(
        batch_size=batch_size,
        plot_heatmaps=False,
        dataset=dataset,
        drift_alignment_within_batch=drift_alignment_within_batch,
        custom_output_dir=custom_output_dir,
    )

    results["ks_drifts"] = fetch_ksddm_drifts(
        batch_size=batch_size,
        dataset=dataset,
        mean_threshold=0.05,
        plot_heatmaps=False,
        text="KSDDM 95",
        drift_alignment_within_batch=drift_alignment_within_batch,
        custom_output_dir=custom_output_dir,
    )

    results["ks_90_drifts"] = fetch_ksddm_drifts(
        batch_size=batch_size,
        dataset=dataset,
        plot_heatmaps=False,
        mean_threshold=0.10,
        text="KSDDM 90",
        drift_alignment_within_batch=drift_alignment_within_batch,
        custom_output_dir=custom_output_dir,
    )

    results["js_drifts"] = fetch_jsddm_drifts(
        batch_size=batch_size,
        plot_heatmaps=False,
        dataset=dataset,
        drift_alignment_within_batch=drift_alignment_within_batch,
        custom_output_dir=custom_output_dir,
    )

    return results


def run_test_simple(
    dataset: str,
    batch_size: int = 1000,
    detected_drifts_dict: dict = None,
):
    """Runs tests on the dataset using multiple drift detection methods."""

    X, Y, batch_predictions_base = run_prequential_naive_bayes(
        dataset=dataset, batch_size=batch_size
    )
    X, Y, batch_predictions_ks = run_prequential_naive_bayes(
        dataset=dataset,
        batch_size=batch_size,
        batches_with_drift_list=detected_drifts_dict["ks_drifts"],
    )
    X, Y, batch_predictions_ks_90 = run_prequential_naive_bayes(
        dataset=dataset,
        batch_size=batch_size,
        batches_with_drift_list=detected_drifts_dict["ks_90_drifts"],
    )
    X, Y, batch_predictions_hd = run_prequential_naive_bayes(
        dataset=dataset,
        batch_size=batch_size,
        batches_with_drift_list=detected_drifts_dict["hd_drifts"],
    )
    X, Y, batch_predictions_js = run_prequential_naive_bayes(
        dataset=dataset,
        batch_size=batch_size,
        batches_with_drift_list=detected_drifts_dict["js_drifts"],
    )
    y_true = Y["class"].values

    results = {
        "KS95": find_indexes(detected_drifts_dict["ks_drifts"].tolist()),
        "KS90": find_indexes(detected_drifts_dict["ks_90_drifts"].tolist()),
        "HD": find_indexes(detected_drifts_dict["hd_drifts"].tolist()),
        "JS": find_indexes(detected_drifts_dict["js_drifts"].tolist()),
    }

    metrics_results = {}

    # Calculate metrics
    for name, predictions in zip(
        ["Base", "KS95", "KS90", "HD", "JS"],
        [
            batch_predictions_base,
            batch_predictions_ks,
            batch_predictions_ks_90,
            batch_predictions_hd,
            batch_predictions_js,
        ],
    ):
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(
            y_true, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(y_true, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)
        fpr, tpr, _ = roc_curve(y_true, predictions, pos_label=1)
        roc_auc = auc(fpr, tpr)

        metrics_results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_curve": (fpr, tpr, roc_auc),
        }

        print(
            f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}"
        )

    return results, metrics_results, X.shape[0]


def save_results_to_csv(
    dataset: str,
    batch_size: int,
    drift_results: Dict[str, List[int]],
    metrics_results: dict,
    num_batches: int,
    csv_file_path: str,
    drift_alignment_with_batch: str = "N/A",
    scenario: str = "N/A",
    type_of_dataset: str = "N/A",
    algorithm: str = "N/A",
):
    """Save experiment results to csv."""
    # Create the data structure to be saved in CSV
    data = []
    for technique, metrics in metrics_results.items():
        if technique == "Base":
            num_drifts = 0
        else:
            num_drifts = len(drift_results[f"{technique}"])

        _, _, roc_auc = metrics["roc_curve"]

        row_data = {
            "dataset": dataset,
            "batch_size": batch_size,
            "technique": technique,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "num_drifts": num_drifts,
            "num_batches": num_batches,
            "auc": roc_auc,
            "drift_alignment_with_batch": drift_alignment_with_batch,
            "scenario": scenario,
            "type_of_dataset": type_of_dataset,
            "algorithm": algorithm,
        }
            
        data.append(row_data)

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Append the DataFrame to the CSV file
    df.to_csv(
        csv_file_path, mode="a", index=False, header=not os.path.exists(csv_file_path)
    )


def prepare_output_path(dataset):
    """Create and return the output directory for a dataset."""
    ensure_results_dirs()
    output_path = FRESH_OUTPUT_DIR / dataset
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def handle_synthetic_dataset(
    scenario,
    dataset,
    dataframe_size,
    batch_size,
    drift_within_batch: float = 1.0,
    features_with_drifts: list = None,
    num_drifts: int = 2,
):
    """Generate, save, and plot synthetic datasets."""
    if features_with_drifts is None:
        features_with_drifts = []
        
    (
        synthetic_df,
        drift_points,
        drift_info,
        accumulated_differences,
        features_with_drifts,
    ) = generate_synthetic_dataset_with_drifts(
        dataframe_size=dataframe_size,
        features_with_drifts=features_with_drifts,
        batch_size=batch_size,
        drift_within_batch=drift_within_batch,
        num_features=5,
        loc=10,
        scale=1,
        seed=42,  # Same seed as original for consistency
        scenario=scenario,
        num_drifts=num_drifts,
    )
    save_synthetic_dataset(synthetic_df, dataset)
    return synthetic_df, accumulated_differences, drift_points


def run_single_experiment(
    dataset,
    batch_size,
    detected_drifts_dict,
):
    """Run the main experiment pipeline for a single dataset and batch size."""

    drift_results, test_results, X_shape = run_test_simple(
        dataset=dataset,
        batch_size=batch_size,
        detected_drifts_dict=detected_drifts_dict,
    )
    num_batches = X_shape // batch_size
    return test_results, drift_results, num_batches


def run_fresh_validation_experiment():
    """Run fresh validation experiment on synthetic datasets only."""

    print("Starting Fresh Validation Experiment")
    print("=" * 60)

    # Create output directory
    ensure_results_dirs()
    FRESH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parameters matching original comparisor.py exactly
    synthetic_datasets = [
        "synthetic_dataset_with_parallel_drifts_abrupt",
        "synthetic_dataset_with_parallel_drifts_incremental", 
        "synthetic_dataset_with_switching_drifts_abrupt",
        "synthetic_dataset_with_switching_drifts_incremental",
        "synthetic_dataset_no_drifts"
    ]
    
    dataframe_size = 80000
    num_drifts = 2
    features_with_drifts = ["feature1", "feature3", "feature5"]
    batch_sizes = [1000, 1500, 2000, 2500]
    algorithm = "NB"

    for dataset in synthetic_datasets:
        print(f"\n=== Processing Dataset: {dataset} ===")

        # Clear old csvs
        output_path = prepare_output_path(dataset)
        csv_file_path = output_path / f"{dataset}_results.csv"

        if csv_file_path.exists():
            csv_file_path.unlink()

        for batch_size in batch_sizes:
            print(f"Processing {dataset} with batch size {batch_size}...")
            
            if dataset == "synthetic_dataset_no_drifts":
                scenario = "no_drifts"
                synthetic_df, accumulated_differences, synthetic_drift_points = handle_synthetic_dataset(
                    scenario,
                    dataset,
                    dataframe_size,
                    batch_size,
                    features_with_drifts=[],
                )

                detected_drifts_dict = fetch_all_drifts_simple(
                    batch_size,
                    dataset,
                    drift_alignment_within_batch=None,
                    custom_output_dir=str(FRESH_OUTPUT_DIR),
                )

                test_results, drift_results, num_batches = run_single_experiment(
                    dataset,
                    batch_size,
                    detected_drifts_dict,
                )

                save_results_to_csv(
                    dataset,
                    batch_size,
                    drift_results,
                    test_results,
                    num_batches,
                    csv_file_path,
                    scenario=scenario,
                    type_of_dataset="synthetic",
                    algorithm=algorithm,
                )

            else:
                drift_within_batch = 1.0
                scenario = "N/A"
                
                if dataset.startswith("synthetic_dataset_with_parallel_drifts"):
                    if "abrupt" in dataset:
                        scenario = "parallel_abrupt"
                    else:
                        scenario = "parallel_incremental"
                elif dataset.startswith("synthetic_dataset_with_switching_drifts"):
                    if "abrupt" in dataset:
                        scenario = "switching_abrupt"
                    else:
                        scenario = "switching_incremental"

                synthetic_df, accumulated_differences, synthetic_drift_points = handle_synthetic_dataset(
                    scenario,
                    dataset,
                    dataframe_size,
                    batch_size,
                    drift_within_batch,
                    features_with_drifts,
                    num_drifts=num_drifts,
                )

                detected_drifts_dict = fetch_all_drifts_simple(
                    batch_size,
                    dataset,
                    drift_alignment_within_batch=drift_within_batch,
                    custom_output_dir=str(FRESH_OUTPUT_DIR),
                )

                test_results, drift_results, num_batches = run_single_experiment(
                    dataset,
                    batch_size,
                    detected_drifts_dict,
                )

                save_results_to_csv(
                    dataset,
                    batch_size,
                    drift_results,
                    test_results,
                    num_batches,
                    csv_file_path,
                    drift_alignment_with_batch=str(drift_within_batch),
                    scenario=scenario,
                    type_of_dataset="synthetic",
                    algorithm=algorithm,
                )

    print(f"\nFresh validation experiment completed!")
    print(f"Results saved to: {FRESH_OUTPUT_DIR}")


if __name__ == "__main__":
    run_fresh_validation_experiment()
