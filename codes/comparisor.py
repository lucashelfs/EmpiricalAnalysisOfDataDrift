import os
from typing import List, Dict, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB

from codes.common import (
    common_datasets,
    define_batches,
    datasets_with_added_drifts,
    find_indexes,
    load_and_prepare_dataset,
)
from codes.config import comparisons_output_dir as output_dir, insects_datasets
from codes.ddm import fetch_ksddm_drifts, fetch_hdddm_drifts, fetch_jsddm_drifts
from codes.drift_generation import create_synthetic_dataframe, save_synthetic_dataset
from codes.drift_generation import generate_synthetic_dataset_with_drifts
from drift_config import drift_config
from codes.common import calculate_index, extract_drift_info


# Set global style parameters
plt.rcParams["figure.facecolor"] = "#EAEAF2"  # Set the background color to grey
plt.rcParams["axes.facecolor"] = "#EAEAF2"  # Set the axes background color to grey
plt.rcParams[
    "savefig.facecolor"
] = "#EAEAF2"  # Set the saved figure background color to grey
plt.rcParams["grid.color"] = "white"  # Set the grid color to white
plt.rcParams["axes.grid"] = True  # Enable grid by default
plt.rcParams["legend.frameon"] = True  # Enable legend frame
plt.rcParams["legend.framealpha"] = 0.9  # Set legend frame transparency
plt.rcParams["legend.facecolor"] = "white"  # Set legend background color
plt.rcParams["legend.edgecolor"] = "black"  # Set legend edge color

###


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


def run_test(dataset: str, batch_size: int = 1000, plot_heatmaps: bool = True):
    """Runs tests on the dataset using multiple drift detection methods."""

    ks_drifts = fetch_ksddm_drifts(
        batch_size=batch_size,
        dataset=dataset,
        mean_threshold=0.05,
        plot_heatmaps=plot_heatmaps,
        text="KSDDM 95",
    )
    ks_90_drifts = fetch_ksddm_drifts(
        batch_size=batch_size,
        dataset=dataset,
        plot_heatmaps=plot_heatmaps,
        mean_threshold=0.10,
        text="KSDDM 90",
    )
    hd_drifts = fetch_hdddm_drifts(
        batch_size=batch_size, plot_heatmaps=plot_heatmaps, dataset=dataset
    )
    js_drifts = fetch_jsddm_drifts(
        batch_size=batch_size, plot_heatmaps=plot_heatmaps, dataset=dataset
    )

    X, Y, batch_predictions_base = run_prequential_naive_bayes(
        dataset=dataset, batch_size=batch_size
    )
    X, Y, batch_predictions_ks = run_prequential_naive_bayes(
        dataset=dataset, batch_size=batch_size, batches_with_drift_list=ks_drifts
    )
    X, Y, batch_predictions_ks_90 = run_prequential_naive_bayes(
        dataset=dataset, batch_size=batch_size, batches_with_drift_list=ks_90_drifts
    )
    X, Y, batch_predictions_hd = run_prequential_naive_bayes(
        dataset=dataset, batch_size=batch_size, batches_with_drift_list=hd_drifts
    )
    X, Y, batch_predictions_js = run_prequential_naive_bayes(
        dataset=dataset, batch_size=batch_size, batches_with_drift_list=js_drifts
    )

    y_true = Y["class"].values

    results = {
        "KS95": find_indexes(ks_drifts.tolist()),
        "KS90": find_indexes(ks_90_drifts.tolist()),
        "HD": find_indexes(hd_drifts.tolist()),
        "JS": find_indexes(js_drifts.tolist()),
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


def plot_drift_points(
    drift_results: Dict[str, List[int]], dataset: str, batch_size: int
):
    """Plot drift points."""
    os.makedirs(output_dir + f"/{dataset}/detected_drifts/", exist_ok=True)
    plt.figure(figsize=(12, 8))
    colors = {"KS95": "r", "KS90": "g", "HD": "b", "JS": "m"}

    # Ensure all methods are on the Y-axis
    methods = ["KS95", "KS90", "HD", "JS"]
    for method in methods:
        drifts = drift_results.get(method, [])
        plt.scatter(
            drifts,
            [method] * len(drifts),
            color=colors[method],
            label=method if len(drifts) > 0 else None,
            s=50,
        )

    # Plot all change points
    change_points = fetch_dataset_change_points(dataset, batch_size)
    label_added = False
    for cp in change_points:
        if not label_added:
            plt.axvline(
                x=cp, color="k", linestyle="--", linewidth=1, label="Change point"
            )
            label_added = True
        else:
            plt.axvline(x=cp, color="k", linestyle="--", linewidth=1)

    plt.xlabel("Batch Index", fontsize=20)
    # plt.ylabel('Detection Method', fontsize=14)
    plt.title(f"Drift Points for {dataset} (Batch Size: {batch_size})", fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.yticks(ticks=methods, labels=methods, fontsize=15)
    plt.xticks(fontsize=15)

    # Set integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig(
        os.path.join(
            output_dir + f"/{dataset}/detected_drifts/",
            f"{dataset}_{batch_size}_drift_points.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_results(
    results: dict[int, dict[Any, dict[str, float | tuple[Any, Any, float]]]],
    dataset: str,
    batch_sizes: List[int],
):
    """Plot metric results."""

    os.makedirs(output_dir + f"/{dataset}/metrics/", exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f1"]

    # Create separate plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model in results[batch_sizes[0]]:
            metric_values = [
                results[batch_size][model][metric] for batch_size in batch_sizes
            ]
            plt.plot(batch_sizes, metric_values, marker="o", label=model)
        plt.xlabel("Batch Size")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Comparison for {dataset}")
        plt.legend()
        plt.grid(True)
        plt.xticks(batch_sizes)
        plt.savefig(os.path.join(output_dir + f"/{dataset}/metrics/", f"{metric}.png"))
        plt.close()

    # Plot ROC Curve for each batch size in separate files
    for batch_size in batch_sizes:
        plt.figure(figsize=(10, 6))
        for model in results[batch_size]:
            fpr, tpr, roc_auc = results[batch_size][model]["roc_curve"]
            plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {dataset} (Batch Size: {batch_size})")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(
                output_dir + f"/{dataset}/metrics/", f"roc_curve_{batch_size}.png"
            )
        )
        plt.close()

    # Collect all ROC curves for top 3 plot
    roc_curves = []
    for batch_size in batch_sizes:
        for model in results[batch_size]:
            fpr, tpr, roc_auc = results[batch_size][model]["roc_curve"]
            roc_curves.append(
                (fpr, tpr, roc_auc, f"{model} (Batch Size: {batch_size})")
            )

    # Sort ROC curves by AUC in descending order and plot the top 3
    roc_curves.sort(key=lambda x: x[2], reverse=True)
    plt.figure(figsize=(10, 6))
    for fpr, tpr, roc_auc, label in roc_curves[:3]:
        plt.plot(fpr, tpr, label=f"{label}, AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Top 3 ROC Curves for {dataset}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir + f"/{dataset}/metrics/", f"top3_roc_curve.png")
    )
    plt.close()


def save_results_to_csv(
    dataset: str,
    batch_size: int,
    drift_results: Dict[str, List[int]],
    metrics_results: dict[Any, dict[str, float | tuple[Any, Any, float]]],
    num_batches: int,
    csv_file_path: str,
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

        data.append(
            {
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
            }
        )

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Append the DataFrame to the CSV file
    df.to_csv(
        csv_file_path, mode="a", index=False, header=not os.path.exists(csv_file_path)
    )
    print(f"Results for {dataset}, batch size {batch_size} saved to {csv_file_path}")


def consolidate_csv_files(csv_file_paths: List[str], target_csv_file: str):
    """Consolidate all csv from experiment onto the target csv."""
    df_list = []

    # Read each CSV file and append it to the list
    for file in csv_file_paths:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all data frames into a single data frame
    consolidated_df = pd.concat(df_list, ignore_index=True)

    # Save the consolidated data frame to the target CSV file
    consolidated_df.to_csv(target_csv_file, index=False)
    print(f"Consolidated results saved to {target_csv_file}")


def fetch_dataset_change_points(dataset_name: str, batch_size: int):
    """Fetch all change points from dataset."""
    change_points = []

    if dataset_name in insects_datasets.keys():
        change_points = insects_datasets[dataset_name]["change_point"]

    if dataset_name in common_datasets.keys():
        change_points = common_datasets[dataset_name]["change_point"]

    if dataset_name in datasets_with_added_drifts:
        df, _, _ = load_and_prepare_dataset(dataset_name)
        _, column, drifts = extract_drift_info(dataset_name)
        for drift_type in drifts:
            for drift in drifts[drift_type]:
                start_index = calculate_index(df, drift[0])
                end_index = calculate_index(df, drift[1])
                change_points.extend([start_index, end_index])

    batches_with_change_points = [cp // batch_size for cp in change_points]
    return batches_with_change_points


def fetch_change_points(dataset_name: str, df: pd.DataFrame) -> list:
    """Fetch change points for the given dataset."""
    change_points = []

    if dataset_name in insects_datasets:
        change_points = insects_datasets[dataset_name]["change_point"]

    if dataset_name in drift_config:
        _, column, drifts = extract_drift_info(dataset_name)
        for drift_type in drifts:
            for drift in drifts[drift_type]:
                start_index = calculate_index(df, drift[0])
                end_index = calculate_index(df, drift[1])
                change_points.extend([start_index, end_index])

    return change_points


def plot_all_features(
    df: pd.DataFrame,
    dataset_name: str,
    drift_points: List[int] = None,
    suffix: str = "",
):
    """Plot all feature columns for a given dataset in individual subplots and save them."""
    # Exclude class column
    feature_columns = [col for col in df.columns if col != "class"]

    # Create a directory for the plots
    dataset_output_dir = os.path.join(output_dir, dataset_name, "feature_plots")
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Filter out drift points at index 0 and at the index with the size of the dataframe
    if drift_points:
        drift_points = [dp for dp in drift_points if dp != 0 and dp != len(df)]

    # Create subplots for each feature column in a single file
    num_features = len(feature_columns)
    fig, axes = plt.subplots(
        num_features, 1, figsize=(14, 7 * num_features), facecolor="white"
    )

    if num_features == 1:
        axes = [axes]

    for ax, column in zip(axes, feature_columns):
        ax.plot(df.index, df[column], label=column)
        if drift_points:
            for i, cp in enumerate(drift_points):
                if i == 0:
                    ax.axvline(
                        x=cp,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        label="Drift point",
                    )
                else:
                    ax.axvline(x=cp, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
        ax.set_title(f"{dataset_name} - {column}")
        ax.legend()
        ax.grid(True)

    # Save the combined plot
    combined_plot_path = os.path.join(
        dataset_output_dir, f"{dataset_name}_all_features{suffix}.png"
    )
    plt.tight_layout()
    plt.savefig(combined_plot_path, facecolor=fig.get_facecolor())
    plt.close()

    print(f"Combined feature plots saved to {combined_plot_path}")

    # Save each feature plot in a separate file
    for column in feature_columns:
        plt.figure(figsize=(14, 7), facecolor="white")
        plt.plot(df.index, df[column], label=column)
        if drift_points:
            for i, cp in enumerate(drift_points):
                if i == 0:
                    plt.axvline(
                        x=cp,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        label="Drift point",
                    )
                else:
                    plt.axvline(x=cp, color="red", linestyle="--", linewidth=1.5)
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.title(f"{dataset_name} - {column}")
        plt.legend()
        plt.grid(True)

        feature_plot_path = os.path.join(
            dataset_output_dir, f"{dataset_name}_{column}{suffix}.png"
        )
        plt.savefig(feature_plot_path, facecolor="white")
        plt.close()


# Generate synthetic dataset without drifts
# dataframe_size = 80000
# synthetic_df_no_drifts = create_synthetic_dataframe(dataframe_size)
#
# # Save the synthetic dataset without drifts
# save_synthetic_dataset(synthetic_df_no_drifts, "synthetic_dataset_no_drifts")
#
#
# # Generate synthetic dataset with parallel drifts
# (
#     synthetic_df_with_parallel_drifts,
#     parallel_drift_points,
#     parallel_drift_info,
# ) = generate_synthetic_dataset_with_drifts(
#     dataframe_size=80000,
#     features_with_drifts=["feature1", "feature3", "feature5"],
#     num_features=5,
#     loc=10,
#     scale=1,
#     seed=42,
#     scenario="parallel",
# )
#
# # Save the synthetic dataset with parallel drifts
# save_synthetic_dataset(
#     synthetic_df_with_parallel_drifts, "synthetic_dataset_with_parallel_drifts"
# )


def generate_synthetic_datasets():
    """Generate and save synthetic datasets with and without drifts."""
    dataframe_size = 80000

    # Generate synthetic dataset without drifts
    synthetic_df_no_drifts = create_synthetic_dataframe(dataframe_size)
    save_synthetic_dataset(synthetic_df_no_drifts, "synthetic_dataset_no_drifts")

    # Generate synthetic dataset with parallel drifts
    (
        synthetic_df_with_parallel_drifts,
        parallel_drift_points,
        parallel_drift_info,
    ) = generate_synthetic_dataset_with_drifts(
        dataframe_size=dataframe_size,
        features_with_drifts=["feature1", "feature3", "feature5"],
        batch_size=1000,  # Example batch size, adjust as needed
        num_features=5,
        loc=10,
        scale=1,
        seed=42,
        scenario="parallel",
    )
    save_synthetic_dataset(
        synthetic_df_with_parallel_drifts, "synthetic_dataset_with_parallel_drifts"
    )
    plot_all_features(
        synthetic_df_with_parallel_drifts,
        "synthetic_dataset_with_parallel_drifts",
        parallel_drift_points,
        suffix="_with_parallel_drifts",
    )

    # Generate synthetic dataset with switching drifts
    (
        synthetic_df_with_switching_drifts,
        switching_drift_points,
        switching_drift_info,
    ) = generate_synthetic_dataset_with_drifts(
        dataframe_size=dataframe_size,
        features_with_drifts=["feature1", "feature3", "feature5"],
        batch_size=1000,  # Example batch size, adjust as needed
        num_features=5,
        loc=10,
        scale=1,
        seed=42,
        scenario="switching",
    )
    save_synthetic_dataset(
        synthetic_df_with_switching_drifts, "synthetic_dataset_with_switching_drifts"
    )
    plot_all_features(
        synthetic_df_with_switching_drifts,
        "synthetic_dataset_with_switching_drifts",
        switching_drift_points,
        suffix="_with_switching_drifts",
    )


def run_full_experiment(max_batch_size: int):
    """Run full experiment for all datasets."""
    datasets = []

    # batch_sizes = [1000]
    # datasets = ["MULTISTAGGER", "MULTISEA"]

    # datasets = ["electricity"]

    batch_sizes = [1000, 1500, 2000, 2500]

    # "magic" --> fix the issue with connection and save the file on path
    # datasets = ["electricity", "magic", "MULTISTAGGER", "MULTISEA", "SEA", "STAGGER"]

    # for dataset in insects_datasets.keys():
    #     if dataset != "Out-of-control":
    #         datasets.append(dataset)

    # datasets.append("Incremental (bal.)")
    # datasets.append("Abrupt (imbal.)")
    #
    # for dataset in datasets_with_added_drifts:
    #     datasets.append(dataset)

    datasets = [
        "synthetic_dataset_no_drifts",
        "synthetic_dataset_with_parallel_drifts",
        "synthetic_dataset_with_switching_drifts",
    ]
    # batch_sizes = [1000, 2500]

    results = {dataset: {} for dataset in datasets}
    csv_file_paths = []

    for dataset in datasets:
        dataset_results = {}
        output_path = os.path.join(output_dir, f"{dataset}")
        os.makedirs(output_path, exist_ok=True)
        csv_file_path = os.path.join(output_path, f"{dataset}_results.csv")

        # Remove the file if it already exists
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        for batch_size in batch_sizes:
            if dataset == "synthetic_dataset_with_parallel_drifts":
                (
                    synthetic_df_with_parallel_drifts,
                    parallel_drift_points,
                    parallel_drift_info,
                ) = generate_synthetic_dataset_with_drifts(
                    dataframe_size=80000,
                    features_with_drifts=["feature1", "feature3", "feature5"],
                    batch_size=max_batch_size,
                    num_features=5,
                    loc=10,
                    scale=1,
                    seed=42,
                    scenario="parallel",
                )
                save_synthetic_dataset(
                    synthetic_df_with_parallel_drifts,
                    "synthetic_dataset_with_parallel_drifts",
                )
                plot_all_features(
                    synthetic_df_with_parallel_drifts,
                    "synthetic_dataset_with_parallel_drifts",
                    parallel_drift_points,
                    suffix="_with_parallel_drifts",
                )

            elif dataset == "synthetic_dataset_with_switching_drifts":
                (
                    synthetic_df_with_switching_drifts,
                    switching_drift_points,
                    switching_drift_info,
                ) = generate_synthetic_dataset_with_drifts(
                    dataframe_size=80000,
                    features_with_drifts=["feature1", "feature3", "feature5"],
                    batch_size=max_batch_size,
                    num_features=5,
                    loc=10,
                    scale=1,
                    seed=42,
                    scenario="switching",
                )
                save_synthetic_dataset(
                    synthetic_df_with_switching_drifts,
                    "synthetic_dataset_with_switching_drifts",
                )
                plot_all_features(
                    synthetic_df_with_switching_drifts,
                    "synthetic_dataset_with_switching_drifts",
                    switching_drift_points,
                    suffix="_with_switching_drifts",
                )

            print(f"{dataset} - {batch_size}")
            drift_results, test_results, X_shape = run_test(
                dataset=dataset,
                batch_size=batch_size,
                plot_heatmaps=True,
            )
            num_batches = X_shape // batch_size  # Calculate the number of batches
            dataset_results[batch_size] = test_results
            plot_drift_points(drift_results, dataset, batch_size)
            save_results_to_csv(
                dataset,
                batch_size,
                drift_results,
                test_results,
                num_batches,
                csv_file_path,
            )
            print()

        results[dataset] = dataset_results
        plot_results(dataset_results, dataset, batch_sizes)
        csv_file_paths.append(csv_file_path)

        # Load the dataset and plot all features
        df, _, _ = load_and_prepare_dataset(dataset)
        plot_all_features(df, dataset)

    # Consolidate all CSV files into a single CSV file
    target_csv_file = os.path.join(output_dir, "consolidated_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    consolidate_csv_files(csv_file_paths, target_csv_file)


if __name__ == "__main__":
    max_batch_size = 2500  # Set the maximum batch size
    run_full_experiment(max_batch_size)
