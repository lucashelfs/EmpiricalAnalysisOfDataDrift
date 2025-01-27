import os
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from codes.config import comparisons_output_dir as output_dir
from utils import fetch_dataset_change_points

import warnings

warnings.filterwarnings("ignore", message="No artists with labels found")


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


def plot_all_features(
    df: pd.DataFrame,
    dataset_name: str,
    drift_points: List[int] = None,
    drift_info: Dict[str, List[Tuple[str, int, int]]] = None,
    suffix: str = "",
    batch_size: int = 1000,
    use_batch_numbers: bool = False,
):
    """Plot all feature columns for a given dataset in individual subplots and save them."""
    # Exclude class column
    feature_columns = [col for col in df.columns if col != "class"]

    # Create a directory for the plots
    dataset_output_dir = os.path.join(output_dir, dataset_name, "feature_plots")
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Create subplots for each feature column in a single file
    num_features = len(feature_columns)
    fig, axes = plt.subplots(
        num_features, 1, figsize=(14, 7 * num_features), facecolor="white"
    )

    if num_features == 1:
        axes = [axes]

    for ax, column in zip(axes, feature_columns):
        ax.plot(df.index, df[column], label=column)
        if drift_info and column in drift_info:
            for drift_type, start_index, end_index in drift_info[column]:
                ax.axvline(
                    x=start_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{drift_type} start",
                )
                ax.axvline(
                    x=end_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{drift_type} end",
                )
                ax.add_patch(
                    patches.Rectangle(
                        (start_index, ax.get_ylim()[0]),
                        end_index - start_index,
                        ax.get_ylim()[1] - ax.get_ylim()[0],
                        color="yellow",
                        alpha=0.3,
                        label=f"{drift_type} drift",
                    )
                )
                ax.text(
                    (start_index + end_index) / 2,
                    ax.get_ylim()[1],
                    f"{drift_type} drift",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=10,
                    color="black",
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )
        if use_batch_numbers:
            # Create secondary x-axis for batch numbers
            ax2 = ax.twiny()
            ax2.set_xticks(ax.get_xticks())
            ax2.set_xbound(ax.get_xbound())
            ax2.set_xticklabels([int(x // batch_size) for x in ax.get_xticks()])
            ax2.set_xlabel("Batch Index")
        else:
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

    # Save each feature plot in a separate file
    for column in feature_columns:
        plt.figure(figsize=(14, 7), facecolor="white")
        plt.plot(df.index, df[column], label=column)
        if drift_info and column in drift_info:
            for drift_type, start_index, end_index in drift_info[column]:
                plt.axvline(
                    x=start_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{drift_type} start",
                )
                plt.axvline(
                    x=end_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{drift_type} end",
                )
                plt.gca().add_patch(
                    patches.Rectangle(
                        (start_index, plt.ylim()[0]),
                        end_index - start_index,
                        plt.ylim()[1] - plt.ylim()[0],
                        color="yellow",
                        alpha=0.3,
                        label=f"{drift_type} drift",
                    )
                )
                plt.text(
                    (start_index + end_index) / 2,
                    plt.ylim()[1],
                    f"{drift_type} drift",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=10,
                    color="black",
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )
        if use_batch_numbers:
            # Create secondary x-axis for batch numbers
            ax2 = plt.gca().twiny()
            ax2.set_xticks(plt.gca().get_xticks())
            ax2.set_xbound(plt.gca().get_xbound())
            ax2.set_xticklabels([int(x // batch_size) for x in plt.gca().get_xticks()])
            ax2.set_xlabel("Batch Index")
        else:
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
