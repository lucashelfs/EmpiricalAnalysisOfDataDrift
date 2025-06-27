import os
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from codes.config import comparisons_output_dir as output_dir
from utils import fetch_dataset_change_points

from PIL import Image


warnings.filterwarnings("ignore", message="No artists with labels found")

# Set global style parameters
plt.rcParams["figure.facecolor"] = "#EAEAF2"  # Set the background color to grey
plt.rcParams["axes.facecolor"] = "#EAEAF2"  # Set the axes background color to grey
plt.rcParams[
    "savefig.facecolor"
] = "white"  # Set the saved figure background color to grey
plt.rcParams["grid.color"] = "white"  # Set the grid color to white
plt.rcParams["axes.grid"] = True  # Enable grid by default
plt.rcParams["legend.frameon"] = True  # Enable legend frame
plt.rcParams["legend.framealpha"] = 0.9  # Set legend frame transparency
plt.rcParams["legend.facecolor"] = "white"  # Set legend background color
plt.rcParams["legend.edgecolor"] = "black"  # Set legend edge color


def plot_legend_on_other_file(dataset, batch_size):
    # Create a separate legend figure
    fig_legend = plt.figure(figsize=(6, 1))  # Adjust size as needed
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")  # Hide axes

    # Create the legend
    legend = ax_legend.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="r",
                markersize=10,
                label="KS95",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="g",
                markersize=10,
                label="KS90",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="b",
                markersize=10,
                label="HD",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="m",
                markersize=10,
                label="JS",
            ),
            plt.Line2D([0], [0], linestyle="--", color="k", label="Change point"),
        ],
        loc="center",
        ncol=5,  # Arrange in a single row
        fontsize=15,
        frameon=False,  # No legend box
    )

    # Save the legend as an image
    legend_path = os.path.join(output_dir, f"{dataset}/detected_drifts/legend.png")
    fig_legend.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_legend)


def map_label_names(method):
    if method == "KS95":
        return "KSDDM 95"
    if method == "KS90":
        return "KSDDM 90"
    if method == "HD":
        return "HDDDM"
    if method == "JS":
        return "JSDDM"


def plot_drift_points(
    drift_results: Dict[str, List[int]],
    dataset: str,
    batch_size: int,
    synthetic_drift_points: dict = None,
    drift_alignment_within_batch: float = None,
    max_index: int = None,
):
    """Plot drift points."""
    os.makedirs(output_dir + f"/{dataset}/detected_drifts/", exist_ok=True)
    plt.figure(figsize=(12, 8))
    colors = {"KS95": "r", "KS90": "g", "HD": "b", "JS": "m"}

    # Ensure all methods are on the Y-axis
    methods = ["KS95", "KS90", "HD", "JS"]
    for method in methods:
        drifts = drift_results.get(method, [])

        # Remove the first batch, which is the start of the detections
        drifts = [x for x in drifts if x != 1]

        plt.scatter(
            drifts,
            [method] * len(drifts),
            color=colors[method],
            label=method if len(drifts) > 0 else None,
            s=50,
        )

    # Initialize change_points and feature_regions
    change_points = []
    feature_regions = []

    if not synthetic_drift_points:
        change_points = fetch_dataset_change_points(dataset, batch_size)
    else:
        for feature, feature_drifts in synthetic_drift_points.items():
            for start, end in feature_drifts:
                start_batch = (start // batch_size) + 1
                end_batch = (end // batch_size) + 1
                change_points.extend([start_batch, end_batch])
                feature_regions.append((start_batch, end_batch, feature))

    # Sort and remove duplicate change points
    change_points = sorted(set(change_points))

    label_added = False
    for cp in change_points:
        if not label_added:
            plt.axvline(
                x=cp, color="k", linestyle="--", linewidth=1, label="Change point"
            )
            label_added = True
        else:
            plt.axvline(x=cp, color="k", linestyle="--", linewidth=1)

    # Adjust title position if feature labels exist
    title_padding = 20 if not feature_regions else 35

    if dataset == "synthetic_dataset_with_switching_drifts_incremental":
        main_title = f"Drift Points for SYN-SI (Batch Size: {batch_size})"
    elif dataset == "synthetic_dataset_with_switching_drifts_abrupt":
        main_title = f"Drift Points for SYN-SA (Batch Size: {batch_size})"
    elif dataset == "synthetic_dataset_with_parallel_drifts_incremental":
        main_title = f"Drift Points for SYN-PI (Batch Size: {batch_size})"
    elif dataset == "synthetic_dataset_with_parallel_drifts_abrupt":
        main_title = f"Drift Points for SYN-PA (Batch Size: {batch_size})"
    else:
        main_title = f"Drift Points for {dataset} (Batch Size: {batch_size})"

    plt.title(main_title, fontsize=20, pad=title_padding)

    # Plot feature labels **above** drift regions
    if feature_regions:
        from collections import defaultdict

        # Group features by (start, end)
        grouped_regions = defaultdict(list)
        for start, end, feature in feature_regions:
            feature_number = feature.replace("feature", "")  # Extract number only
            grouped_regions[(start, end)].append(feature_number)

        # Create the final list with combined feature names
        merged_feature_regions = [
            (
                start,
                end,
                f"F: {', '.join(features)} - ({start}, {end})"
                if len(features) == 1
                else f"F: {', '.join(features)} - ({start}, {end})",
            )
            for (start, end), features in grouped_regions.items()
        ]

        y_max = plt.gca().get_ylim()[1]  # Get max y-axis value for placement
        y_label_pos = y_max + (0.005 * y_max)  # Position labels above the plot

        for start, end, feature in merged_feature_regions:
            mid_point = (start + end) / 2  # Center label between change points
            plt.text(
                mid_point,
                y_label_pos,
                feature,
                fontsize=14,
                color="black",
                ha="center",
                va="bottom",
            )

    plt.xlabel("Batch Index", fontsize=20)
    plt.grid(True)
    plt.yticks(ticks=methods, labels=methods, fontsize=15)

    lower_index = 0
    if max_index is not None:
        plt.xlim(lower_index, max_index)
    else:
        plt.xlim(lower_index, None)  # for non synthetics

    plt.xticks(fontsize=15)  # breakpoint here

    # Set integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Set x-axis ticks to start at 2
    current_ticks = plt.gca().get_xticks()
    new_ticks = [2] + [tick for tick in current_ticks if tick >= 2]
    plt.gca().set_xticks(new_ticks)

    if drift_alignment_within_batch:
        filename = (
            f"{dataset}_{batch_size}_{drift_alignment_within_batch}_drift_points.png"
        )
    else:
        filename = f"{dataset}_{batch_size}_drift_points.png"

    plt.savefig(
        os.path.join(
            output_dir + f"/{dataset}/detected_drifts/",
            filename,
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
            added_labels = set()  # Track labels that have been added to the legend
            for drift_type, start_index, end_index in drift_info[column]:
                # Always plot the lines and rectangles
                ax.axvline(
                    x=start_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label="drift limit" if "drift limit" not in added_labels else None,
                )
                ax.axvline(
                    x=end_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=None,  # No label for the end line to avoid duplication
                )
                ax.add_patch(
                    patches.Rectangle(
                        (start_index, ax.get_ylim()[0]),
                        end_index - start_index,
                        ax.get_ylim()[1] - ax.get_ylim()[0],
                        color="yellow",
                        alpha=0.3,
                        label=f"{drift_type} drift"
                        if f"{drift_type} drift" not in added_labels
                        else None,
                    )
                )
                # Add labels to the set to avoid duplicates in the legend
                added_labels.add("drift limit")
                added_labels.add(f"{drift_type} drift")

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

        formatted_title = convert_dataset_name_for_plots(dataset_name)
        ax.set_title(f"{formatted_title} - {column}")

        # Make dataset name properly formatted
        ax.legend()  # Ensure the legend is associated with the primary axis (ax)
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
            added_labels = set()  # Track labels that have been added to the legend
            for drift_type, start_index, end_index in drift_info[column]:
                # Always plot the lines and rectangles
                plt.axvline(
                    x=start_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label="drift limit" if "drift limit" not in added_labels else None,
                )
                plt.axvline(
                    x=end_index,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=None,  # No label for the end line to avoid duplication
                )
                plt.gca().add_patch(
                    patches.Rectangle(
                        (start_index, plt.ylim()[0]),
                        end_index - start_index,
                        plt.ylim()[1] - plt.ylim()[0],
                        color="yellow",
                        alpha=0.3,
                        label=f"{drift_type} drift"
                        if f"{drift_type} drift" not in added_labels
                        else None,
                    )
                )
                # Add labels to the set to avoid duplicates in the legend
                added_labels.add("drift limit")
                added_labels.add(f"{drift_type} drift")

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

        plt.ylabel(column)
        formatted_title = convert_dataset_name_for_plots(dataset_name)
        plt.title(f"{formatted_title} - {column}")
        plt.legend()  # Ensure the legend is associated with the primary axis
        plt.grid(True)

        if use_batch_numbers:
            ax2 = plt.gca().twiny()
            ax2.set_xbound(ax.get_xbound())

            # Calculate batch numbers based on the index
            batch_ticks = [i for i in range(0, len(df) + 1, batch_size)]
            batch_labels = [i // batch_size for i in batch_ticks]

            # Slice the batch_ticks and batch_labels to include only every 10th batch
            batch_ticks_sliced = batch_ticks[::10]  # Every 10th batch tick
            batch_labels_sliced = batch_labels[::10]  # Every 10th batch label

            # Set the sliced ticks and labels
            ax2.set_xticks(batch_ticks_sliced)
            ax2.set_xticklabels(batch_labels_sliced)
            ax2.set_xlabel("Batch Index")
        else:
            plt.xlabel("Index")

        feature_plot_path = os.path.join(
            dataset_output_dir, f"{dataset_name}_{column}{suffix}.png"
        )
        plt.savefig(feature_plot_path, facecolor="white")
        plt.close()


def convert_dataset_name_for_plots(dataset: str):
    if dataset == "synthetic_dataset_with_switching_drifts_abrupt":
        return "Synth. Switching Abrupt"
    elif dataset == "synthetic_dataset_with_switching_drifts_incremental":
        return "Synth. Switching Incremental"
    elif dataset == "synthetic_dataset_with_parallel_drifts_abrupt":
        return "Synth. Parallel Abrupt"
    elif dataset == "synthetic_dataset_with_parallel_drifts_incremental":
        return "Synth. Parallel Incremental"
    else:
        return dataset


def plot_feature_and_its_variations(
    dataset_name: str,
    column: str,
    suffix: str,
):
    # Create a directory for the plots
    dataset_output_dir = os.path.join(output_dir, dataset_name, "feature_plots")
    os.makedirs(dataset_output_dir, exist_ok=True)

    concated_features_plot = os.path.join(
        dataset_output_dir, f"{dataset_name}_{column}{suffix}_variations.png"
    )

    # Find all image files that match the pattern
    image_files = [
        os.path.join(dataset_output_dir, f)
        for f in os.listdir(dataset_output_dir)
        if f.startswith(f"{dataset_name}_{column}{suffix}") and f.endswith(".png")
    ]

    image_files = [file for file in image_files if not file.endswith("_variations.png")]

    print(f"Images being used for concatenation: {image_files}")

    # Load all images
    images = [Image.open(img_file) for img_file in image_files]

    # Calculate the total height and maximum width for the concatenated image
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    # Create a new blank image with the calculated dimensions
    concatenated_image = Image.new("RGB", (max_width, total_height))

    # Paste each image into the concatenated image
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the concatenated image
    concatenated_image.save(concated_features_plot)
    print(f"Concatenated image saved to {concated_features_plot}")
