import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from codes.common import load_and_prepare_dataset, find_indexes
from codes.config import comparisons_output_dir as output_dir

from codes.KSDDM import initialize_ksddm, process_batches
from codes.HDDDM import HDDDM, JSDDM


def define_batches(X, batch_size):
    X["Batch"] = (X.index // batch_size) + 1
    return X


def plot_heatmap(
    technique, heatmap_data, dataset, batch_size, change_points=None, suffix=""
):
    os.makedirs(output_dir + f"/{dataset}/heatmaps/", exist_ok=True)
    sns.set(rc={"figure.figsize": (12, 8)})
    grid_kws = {"height_ratios": (0.9, 0.05), "hspace": 0.3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)

    coloring = sns.cubehelix_palette(
        start=0.8,
        rot=-0.5,
        as_cmap=True,
        reverse=True if "KSDDM" in technique else False,
    )

    min_value = 0 if "KSDDM" in technique else None
    max_value = 1

    if "Chunked" in suffix:
        # Little epsilon correction to avoid rounding and missing values
        if "95" in technique:
            max_value = float(0.05 / heatmap_data.shape[0])
        if "90" in technique:
            max_value = float(0.1 / heatmap_data.shape[0])

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=coloring,
        vmin=min_value,
        vmax=max_value,
        xticklabels=heatmap_data.columns,
        yticklabels=heatmap_data.index,
        linewidths=0.5,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
    )

    if change_points:
        for i, cp in enumerate(change_points):
            batch_number = cp // batch_size
            ax.axvline(
                x=batch_number + 0.5,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Change Point" if i == 0 else "",
            )

    ax.set_title(
        f"{technique} - Heatmap for dataset {dataset} with batch size {batch_size}",
        fontsize=20,
    )

    ax.set(xlabel="Batch", ylabel="Features")

    label_text = (
        "P-values of test between batch and reference"
        if "KSDDM" in technique
        else "Distance between batch and reference"
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
    ax.collections[0].colorbar.set_label(label=label_text, fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)

    # Set x-ticks to skip some values for readability
    max_xticks = 20  # Adjust this value as needed

    if len(heatmap_data.columns) > max_xticks:
        xticks = list(
            range(
                0,
                len(heatmap_data.columns),
                len(heatmap_data.columns) // max_xticks + 1,
            )
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [heatmap_data.columns[i] for i in xticks],
            rotation=45,
            ha="right",
            fontsize=15,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=15)
    ax.set_ylabel("Features", fontsize=20)
    ax.set_xlabel("Batch Index", fontsize=20)

    if "Chunked" in suffix:
        filename = os.path.join(
            output_dir + f"/{dataset}/heatmaps/",
            f"{batch_size}_{technique}_chunked_heatmap.png",
        )
    else:
        filename = os.path.join(
            output_dir + f"/{dataset}/heatmaps/",
            f"{batch_size}_{technique}_heatmap.png",
        )

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def add_drift_spans(ax, plot_data):
    for i, t in enumerate(
        plot_data.loc[plot_data["Detected Drift"] == "drift"]["Batch"]
    ):
        ax.axvspan(
            t - 0.2,
            t + 0.2,
            alpha=0.5,
            color="red",
            label=("Drift Detected" if i == 0 else None),
        )


def fetch_ksddm_drifts(
    batch_size=1000,
    dataset=None,
    mean_threshold=0.05,
    plot_heatmaps=False,
    text="KSDDM",
):
    if dataset is None:
        return

    reference_batch = 1
    X, _, _ = load_and_prepare_dataset(dataset)
    X = define_batches(X, batch_size)

    reference = X[X.Batch == reference_batch].iloc[:, :-1]
    all_test = X[X.Batch != reference_batch]

    ksddm = initialize_ksddm(reference, mean_threshold=mean_threshold)
    heatmap_data, _, detected_drift = process_batches(ksddm, X, reference_batch)

    plot_data = pd.DataFrame(
        {"Batch": all_test.Batch.unique(), "Detected Drift": ksddm.drift_state}
    )

    if plot_heatmaps:
        drift_list = find_indexes(plot_data["Detected Drift"])
        plot_heatmap(text, heatmap_data, dataset, batch_size)
        plot_heatmap(text, heatmap_data, dataset, batch_size, suffix="Chunked")

    return plot_data["Detected Drift"]


def fetch_hdddm_drifts(
    batch_size=1000, statistic="stdev", dataset=None, plot_heatmaps=False
):
    if dataset is not None:
        reference_batch = 1
        X, _, dataset_filename_str = load_and_prepare_dataset(dataset)
        X = define_batches(X, batch_size)

        reference = X[X.Batch == reference_batch].iloc[:, :-1]
        all_test = X[X.Batch != reference_batch]

        hdddm = HDDDM(statistic=statistic, significance=1, detect_batch=1)
        batches = all_test.Batch.unique()
        heatmap_data = pd.DataFrame(columns=batches)
        detected_drift = []

        # Run HDDDM
        hdddm.set_reference(reference)
        for batch, subset_data in X[X.Batch != reference_batch].groupby("Batch"):
            hdddm.update(subset_data.iloc[:, :-1])
            heatmap_data[batch] = hdddm.feature_distances
            detected_drift.append(hdddm.drift_state)

        # Plot data: Hellinger distance for each batch with detected drift
        plot_data = pd.DataFrame(
            {
                "Batch": batches,
                "Detected Drift": detected_drift,
            }
        )

        if plot_heatmaps:
            drift_list = find_indexes(plot_data["Detected Drift"])
            plot_heatmap("HDDDM", heatmap_data, dataset, batch_size, change_points=None)

        return plot_data["Detected Drift"]


def fetch_jsddm_drifts(
    batch_size=1000, statistic="stdev", dataset=None, plot_heatmaps=False
):
    if dataset is not None:
        X, _, _ = load_and_prepare_dataset(dataset=dataset)
        reference_batch = 1
        X["Batch"] = (X.index // batch_size) + 1

        reference = X[X.Batch == reference_batch].iloc[:, :-1]
        all_test = X[X.Batch != reference_batch]

        jsddm = JSDDM(statistic=statistic, significance=1, detect_batch=1)
        batches = all_test.Batch.unique()
        heatmap_data = pd.DataFrame(columns=batches)
        detected_drift = []

        # Run jsddm
        jsddm.set_reference(reference)
        for batch, subset_data in X[X.Batch != reference_batch].groupby("Batch"):
            jsddm.update(subset_data.iloc[:, :-1])
            heatmap_data[batch] = jsddm.feature_distances
            detected_drift.append(jsddm.drift_state)

        # Plot data: Jensen Shannon for each batch with detected drift
        plot_data = pd.DataFrame(
            {
                "Batch": batches,
                "Detected Drift": detected_drift,
            }
        )

        if plot_heatmaps:
            drift_list = find_indexes(plot_data["Detected Drift"])
            plot_heatmap("JSDDM", heatmap_data, dataset, batch_size, change_points=None)

        return plot_data["Detected Drift"]
