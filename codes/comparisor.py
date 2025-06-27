import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
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
from codes.config import comparisons_output_dir as output_dir
from codes.ddm import fetch_hdddm_drifts, fetch_jsddm_drifts, fetch_ksddm_drifts
from codes.drift_generation import (
    generate_synthetic_dataset_with_drifts,
    plot_accumulated_differences,
    save_synthetic_dataset,
)
from codes.plots import (
    plot_all_features,
    plot_results,
    plot_drift_points,
    plot_feature_and_its_variations,
)

from river import tree


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


def run_prequential_hoeffding_tree(
    dataset: str,
    batch_size: int = 1000,
    batches_with_drift_list: Optional[List[str]] = None,
):
    # Paper: https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf
    X, Y, _ = load_and_prepare_dataset(dataset)
    X = define_batches(X=X, batch_size=batch_size)
    Y = pd.DataFrame(Y, columns=["class"])
    Y = define_batches(X=Y, batch_size=batch_size)

    reference_batch = 1
    reference = X[X.Batch == reference_batch].iloc[:, :-1]
    y_reference = Y[Y.Batch == reference_batch].iloc[:, :-1].values.ravel()

    model = tree.HoeffdingTreeClassifier()

    # Train on reference batch
    for x, y in zip(reference.to_dict(orient="records"), y_reference):
        model.learn_one(x, y)

    # Test and store predictions for the reference batch
    y_pred = [model.predict_one(x) for x in reference.to_dict(orient="records")]
    batch_predictions = y_pred.copy()

    batches = list(set(X.Batch) - {reference_batch})
    drift_indexes = []

    if batches_with_drift_list is not None:
        drift_indexes = [
            index + 1  # Adjust index to align with batch numbering
            for index, value in enumerate(batches_with_drift_list)
            if value == "drift"
        ]

    for batch in batches:
        X_batch = X[X.Batch == batch].iloc[:, :-1]
        Y_batch = Y[Y.Batch == batch].iloc[:, :-1].values.ravel()

        # Test
        y_pred = [model.predict_one(x) for x in X_batch.to_dict(orient="records")]
        batch_predictions.extend(y_pred)

        # Train
        if batches_with_drift_list is not None and batch in drift_indexes:
            # Reset the model due to detected drift
            model = tree.HoeffdingTreeClassifier()
            for x, y in zip(X_batch.to_dict(orient="records"), Y_batch):
                model.learn_one(x, y)
        else:
            # Incremental learning without reset
            for x, y in zip(X_batch.to_dict(orient="records"), Y_batch):
                model.learn_one(x, y)

    return X, Y, batch_predictions


def fetch_all_drifts(
    batch_size,
    dataset,
    drift_alignment_within_batch: Optional[float] = None,
    plot_heatmaps: bool = True,
):
    hd_drifts = fetch_hdddm_drifts(
        batch_size=batch_size,
        plot_heatmaps=plot_heatmaps,
        dataset=dataset,
        drift_alignment_within_batch=drift_alignment_within_batch,
    )

    ks_drifts = fetch_ksddm_drifts(
        batch_size=batch_size,
        dataset=dataset,
        mean_threshold=0.05,
        plot_heatmaps=plot_heatmaps,
        text="KSDDM 95",
        drift_alignment_within_batch=drift_alignment_within_batch,
    )

    ks_90_drifts = fetch_ksddm_drifts(
        batch_size=batch_size,
        dataset=dataset,
        plot_heatmaps=plot_heatmaps,
        mean_threshold=0.10,
        text="KSDDM 90",
        drift_alignment_within_batch=drift_alignment_within_batch,
    )

    js_drifts = fetch_jsddm_drifts(
        batch_size=batch_size,
        plot_heatmaps=plot_heatmaps,
        dataset=dataset,
        drift_alignment_within_batch=drift_alignment_within_batch,
    )

    return {
        "ks_drifts": ks_drifts,
        "ks_90_drifts": ks_90_drifts,
        "hd_drifts": hd_drifts,
        "js_drifts": js_drifts,
    }


def run_test(
    dataset: str,
    batch_size: int = 1000,
    plot_heatmaps: bool = True,
    algorithm: str = "NB",
    drift_alignment_within_batch: Optional[float] = None,
    detected_drifts_dict: dict = None,
):
    """Runs tests on the dataset using multiple drift detection methods."""

    # This is where we get the info...

    if algorithm == "NB":
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

    if algorithm == "HT":
        X, Y, batch_predictions_base = run_prequential_hoeffding_tree(
            dataset=dataset, batch_size=batch_size
        )
        X, Y, batch_predictions_ks = run_prequential_hoeffding_tree(
            dataset=dataset,
            batch_size=batch_size,
            batches_with_drift_list=detected_drifts_dict["ks_drifts"],
        )
        X, Y, batch_predictions_ks_90 = run_prequential_hoeffding_tree(
            dataset=dataset,
            batch_size=batch_size,
            batches_with_drift_list=detected_drifts_dict["ks_90_drifts"],
        )
        X, Y, batch_predictions_hd = run_prequential_hoeffding_tree(
            dataset=dataset,
            batch_size=batch_size,
            batches_with_drift_list=detected_drifts_dict["hd_drifts"],
        )
        X, Y, batch_predictions_js = run_prequential_hoeffding_tree(
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
    metrics_results: dict[Any, dict[str, float | tuple[Any, Any, float]]],
    num_batches: int,
    csv_file_path: str,
    drift_alignment_with_batch: float = "N/A",
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
                "drift_alignment_with_batch": drift_alignment_with_batch,
                "scenario": scenario,
                "type_of_dataset": type_of_dataset,
                "algorithm": algorithm,
            }
        )

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Append the DataFrame to the CSV file
    df.to_csv(
        csv_file_path, mode="a", index=False, header=not os.path.exists(csv_file_path)
    )


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


def prepare_datasets():
    """Prepare the list of datasets to be processed."""
    return [
        "synthetic_dataset_with_parallel_drifts_abrupt",
        "synthetic_dataset_with_switching_drifts_incremental",
        "synthetic_dataset_with_parallel_drifts_incremental",
        "synthetic_dataset_with_switching_drifts_abrupt",
        "synthetic_dataset_no_drifts",
        # Concept drift datasets below
        "MULTISTAGGER",
        "MULTISEA",
        "SEA",
        "STAGGER",
        "electricity",
        "magic",
        "Abrupt (imbal.)",
        "Abrupt (bal.)",
        "Incremental (bal.)",
        "Incremental (imbal.)",
        "Incremental-gradual (bal.)",
        "Incremental-gradual (imbal.)",
        "Incremental-abrupt-reoccurring (bal.)",
        "Incremental-abrupt-reoccurring (imbal.)",
        "Incremental-reoccurring (bal.)",
        "Incremental-reoccurring (imbal.)",
    ]


def prepare_output_path(dataset):
    """Create and return the output directory for a dataset."""
    output_path = os.path.join(output_dir, f"{dataset}")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def handle_synthetic_dataset(
    scenario,
    dataset,
    dataframe_size,
    batch_size,
    drift_within_batch: float = 1.0,
    features_with_drifts: list[str] = None,
    num_drifts: int = 2,
):
    """Generate, save, and plot synthetic datasets."""
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
        seed=42,
        scenario=scenario,
        num_drifts=num_drifts,
    )
    save_synthetic_dataset(synthetic_df, dataset)
    plot_all_features(
        synthetic_df,
        dataset,
        drift_points,
        suffix=f"_{scenario}_drifts_{batch_size}_{drift_within_batch}",
        drift_info=drift_info,
    )
    plot_all_features(
        synthetic_df,
        dataset,
        drift_points,
        suffix=f"_{scenario}_drifts_and_batches_{batch_size}_{drift_within_batch}",
        drift_info=drift_info,
        batch_size=batch_size,
        use_batch_numbers=True,
    )
    plot_feature_and_its_variations(
        dataset_name=dataset,
        column="feature1",
        suffix=f"_{scenario}_drifts_{batch_size}_",
    )

    if scenario != "no_drifts":
        # TODO: this must be on other place, we need to plot differences after the drifts are detected
        # TODO: This plot breaks when we use batch size of 1500, needs to be fixed but dropped for now
        # plot_accumulated_differences(
        #     accumulated_differences,
        #     features_with_drifts,
        #     dataset,
        #     batch_size=batch_size,
        # )
        pass

    return synthetic_df, accumulated_differences, drift_points


import json
import os
from typing import List


def concatenate_json_files(
    json_file_paths: List[str],
    output_filename: str = "consolidated_drift_results.json",
):
    """Concatenate multiple JSON files into a single JSON file without merging their contents."""
    all_jsons = []

    for file_path in json_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_jsons.append(data)  # Append each JSON object to the list
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing {file_path}: {e}")

    os.makedirs(output_dir, exist_ok=True)
    target_file = os.path.join(output_dir, output_filename)

    with open(target_file, "w", encoding="utf-8") as output_file:
        json.dump(all_jsons, output_file, indent=4)

    print(f"JSON files concatenated and saved to {target_file}")


def save_drift_points_to_file(
    synthetic_drift_points,
    detected_drifts_dict,
    batch_size,
    dataset_name,
    drift_within_batch,
):
    # Step 1: Process synthetic drift points
    synthetic_drift_batches = {}
    for feature, drift_ranges in synthetic_drift_points.items():
        batch_ranges = []
        for start, end in drift_ranges:
            batch_ranges.extend(
                range(start // batch_size + 1, (end // batch_size) + 2)
            )  # can change this to avoid range, depending on the dataset name
        synthetic_drift_batches[feature] = sorted(set(batch_ranges))

    # Step 2: Process detected drift points
    detected_drift_batches = {}
    for method, drifts in detected_drifts_dict.items():
        detected_drift_batches[method] = [
            pair[0] + 2 for pair in enumerate(drifts) if pair[1] == "drift"
        ]

    # Step 3: Organize data
    drift_data = {
        "dataset": dataset_name,
        "drift_within_batch": drift_within_batch,
        "batch_size": batch_size,
        "synthetic_drifts": synthetic_drift_batches,
        "detected_drifts": detected_drift_batches,
    }

    dataset_output_dir = os.path.join(output_dir, dataset_name, "drift_files")
    os.makedirs(dataset_output_dir, exist_ok=True)

    output_file = (
        dataset_output_dir + f"/drift_analysis_{batch_size}_{drift_within_batch}.json"
    )

    # Step 4: Save to JSON file
    with open(output_file, "w") as f:
        json.dump(drift_data, f, indent=4)

    print(f"Drift data saved to {output_file}")

    return output_file


def run_single_experiment(
    dataset,
    batch_size,
    algorithm,
    drift_alignment_within_batch=None,
    detected_drifts_dict=None,
):
    """Run the main experiment pipeline for a single dataset and batch size."""

    drift_results, test_results, X_shape = run_test(
        dataset=dataset,
        batch_size=batch_size,
        plot_heatmaps=True,
        algorithm=algorithm,
        drift_alignment_within_batch=drift_alignment_within_batch,
        detected_drifts_dict=detected_drifts_dict,
    )
    num_batches = X_shape // batch_size
    return test_results, drift_results, num_batches


def consolidate_results(csv_file_paths):
    """Consolidate individual CSV files into a single CSV."""
    target_csv_file = os.path.join(output_dir, "consolidated_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    consolidate_csv_files(csv_file_paths, target_csv_file)


def run_full_experiment():
    """Run the full experiment pipeline."""

    results = {}
    csv_file_paths = []
    json_file_paths = []

    datasets = prepare_datasets()
    dataframe_size = 80000
    num_drifts = 2
    features_with_drifts = ["feature1", "feature3", "feature5"]

    batch_sizes = [
        1000,
        1500,
        2000,
        2500,
    ]

    # drift_alignment_batch_percentages = [0.5, 1.0, 0.05]
    drift_alignment_batch_percentages = [1.0]
    algorithms = ["NB"]
    # algorithms = ["NB", "HT"]

    for dataset in datasets:
        dataset_results = {}

        # Clear old csvs
        output_path = prepare_output_path(dataset)
        csv_file_path = os.path.join(output_path, f"{dataset}_results.csv")

        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        for algorithm in algorithms:
            print(f"Algorithm - {algorithm}")

            for batch_size in batch_sizes:
                if dataset.startswith("synthetic_"):
                    type_of_dataset = "synthetic"

                    if dataset == "synthetic_dataset_no_drifts":
                        scenario = "no_drifts"
                        (
                            synthetic_df,
                            accumulated_differences,
                            synthetic_drift_points,
                        ) = handle_synthetic_dataset(
                            scenario,
                            dataset,
                            dataframe_size,
                            batch_size,
                            features_with_drifts=[],
                        )

                        print(f"{dataset} - {batch_size}")

                        detected_drifts_dict = fetch_all_drifts(
                            batch_size,
                            dataset,
                            drift_alignment_within_batch=None,
                            plot_heatmaps=True,
                        )

                        (
                            test_results,
                            drift_results,
                            num_batches,
                        ) = run_single_experiment(
                            dataset,
                            batch_size,
                            algorithm,
                            detected_drifts_dict=detected_drifts_dict,
                        )
                        plot_drift_points(
                            drift_results,
                            dataset,
                            batch_size,
                            synthetic_drift_points=synthetic_drift_points,
                            max_index=int(dataframe_size / batch_size),
                        )

                        save_results_to_csv(
                            dataset,
                            batch_size,
                            drift_results,
                            test_results,
                            num_batches,
                            csv_file_path,
                            scenario=scenario,
                            type_of_dataset=type_of_dataset,
                            algorithm=algorithm,
                        )

                        dataset_results[batch_size] = test_results

                    else:
                        for drift_within_batch in drift_alignment_batch_percentages:
                            scenario = "N/A"
                            synthetic_df, accumulated_differences = None, None
                            if dataset.startswith(
                                "synthetic_dataset_with_parallel_drifts"
                            ):
                                if "abrupt" in dataset:
                                    scenario = "parallel_abrupt"
                                else:
                                    scenario = "parallel_incremental"

                                (
                                    synthetic_df,
                                    accumulated_differences,
                                    synthetic_drift_points,
                                ) = handle_synthetic_dataset(
                                    scenario,
                                    dataset,
                                    dataframe_size,
                                    batch_size,
                                    drift_within_batch,
                                    features_with_drifts,
                                    num_drifts=num_drifts,
                                )

                            elif dataset.startswith(
                                "synthetic_dataset_with_switching_drifts"
                            ):
                                if "abrupt" in dataset:
                                    scenario = "switching_abrupt"
                                else:
                                    scenario = "switching_incremental"

                                (
                                    synthetic_df,
                                    accumulated_differences,
                                    synthetic_drift_points,
                                ) = handle_synthetic_dataset(
                                    scenario,
                                    dataset,
                                    dataframe_size,
                                    batch_size,
                                    drift_within_batch,
                                    features_with_drifts,
                                    num_drifts=num_drifts,
                                )

                            print(f"{dataset} - {batch_size} - {drift_within_batch}")

                            detected_drifts_dict = fetch_all_drifts(
                                batch_size,
                                dataset,
                                drift_alignment_within_batch=drift_within_batch,
                                plot_heatmaps=True,
                            )

                            # TODO: Generate dataset synthetic_drift_points, detected_drifts_dict
                            output_file_drifts_analysis_file = (
                                save_drift_points_to_file(
                                    synthetic_drift_points,
                                    detected_drifts_dict,
                                    batch_size,
                                    dataset,
                                    drift_within_batch,
                                )
                            )

                            json_file_paths.append(output_file_drifts_analysis_file)

                            (
                                test_results,
                                drift_results,
                                num_batches,
                            ) = run_single_experiment(
                                dataset,
                                batch_size,
                                algorithm,
                                drift_alignment_within_batch=drift_within_batch,
                                detected_drifts_dict=detected_drifts_dict,
                            )
                            plot_drift_points(
                                drift_results,
                                dataset,
                                batch_size,
                                synthetic_drift_points=synthetic_drift_points,
                                drift_alignment_within_batch=drift_within_batch,
                                max_index=int(dataframe_size / batch_size),
                            )

                            # TODO: fix the plot acc diff stuff
                            # Plot accumulated distances until each drift detection, for all techniques
                            # plot_accumulated_differences(
                            #     accumulated_differences,
                            #     features_with_drifts,
                            #     dataset,
                            #     batch_size=batch_size,
                            #     detected_drifts=drift_results,
                            #     drift_within_batch=drift_within_batch,
                            # )

                            save_results_to_csv(
                                dataset,
                                batch_size,
                                drift_results,
                                test_results,
                                num_batches,
                                csv_file_path,
                                drift_alignment_with_batch=drift_within_batch,
                                scenario=scenario,
                                type_of_dataset=type_of_dataset,
                                algorithm=algorithm,
                            )

                            dataset_results[batch_size] = test_results
                else:
                    print(f"{dataset} - {batch_size}")

                    detected_drifts_dict = fetch_all_drifts(
                        batch_size,
                        dataset,
                        drift_alignment_within_batch=None,
                        plot_heatmaps=True,
                    )

                    (
                        test_results,
                        drift_results,
                        num_batches,
                    ) = run_single_experiment(
                        dataset,
                        batch_size,
                        algorithm,
                        detected_drifts_dict=detected_drifts_dict,
                    )

                    save_results_to_csv(
                        dataset,
                        batch_size,
                        drift_results,
                        test_results,
                        num_batches,
                        csv_file_path,
                        algorithm=algorithm,
                    )
                    plot_drift_points(drift_results, dataset, batch_size)

                    dataset_results[batch_size] = test_results

            # TODO: check the plots for all the algorithms separately
            results[dataset] = dataset_results
            plot_results(dataset_results, dataset, batch_sizes)
            csv_file_paths.append(csv_file_path)

            # Load and plot original dataset features
            df, _, _ = load_and_prepare_dataset(dataset)
            plot_all_features(df, dataset)

    # Consolidate all CSV files into a single CSV
    consolidate_results(list(set(csv_file_paths)))

    # Consolidate all JSON files into a single JSON
    concatenate_json_files(list(set(json_file_paths)))


if __name__ == "__main__":
    run_full_experiment()
