# Empirical Analysis of Data Drift

This repository contains the code for experiments from the paper *An Empirical Analysis of Data Drift Detection Techniques in Machine Learning Systems* published at SBBD 2024, as well as the extended code and results that were submitted to the Journal of Information and Data Management 2025 (JIDM 2025).

## Publication

**SBBD 2024**: The original research was published in the proceedings of the 39th Brazilian Symposium on Databases (SBBD 2024). The full paper is available at: https://sol.sbc.org.br/index.php/sbbd/article/view/30681

## Overview

This project implements and evaluates several data drift detection techniques:

- HDDDM (Hellinger Distance Drift Detection Method)
- KSDDM (Kolmogorov-Smirnov Drift Detection Method)
- JSDDM (Jensen-Shannon Drift Detection Method)

The framework includes tools for generating synthetic datasets with controlled drifts, applying different drift detection methods, and evaluating their performance.

## Installation

On an environment with Python 3.12.0, install the Poetry package manager. You can find instructions [here](https://python-poetry.org/docs/).

After that, run the following command to install the dependencies:

```bash
poetry install
```

## Files for the experiments

Download the file from USP Data repo onto the data/ folder.
On that folder, there is a specific README file with instructions on how to download the data and structure it.


## Running the experiments

To run the full experiment, you can use the following commands:

```bash
poetry shell
PYTHONPATH=. python codes/comparisor.py
```

## Results

After running the experiment, the results can be found on the folder _comparison_results_. 
Each subfolder contains the results for a specific dataset, along with the plots and the metrics results.
Also on the folder _comparison_results_, there is a CSV file named _consolidated_results.csv_ with all the metrics obtained in the experiments for all batch sizes, datasets and techniques.

## Documentation

This repository includes comprehensive documentation for different aspects of the framework:

### [Comparison Results Documentation](comparison_results/README.md)
Detailed documentation of the experimental results structure, including:
- Directory organization and file formats
- Dataset types (real vs synthetic)
- Performance metrics and visualizations
- Drift detection method comparisons

### [Synthetic Datasets Documentation](synthetic_datasets_documentation.md)
Complete specification of the synthetic datasets used in the empirical analysis:
- Theoretical foundation and drift scenario design
- Base dataset generation methodology
- Drift configuration parameters
- Implementation details and reproducibility guidelines
- Visual examples and plot interpretations

### [Experimental Results Tables](experimental_results_tables.md)
Comprehensive tables presenting F1 and AUC metrics across different batch sizes:
- Performance comparison of all drift detection methods (Base, KS95, KS90, HD, JS)
- Results for batch sizes 1,000, 1,500, 2,000, and 2,500
- Detailed metrics for both real and synthetic datasets
- Clear formatting with highlighted best-performing methods
- Dataset abbreviations and method explanations

These documentation files provide essential information for understanding the experimental setup, reproducing results, and extending the framework for future research.
