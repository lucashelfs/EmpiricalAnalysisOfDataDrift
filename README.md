# Empirical Analysis of Data Drift

This repository contains the code for experiments from the paper *An Empirical Analysis of Data Drift Detection Techniques in Machine Learning Systems* for SBBD 2024.

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

## Running Tests

The repository includes a comprehensive test suite that covers the key components of the framework. To run the tests:

### Using the test runner script

The easiest way to run tests is using the provided script:

```bash
poetry shell
./run_tests.py  # Run tests with pytest (default)
./run_tests.py --framework unittest  # Run tests with unittest
./run_tests.py --framework both  # Run tests with both frameworks
./run_tests.py -v  # Run tests with verbose output
```

### Manually running tests

You can also run tests manually:

#### Using unittest

```bash
poetry shell
python -m unittest discover tests
```

#### Using pytest

```bash
poetry shell
pytest tests/
```

For more details about the tests and how to run specific test cases, see the [tests/README.md](tests/README.md) file.
