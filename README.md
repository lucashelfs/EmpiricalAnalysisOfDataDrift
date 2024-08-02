# Empirical analysis of data drift

This repository contains the code for experiments from the paper *An Empirical Analysis of Data Drift Detection Techniques in Machine Learning Systems* for SBBD 2024.

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
python codes/comparisor.py
```

## Results

After running the experiment, the results can be found on the folder _comparison_results_. 
Each subfolder contains the results for a specific dataset, along with the plots and the metrics results.
Also on the folder _comparison_results_, there is a CSV file named _consolidated_results.csv_ with all the metrics obtained in the experiments for all batch sizes, datasets and techniques.
