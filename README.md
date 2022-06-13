# Data Mining for Early Detection of Cardiac Amyloidosis using Routinely Collected Laboratory parameters

This repository contains the code and notebooks for reproducing the results in the thesis. The data sets are not included in the repository, therefore you will not be able to run the analysis.

The repository consists of:
- Analysis - Data understanding.ipynb includes the general data summarization and initial deep dive into the data. Running all the cells in this notebook will generate the figures in the folder "output_data_understanding/".
- Analysis - Preparation and Modeling.ipynb includes all the processes for preparing the data and building the models. Running all the cells in this notebook will generate the figures and plots used in the thesis.
- "src" folder includes all (or most) of the pipelines and modified algorithms used in the analyses notebooks.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```
pip install -r requirements.txt
```

## Usage
To run the analysis with jupyter notebook:
```
jupyter notebook
```

