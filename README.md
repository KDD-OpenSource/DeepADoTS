# MP-2018 [![CircleCI](https://circleci.com/gh/KDD-OpenSource/MP-2018/tree/master.svg?style=svg&circle-token=2f20af2255f5f2d1ca22193c1b896d1c97b270d3)](https://circleci.com/gh/KDD-OpenSource/MP-2018/tree/master)
Repository for the summer term master project 2018 on "Unsupervised Anomaly Detection: Representation Learning for Predictive Maintenance over Time"

## Installation
```bash
git clone --recurse-submodules -j8 git://github.com/KDD-OpenSource/MP-2018.git  
virtualenv venv -p /usr/bin/python3  
source venv/bin/activate  
pip install -r requirements.txt
```

## Project Structure
```
├── data
│   ├── processed           <- The final, canonical data sets for modeling
│   └── raw                 <- The original, immutable data dump
│
├── models                  <- Trained and serialized models
│
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
│                              the creator's initials, and a short `-` delimited description, e.g.
│                              `1.0-jqp-initial-data-exploration`
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated graphics and figures to be used in reporting
│
├── requirements.txt        <- The requirements file for reproducing the analysis environment
│
├── third_party             <- Contains repositories of related work that can be used
│
├── src                     <- Source code for use in this project
│   ├── __init__.py         <- Makes src a Python module
│   ├── preprocessing.py    <- Methods to transform the raw data into usable representations
│   └── algorithms          <- Contains wrappers of the used approaches
│
└── main.py                 <- Script that orchestrates the components in the project
```
