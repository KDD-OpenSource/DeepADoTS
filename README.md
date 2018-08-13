
# Unsupervised Anomaly Detection: Representation Learning for Predictive Maintenance over Time [![CircleCI](https://circleci.com/gh/KDD-OpenSource/MP-2018/tree/master.svg?style=svg&circle-token=2f20af2255f5f2d1ca22193c1b896d1c97b270d3)](https://circleci.com/gh/KDD-OpenSource/MP-2018/tree/master)

* Repository for the summer term master project 2018 
* The goal of this repository is to provide a benchmarking pipeline for anomaly detection on time series data for multiple state-of-the-art deep learning methods


## Installation

```bash
git clone --recurse-submodules -j8 git://github.com/KDD-OpenSource/MP-2018.git  
virtualenv venv -p /usr/bin/python3  
source venv/bin/activate  
pip install -r requirements.txt
```

## Deployment

- Install nvidia-docker
- `docker build -t mp2018 .`
- `nvidia-docker run -ti mp2018 /bin/bash -c "python3.6 /repo/main.py"`

## Usage
In local repository folder, activate virtual environment first

```
source venv/bin/activate
python3 main.py
``` 

## Authors/Contributors
Team:
* Maxi Fischer
* Willi Gierke
* Thomas Kellermeier
* Ajay Kesar
* Axel Stebner
* Daniel Thevessen

Supervisors:
* Lukas Ruff
* Fabian Geier
* Emmanuel Müller

## Implemented Algorithms

### LSTM-AD
Malhotra, Pankaj, et al. "Long short term memory networks for anomaly detection in time series." Proceedings. Presses universitaires de Louvain, 2015.

### LSTM-ED
Malhotra, Pankaj, et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection." arXiv preprint arXiv:1607.00148 (2016).

### Simple Autoencoder baseline
Hawkins, Simon, et al. "Outlier detection using replicator neural networks." International Conference on Data Warehousing and Knowledge Discovery. Springer, Berlin, Heidelberg, 2002.

### Donut
Xu, Haowen, et al. "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

### REBM using a restricted Boltzmann Machine as energy-based model
Zhai, Shuangfei, et al. "Deep structured energy based models for anomaly detection." arXiv preprint arXiv:1605.07717 (2016).

### Dagmm
Zong, Bo, et al. "Deep autoencoding gaussian mixture model for unsupervised anomaly detection." (2018).

### LSTM-DAGMM
Extension of Dagmm using the LSTM-Autoencoder from LSTM-ED instead of a simple Neural Network Autoencoder

## Credits

[Inspiration for Dagmm Implementation](https://github.com/danieltan07/dagmm)