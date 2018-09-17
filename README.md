
# Anomaly Detection on Time Series: An Evaluation of Deep Learning Methods. [![CircleCI](https://circleci.com/gh/KDD-OpenSource/DeepADoTS/tree/master.svg?style=svg&circle-token=2f20af2255f5f2d1ca22193c1b896d1c97b270d3)](https://circleci.com/gh/KDD-OpenSource/DeepADoTS/tree/master)

The goal of this repository is to provide a benchmarking pipeline for anomaly detection on time series data for multiple state-of-the-art deep learning methods.


## Authors/Contributors
Team:
* [Maxi Fischer](https://github.com/maxifischer)
* [Willi Gierke](https://github.com/WGierke)
* [Thomas Kellermeier](https://github.com/Chaoste)
* [Ajay Kesar](https://github.com/weaslbe)
* [Axel Stebner](https://github.com/xasetl)
* [Daniel Thevessen](https://github.com/danthe96)

Supervisors:
* [Lukas Ruff](https://github.com/lukasruff)
* [Fabian Geier](https://github.com/fabiangei)
* [Emmanuel Müller](https://github.com/emmanuel-mueller)


## Implemented Algorithms

### LSTM-AD
Malhotra, Pankaj, et al. "Long short term memory networks for anomaly detection in time series." Proceedings. Presses universitaires de Louvain, 2015.

### LSTM-ED
Malhotra, Pankaj, et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection." ICML, 2016.

### Autoencoder
Hawkins, Simon, et al. "Outlier detection using replicator neural networks." DaWaK, 2002.

### Donut
Xu, Haowen, et al. "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications." WWW, 2018.

### REBM using a restricted Boltzmann Machine as energy-based model
Zhai, Shuangfei, et al. "Deep structured energy based models for anomaly detection." ICML, 2016.

### DAGMM
Zong, Bo, et al. "Deep autoencoding gaussian mixture model for unsupervised anomaly detection." ICLR, 2018.

### LSTM-DAGMM
Extension of Dagmm using an LSTM-Autoencoder instead of a Neural Network Autoencoder


## Installation

```bash
git clone --recurse-submodules -j8 git://github.com/KDD-OpenSource/DeepADoTS.git  
virtualenv venv -p /usr/bin/python3  
source venv/bin/activate  
pip install -r requirements.txt
```

## Usage
In the local repository folder, activate virtual environment first

```
source venv/bin/activate
python3 main.py
``` 

## Deployment

- You can use nvidia-docker
- `docker build -t mp2018 .`
- `nvidia-docker run -ti mp2018 /bin/bash -c "python3.6 /repo/main.py"`


## Credits
[Base implementation for DAGMM](https://github.com/danieltan07/dagmm)  
[Base implementation for Donut](https://github.com/haowen-xu/donut)  
[Base implementation for Recurrent EBM](https://github.com/dshieble/Music_RNN_RBM)  
