## Empirical Study on Log-based Anomaly Detection

This repository provides the implementation of recent log-based anomaly detection methods. 
The process includes downloading raw data online, parsing logs into structured data, 
creating log sequences and finally modeling.

<!-- **Note: This repo is built based on [LogBERT](https://github.com/HelenGuohx/logbert) and [logdeep](https://github.com/donglee-afar/logdeep)** -->

### Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.6
  

### Installation
This code is written in Python 3.8 and requires the packages listed in requirements.txt.
An virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
