## LogBERT: Log Anomaly Detection via BERT
This repository is under construction: 80% done_

This repository provides the implementation of Logbert method for log anomaly detection. 
The process includes downloading raw data online, parsing logs into structured data, 
creating log sequences and finally modeling. 

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

### Experiment
We currently have implemented our model Logbert and other baseline models on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [BGL](https://github.com/logpai/loghub/tree/master/BGL), and [thunderbird]() datasets

 ```main_run.py``` contains all parameters and options.
 
 #### BGL example
 ```shell script
cd scripts

# running on 2000 bgl samples for testing and debugging
sh download_bgl_2k.sh
sh run_logbert_bgl_2k.sh

# runnning on bgl dataset
sh download_bgl.sh
sh run_logbert_bgl.sh

```

 #### HDFS example
 ```shell script
cd scripts

# running on 2000 bgl samples for testing and debugging
sh download_hdfs_2k.sh
sh run_logbert_hdfs_2k.sh

# runnning on bgl dataset
sh download_hdfs.sh
sh run_logbert_hdfs.sh

```




