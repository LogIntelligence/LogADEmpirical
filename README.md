## Log-based Anomaly Detection with Deep Learning: How Far Are We?

**Abstract**: Software-intensive systems produce logs for troubleshooting purposes. Recently, many deep learning models
have been proposed to automatically detect system anomalies based on log data. These models typically claim very high
detection accuracy. For example, most models report an F-measure greater than 0.9 on the commonly-used HDFS dataset. To
achieve a profound understanding of how far we are from solving the problem of log-based anomaly detection, in this
paper, we conduct an in-depth analysis of five state-of-the-art deep learning-based models for detecting system
anomalies on four public log datasets. Our experiments focus on several aspects of model evaluation, including training
data selection, data grouping, class distribution, data noise, and early detection ability. Our results point out that
all these aspects have significant impact on the evaluation, and that all the studied models do not always work well.
The problem of log-based anomaly detection has not been solved yet. Based on our findings, we also suggest possible
future work.
This repository provides the implementation of recent log-based anomaly detection methods.

### I. Studied Models

| Model                        | Paper                                                                                                                                          |
|:-----------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Unsupervised**             |                                                                                                                                                |
| DeepLog (CCS '17)            | [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://dl.acm.org/doi/abs/10.1145/3133956.3134015)          |
| LogAnomaly (IJCAI '19)       | [LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs](https://www.ijcai.org/proceedings/2019/658) |
| LogBERT (IJCNN '21) (coming) | [LogBERT: Log Anomaly Detection via BERT](https://ieeexplore.ieee.org/abstract/document/9534113)                                               |
| **Semi-supervised**          |                                                                                                                                                |
| PLELog (ICSE '21)            | [Semi-Supervised Log-Based Anomaly Detection via Probabilistic Label Estimation](https://ieeexplore.ieee.org/document/9401970/)                |
| **Supervised**               |                                                                                                                                                |
| CNN (DSAC '18)               | [Detecting Anomaly in Big Data System Logs Using Convolutional Neural Network](https://ieeexplore.ieee.org/document/8511880)                   |
| LogRobust (ESEC/FSE '19)     | [Robust log-based anomaly detection on unstable log data](https://dl.acm.org/doi/10.1145/3338906.3338931)                                      |
| NeuralLog (ASE '21) (coming) | [Log-based Anomaly Detection Without Log Parsing](https://ieeexplore.ieee.org/document/9678773)                                                |

### II. Requirements

- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch

The required packages are listed in requirements.txt. Install:

```
pip install -r requirements.txt
```

### III. Usage

#### 1. Data Preparation
##### 1.1. Datasets

We use datasets collected by LogPAI for evaluation. The datasets are available at [loghub](https://github.com/logpai/loghub).
The details of datasets is shown as belows:

| **Dataset**  | **Size** | **# Logs**  | **# Anomalies** | **Anomaly Ratio** |
|:-------------|:---------|:------------|:----------------|:------------------|
| HDFS         | 1.5  GB  | 11,1 75,629 | 16,8 38         | 2.93%             |
| BGL          | 743 MB   | 4,747,963   | 348,460         | 7.34 %            |
| Thunderbird  | 1.4 GB   | 10,000,000  | 4,934           | 0.49%             |
| Spirit       | 1.4 GB   | 5,000,000   | 764,500         | 15.29%            |

##### 1.2. Parsing



#### 2. Training

```
```

#### 3. Testing

```
```

### Citation

If you find the code and models useful for your research, please cite the following paper:

```
@inproceedings{le2022log,
  title={Log-based Anomaly Detection with Deep Learning: How Far Are We?},
  author={Le, Van-Hoang and Zhang, Hongyu},
  booktitle={2022 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)},
  year={2022}
}
```
