## Log-based Anomaly Detection with Deep Learning: How Far Are We?
**Abstract**: Software-intensive systems produce logs for troubleshooting purposes. Recently, many deep learning models
 have been proposed to automatically detect system anomalies based on log data. These models typically claim very high detection accuracy. For example, most models report an F-measure greater than 0.9 on the commonly-used HDFS dataset. To achieve a profound understanding of how far we are from solving the problem of log-based anomaly detection, in this paper, we conduct an in-depth analysis of five state-of-the-art deep learning-based models for detecting system anomalies on four public log datasets. Our experiments focus on several aspects of model evaluation, including training data selection, data grouping, class distribution, data noise, and early detection ability. Our results point out that all these aspects have significant impact on the evaluation, and that all the studied models do not always work well. The problem of log-based anomaly detection has not been solved yet. Based on our findings, we also suggest possible future work.
This repository provides the implementation of recent log-based anomaly detection methods.

### Studied Models
| Model | Paper |
| :--- | :--- |
| DeepLog | [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://dl.acm.org/doi/abs/10.1145/3133956.3134015) |
| LogAnomaly | [LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs](https://www.ijcai.org/proceedings/2019/658) |
| PLELog | [Semi-Supervised Log-Based Anomaly Detection via Probabilistic Label Estimation](https://ieeexplore.ieee.org/document/9401970/) |
| LogRobust | [Robust log-based anomaly detection on unstable log data](https://dl.acm.org/doi/10.1145/3338906.3338931) |
| CNN | [Detecting Anomaly in Big Data System Logs Using Convolutional Neural Network](https://ieeexplore.ieee.org/document/8511880) |

### Requirements
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.7.0
  
The required packages are listed in requirements.txt. Install:

```
pip install -r requirements.txt
```

### Demo
- Example of DeepLog on BGL with fixed window size of 1 hour:
```shell script
python main_run.py --folder=bgl/ --log_file=BGL.log --dataset_name=bgl --model_name=deeplog --window_type=sliding
 --sample=sliding_window --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100
 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=1024 --num_candidates=150 --history_size=10 --lr=0.001
 --accumulation_step=5 --session_level=hour --window_size=60 --step_size=60 --output_dir=experimental_results/demo
/random/ --is_process
```
- For more explanation of parameters:
```shell script
python main_run.py --help
```

## Citation
If you find the code and models useful for your research, please cite the following paper:
```
@inproceedings{le2022log,
  title={Log-based Anomaly Detection with Deep Learning: How Far Are We?},
  author={Le, Van-Hoang and Zhang, Hongyu},
  booktitle={2022 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)},
  year={2022}
}
```
