#!/bin/bash

# hyperparameters

FOLDER='hdfs/'
DATASET_NAME='hdfs_2k'
LOG_FILE='HDFS_2k.log'
MODEL_NAME='logbert'

min_len=2
PARSER_TYPE='drain'
LOG_FORMAT='Date,Time,Pid,Level,Component,Content'
WINDOW_TYPE='session'
TRAIN_SIZE=0.4

TRAIN_RATIO=1
VALID_RATIO=0.1
TEST_RATIO=1

MAX_EPOCH=20
N_EPOCHS_STOP=10
N_WARM_UP_EPOCH=0
BATCH_SIZE=32
MASK_RATIO=0.65
NUM_CANDIDATES=6

EXP_NAME=${DATASET_NAME}_${MASK_RATIO}_${NUM_CANDIDATES}

python ../main_run.py \
--folder=$FOLDER \
--log_file=$LOG_FILE \
--dataset_name=$DATASET_NAME \
--model_name=$MODEL_NAME \
--parser_type=$PARSER_TYPE \
--log_format=$LOG_FORMAT \
--is_process \
--window_type=$WINDOW_TYPE \
--is_logkey \
--min_len=$min_len \
--train_size=$TRAIN_SIZE \
--train_ratio=$TRAIN_RATIO \
--valid_ratio=$VALID_RATIO \
--test_ratio=$TEST_RATIO \
--max_epoch=$MAX_EPOCH \
--n_warm_up_epoch=$N_WARM_UP_EPOCH \
--n_epochs_stop=$N_EPOCHS_STOP \
--batch_size=$BATCH_SIZE \
--mask_ratio=$MASK_RATIO \
--adaptive_window \
--deepsvdd_loss \
--num_candidates=$NUM_CANDIDATES \
2>&1 | tee -a $OUTPT_DIR$FOLDER$MODEL_NAME/$EXP_NAME.log


