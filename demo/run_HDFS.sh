# Description: Run DeepLog on HDFS dataset
python main_run.py --data_dir ./dataset/HDFS --output_dir ./output/ --model_name DeepLog --dataset_name HDFS --grouping session --window_size 20 --step_size 1 --train_size 0.01 --is_chronological --session_level entry --log_file HDFS.log --batch_size 2048 --lr 0.001 --accumulation_step 1 --optimizer adamw --sequential --history_size 10 --embeddings embeddings_average.json --hidden_size 128 --num_layers 2 --embedding_dim 300 --topk 10 --dropout 0.1 --max_epoch 10

# Description: Run LogAnomaly on HDFS dataset
python main_run.py --data_dir ./dataset/HDFS --output_dir ./output/ --model_name LogAnomaly --dataset_name HDFS --grouping session --window_size 20 --step_size 1 --train_size 0.01 --is_chronological --session_level entry --log_file HDFS.log --batch_size 2048 --lr 0.001 --accumulation_step 1 --optimizer adamw --sequential --quantitative --history_size 10 --embeddings embeddings_average.json --hidden_size 128 --num_layers 2 --embedding_dim 300 --topk 10 --dropout 0.1 --max_epoch 10

# Description: Run LogRobust on HDFS dataset
python main_run.py --data_dir ./dataset/HDFS --output_dir ./output/ --model_name LogRobust --dataset_name HDFS --grouping session --window_size 20 --step_size 1 --train_size 0.01 --is_chronological --session_level entry --log_file HDFS.log --batch_size 2048 --lr 0.001 --accumulation_step 1 --optimizer adamw --semantic --history_size 10 --embeddings embeddings_average.json --hidden_size 128 --num_layers 2 --embedding_dim 300 --dropout 0.1 --max_epoch 10 --config_file config/logrobust.yaml

# Description: Run CNN on HDFS dataset
python main_run.py --data_dir ./dataset/HDFS --output_dir ./output/ --model_name CNN --dataset_name HDFS --grouping session --window_size 20 --step_size 1 --train_size 0.01 --is_chronological --session_level entry --log_file HDFS.log --batch_size 2048 --lr 0.001 --accumulation_step 1 --optimizer adamw --history_size 10 --embeddings embeddings_average.json --hidden_size 128 --num_layers 2 --embedding_dim 300 --topk 10 --dropout 0.1 --max_epoch 10 

# Description: Run LogBert on HDFS dataset
python main_run.py --data_dir ./dataset/HDFS --output_dir ./output/ --model_name LogBert --dataset_name HDFS --grouping session --window_size 20 --step_size 1 --train_size 0.01 --is_chronological --session_level entry --log_file HDFS.log --batch_size 2048 --lr 0.001 --accumulation_step 1 --optimizer adamw --sequential --quantitative --history_size 30 --embeddings embeddings_average.json --hidden_size 128 --num_layers 2 --embedding_dim 300 --topk 10 --dropout 0.1 --max_epoch 10 --config_file config/LogBert

#run DeepLog on BGL dataset
python main_run.py --data_dir ./dataset/BGL --output_dir ./output/ --model_name DeepLog --dataset_name BGL --grouping session --window_size 20 --step_size 1 --train_size 0.01 --is_chronological --session_level entry --log_file BGL.log --batch_size 2048 --lr 0.001 --accumulation_step 1 --optimizer adamw --sequential --history_size 10 --embeddings embeddings_average.json --hidden_size 128 --num_layers 2 --embedding_dim 300 --topk 10 --dropout 0.1 --max_epoch 10

#run LogBert on  BGL dataset
python main_run.py --data_dir ./dataset/BGL --output_dir ./output/ --model_name LogBert --dataset_name BGL --grouping session --window_size 20 --step_size 1 --train_size 0.01 --is_chronological --session_level entry --log_file BGL.log --batch_size 2048 --lr 0.001 --accumulation_step 1 --optimizer adamw --sequential --quantitative --history_size 30 --embeddings embeddings_average.json --hidden_size 128 --num_layers 2 --embedding_dim 300 --topk 10 --dropout 0.1 --max_epoch 10 --config_file config/LogBert


