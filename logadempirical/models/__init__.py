from logadempirical.models.lstm import DeepLog, LogRobust, LogAnomaly
from logadempirical.models.cnn import TextCNN as CNN
from logadempirical.models.transformers import NeuralLog
import torch
from typing import Optional


class ModelConfig:
    def __init__(self, num_layers: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 vocab_size: Optional[int] = None,
                 embedding_dim: Optional[int] = None,
                 criterion: Optional[torch.nn.Module] = None,
                 dropout: float = 0.5,
                 is_bilstm: Optional[bool] = None,
                 n_class: Optional[int] = None,
                 max_seq_len: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 use_semantic: Optional[bool] = False,
                 ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.criterion = criterion
        self.dropout = dropout
        self.is_bilstm = is_bilstm
        self.n_class = n_class
        self.max_seq_len = max_seq_len
        self.out_channels = out_channels
        self.use_semantic = use_semantic


def get_model(model_name, config):
    if model_name == 'DeepLog':
        model = DeepLog(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            criterion=config.criterion
        )
    elif model_name == 'LogRobust':
        model = LogRobust(
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            is_bilstm=config.is_bilstm,
            n_class=config.n_class,
            criterion=config.criterion
        )
    elif model_name == 'LogAnomaly':
        model = LogAnomaly(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            criterion=config.criterion
        )
    elif model_name == 'CNN':
        model = CNN(
            embedding_dim=config.embedding_dim,
            max_seq_len=config.max_seq_len,
            n_class=config.n_class,
            out_channels=config.out_channels,
            criterion=config.criterion
        )
    elif model_name == 'NeuralLog':
        model = NeuralLog(
            num_encoder_layers=config.num_encoder_layers,
            dim_model=config.dim_model,
            n_class=config.n_class,
            criterion=config.criterion
        )
    else:
        raise NotImplementedError
    return model
