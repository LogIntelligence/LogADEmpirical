from logadempirical.models.lstm import DeepLog, LogRobust, LogAnomaly
from logadempirical.models.cnn import TextCNN as CNN
from logadempirical.models.transformers import NeuralLog
from logadempirical.models.utils import ModelConfig


def get_model(model_name: str, config: ModelConfig):
    if model_name == 'DeepLog':
        model = DeepLog(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            criterion=config.criterion
        )
    elif model_name == 'LogRobust':
        model = LogRobust(
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            is_bilstm=config.is_bilstm,
            dropout=config.dropout,
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
            dropout=config.dropout,
            out_channels=config.out_channels,
            criterion=config.criterion
        )
    elif model_name == 'NeuralLog':
        model = NeuralLog(
            num_encoder_layers=config.num_layers,
            dim_model=config.embedding_dim,
            n_class=config.n_class,
            dropout=config.dropout,
            criterion=config.criterion
        )
    else:
        raise NotImplementedError
    return model
