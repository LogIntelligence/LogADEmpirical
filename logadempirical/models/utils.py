import torch


def time_encoding(timestamps, dimension):
    """Time encoding for transformer model to encode delta time between events from timestamps.
    Args:
        timestamps: torch.FloatTensor, shape (batch_size, seq_len)
        dimension: int, dimension of the model
    Returns:
        torch.Tensor, shape (batch_size, seq_len, dimension)
    """
    seq_len = timestamps.shape[1]
    pe = torch.zeros(timestamps.shape[0], seq_len, dimension)
    div_term = torch.exp(
        torch.arange(0, dimension, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / dimension)
    )
    pe[:, :, 0::2] = torch.sin(timestamps.unsqueeze(-1) * div_term)
    pe[:, :, 1::2] = torch.cos(timestamps.unsqueeze(-1) * div_term)
    # print(pe.shape)
    return pe


def positional_encoding(seq_len, dimension):
    """Positional encoding for transformer model.
    Args:
        seq_len: int, length of the sequence
        dimension: int, dimension of the model
    Returns:
        torch.Tensor, shape (seq_len, dimension)
    """
    pe = torch.zeros(seq_len, dimension)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / dimension)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ModelOutput:
    def __init__(self, logits, probabilities, loss=None, embeddings=None):
        self.logits = logits
        self.probabilities = probabilities
        self.loss = loss
        self.embeddings = embeddings
