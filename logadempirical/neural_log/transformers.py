from torch import Tensor
import torch.nn.functional as F
import torch
from torch import nn

from logadempirical.neural_log.encoding import position_encoding, PositionalEmbedding, PositionalEncoding
from logadempirical.neural_log.attention import MultiHeadAttention
from logadempirical.neural_log.bottleneck_transformer_helpers import RelPosEmb, AbsPosEmb


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 512,
            num_heads: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.pos_encoding = PositionalEncoding(dim_model)

    def forward(self, src: Tensor, device: str = "cuda:0") -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        # src += position_encoding(seq_len, dimension)
        # src = self.pos_encoding(src)
        for layer in self.layers:
            src = layer(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        return self.feed_forward(tgt)


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 512,
            num_heads: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)


class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # self.decoder = TransformerDecoder(
        #     num_layers=num_decoder_layers,
        #     dim_model=dim_model,
        #     num_heads=num_heads,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        # )

    def forward(self, src: Tensor, device: str = "cuda:0") -> Tensor:
        return self.encoder(src, device)


class NeuralLog(nn.Module):
    def __init__(self,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_model: int = 512,
                 num_heads: int = 6,
                 dim_feedforward: int = 2048,
                 droput: float = 0.1,
                 activation: nn.Module = nn.ReLU()):
        super(NeuralLog, self).__init__()
        self.num_heads = num_heads
        self.transformers = Transformer(num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_model=dim_model,
                                        num_heads=num_heads,
                                        dim_feedforward=dim_feedforward,
                                        dropout=droput,
                                        activation=activation)
        self.dropout1 = nn.Dropout(droput)
        self.fc1 = nn.Linear(dim_model, 32)
        self.dropout2 = nn.Dropout(droput)
        self.fc2 = nn.Linear(32, 2)
        self.pos_emb1 = RelPosEmb(1024, dim_model)
        self.pos_emb2 = AbsPosEmb(1024, dim_model)


    def forward(self, features: list, device: str ='cuda:0'):
        inp = features[2]
        # print(inp.shape)
        x = self.transformers(inp, device)
        x = torch.sum(x, dim = 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x, x


if __name__ == '__main__':
    src = torch.rand(64, 100, 768)
    model = NeuralLog(dim_model=768)
    out, _ = model([src])
    print(out.shape)
