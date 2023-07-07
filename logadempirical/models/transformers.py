from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import torch
from logadempirical.models.utils import positional_encoding, ModelOutput
from typing import Optional


class NeuralLog(torch.nn.Module):
    """A transformer model with n encoder layers, d_model embedding dim, and positional encoding.
    """

    def __init__(self,
                 num_encoder_layers: int = 6,
                 dim_model: int = 512,
                 n_class: int = 2,
                 dropout: float = 0.1,
                 dim_feedforward: int = 2048,
                 num_heads: int = 8,
                 criterion: Optional[torch.nn.Module] = torch.nn.CrossEntropyLoss(),
                 ):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout=dropout)
        self.encoder = TransformerEncoder(
            self.encoder_layer, num_encoder_layers
        )
        self.linear = torch.nn.Linear(dim_model, n_class)
        self.criterion = criterion

    def forward(self, batch, device="cpu"):
        x = batch['semantic']
        try:
            y = batch['label']
        except KeyError:
            y = None
        x = x + positional_encoding(x.shape[1], x.shape[2]).to(device)
        x = self.encoder(x)
        x = x.sum(dim=1)
        logits = self.linear(x)
        probabilities = torch.softmax(logits, dim=-1)
        loss = None
        if y is not None and self.criterion is not None:
            loss = self.criterion(logits, y.view(-1).to(device))

        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, src, device="cpu"):
        del src['label']
        return self.forward(src, device=device).probabilities

    def predict_class(self, src, device="cpu"):
        del src['label']
        return torch.argmax(self.forward(src, device=device).probabilities, dim=-1)


if __name__ == '__main__':
    # loss for binary classification
    model = NeuralLog(num_encoder_layers=6, dim_model=128, n_class=2, criterion=torch.nn.CrossEntropyLoss())
    # test_model
    inp = torch.rand(32, 100, 128)
    output = model.forward(inp, device='cpu')
    print(output.embeddings.shape)
