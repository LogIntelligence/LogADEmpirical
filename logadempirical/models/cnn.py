import torch
import torch.nn as nn
from logadempirical.models.utils import ModelOutput
from typing import Optional

import warnings

warnings.filterwarnings("ignore")


class TextCNN(nn.Module):
    def __init__(self,
                 embedding_dim: int = 300,
                 max_seq_len: int = 100,
                 out_channels: int = 100,
                 n_class: int = 2,
                 criterion: Optional[nn.Module] = None):
        super(TextCNN, self).__init__()

        in_channels = 1
        self.kernel_size_list = [3, 4, 5]

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d((max_seq_len - kernel_size + 1, 1))
        ) for kernel_size in self.kernel_size_list])

        self.fc = nn.Linear(len(self.kernel_size_list) * out_channels, n_class)
        self.criteria = criterion

    def forward(self, batch, device='cpu'):
        x = batch['semantic']
        y = batch['label']
        x = torch.unsqueeze(x.to(device), 1)
        batch_size = x.size(0)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(batch_size, -1)
        logits = self.fc(x)
        probabilities = torch.softmax(logits, dim=-1)
        loss = None
        if y is not None and self.criteria is not None:
            loss = self.criteria(logits, y.view(-1).to(device))

        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, src, device="cpu"):
        return self.forward(src, device=device).probabilities

    def predict_class(self, src, device="cpu"):
        return torch.argmax(self.forward(src, device=device).probabilities, dim=-1)


if __name__ == '__main__':
    model = TextCNN(300, 100, 8)
    inp = torch.rand(64, 100, 300)
    out = model(inp)
    print(out.logits.shape)
