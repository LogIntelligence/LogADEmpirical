import torch
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore")


class TextCNN(nn.Module):
    def __init__(self, vector_size, max_seq_len, out_channels):
        super(TextCNN, self).__init__()

        in_channels = 1
        self.kernel_size_list = [3, 4, 5]

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, vector_size)),
            nn.ReLU(),
            nn.MaxPool2d((max_seq_len - kernel_size + 1, 1))
        ) for kernel_size in self.kernel_size_list])

        self.fc = nn.Linear(len(self.kernel_size_list) * out_channels, 2)

    def forward(self, features, device='cuda'):
        input = features[2]
        # print(input.shape)
        input = torch.unsqueeze(input, 1)
        batch_size = input.size(0)
        input = [conv(input) for conv in self.convs]
        input = torch.cat(input, dim=1)
        input = input.view(batch_size, -1)
        output = self.fc(input)

        return output, input

if __name__ == '__main__':
    model = TextCNN(300, 100, 8)
    inp = torch.rand(64, 100, 300)
    out, _ = model(inp)
    print(out.shape)