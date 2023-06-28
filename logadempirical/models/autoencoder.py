import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(
            self,
            hidden_size=100,
            num_layers=2,
            num_directions=2,
            embedding_dim=16
    ):
        super(AutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.embedding_dim = embedding_dim
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=(self.num_directions == 2),
        )

        self.encoder = nn.Linear(
            self.hidden_size * self.num_directions, self.hidden_size // 2
        )

        self.decoder = nn.Linear(
            self.hidden_size // 2, self.hidden_size * self.num_directions
        )
        self.criterion = nn.MSELoss(reduction="none")

        self.attention_size = self.hidden_size
        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))
        self.sequence_length = 100

    def attention_net(self, lstm_output, device):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device='cuda'):
        x = features[2].to(device)
        # print(x.shape)
        self.batch_size = x.size()[0]
        self.sequence_length = x.size(1)

        outputs, hidden = self.rnn(x.float())

        representation = self.attention_net(outputs, device)
        x_internal = self.encoder(representation)
        x_recst = self.decoder(x_internal)
        pred = self.criterion(x_recst, representation).mean(dim=-1)
        loss = pred.mean()
        return_dict = {"loss": loss, "y_pred": pred, "repr": representation}
        return return_dict, None


if __name__ == '__main__':
    model = AutoEncoder(
        hidden_size=128,
        num_directions=2, num_layers=2, embedding_dim=300).cuda()
    inp = torch.rand(64, 10, 300)
    out = model(inp)
    print(out['y_pred'].shape)
