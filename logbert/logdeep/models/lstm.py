import torch
import torch.nn as nn
from torch.autograd import Variable

class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(self.embedding_dim,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc0 = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, device):
        input0 = features[0]
        embed0 = self.embedding(input0)
        h0 = torch.zeros(self.num_layers, embed0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, embed0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(embed0, (h0, c0))
        out0 = self.fc0(out[:, -1, :])
        return out0, out0


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device):
        input0 = features[2]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.attention_net(out)
        out = self.fc(out[:, -1, :])
        return out


#log key add embedding
class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.embedding_size = vocab_size
        self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm0 = nn.LSTM(self.embedding_dim,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, features, device):
        input0, input1 = features[0], features[1]
        embed0 = self.embedding(input0)

        h0_0 = torch.zeros(self.num_layers, embed0.size(0),
                           self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, embed0.size(0),
                           self.hidden_size).to(device)

        out0, _ = self.lstm0(embed0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))

        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out, out

