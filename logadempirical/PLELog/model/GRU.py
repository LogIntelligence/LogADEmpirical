from logadempirical.PLELog.module.Attention import *
from logadempirical.PLELog.module.CPUEmbedding import *
from logadempirical.PLELog.module.Common import *
import torch
import random





class AttGRUModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(AttGRUModel, self).__init__()
        self.config = config
        # self.use_cosine = config.use_cosine
        if pretrained_embedding is not None:
            vocab_size, word_dims = pretrained_embedding.shape
        else:
            vocab_size, word_dims = vocab.vocab_size, config.word_dims
        self.vocab = vocab
        if vocab.vocab_size != vocab_size:
            print("word vocab size does not match, check word embedding file")
        self.word_embed = CPUEmbedding(vocab.vocab_size, word_dims, padding_idx=vocab.PAD)
        if pretrained_embedding is not None:
            self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

        self.rnn = nn.GRU(input_size=word_dims, hidden_size=config.lstm_hiddens, num_layers=config.lstm_layers,
                          batch_first=True, bidirectional=True, dropout=config.dropout_lstm_hidden)

        self.sent_dim = 2 * config.lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        self.proj = NonLinear(self.sent_dim, vocab.tag_size)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(vocab.vocab_size, word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def word_embed_onehot(self, words):
        reps = []
        base = []
        for i in range(self.vocab.vocab_size):
            base.append(0)
        batch_size, src_len = words.shape
        for i in range(batch_size):
            rep = base
            for j in range(src_len):
                rep[words[i][j].data.item()] += 1
            reps.append(rep)
        res = torch.from_numpy(np.asarray(reps, dtype=np.float32)).unsqueeze(dim=2)
        masks = torch.ones_like(torch.from_numpy(np.asarray(reps, dtype=np.float32)))
        return res, masks
        pass

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.config.dropout_emb)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs, represents




