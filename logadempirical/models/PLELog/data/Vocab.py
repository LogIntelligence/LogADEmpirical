import numpy as np




class Vocab(object):
    ##please always set PAD to zero, otherwise will cause a bug in pad filling (Tensor)
    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self, tag_counter):
        self._id2tag = []
        self._id2tag.append('yes')
        self._id2tag.append('no')
        # for tag, count in tag_counter.most_common():
        #     self._id2tag.append(tag)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: output tags dumplicated, please check!")

        print("Vocab info: #output tags %d" % (self.tag_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        self._id2word = []
        allwords = set()
        for special_word in ['<pad>', '<bos>', '<eos>', '<oov>']:
            if special_word not in allwords:
                allwords.add(special_word)
                self._id2word.append(special_word)

        with open(embfile, encoding='utf-8') as f:
            line = f.readline()
            vocabSize, embedding_dim = line.strip().split()
            embedding_dim = int(embedding_dim)
            for line in f.readlines():
                values = line.strip().split()
                if len(values) == embedding_dim + 1:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        self._id2word.append(curword)
        word_num = len(self._id2word)
        print('Total words: ' + str(word_num) + '\n')
        print('The dim of pretrained embeddings: %d \n' % (embedding_dim))

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)

        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        oov_id = self._word2id.get('<oov>')
        if self.UNK != oov_id:
            print("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            # line = f.readline()
            tem_count = 0
            for line in f.readlines():
                values = line.split()
                if len(values) == embedding_dim + 1:
                    index = self._word2id.get(values[0])
                    vector = np.array(values[1:], dtype='float64')
                    embeddings[index] = vector
                    embeddings[self.UNK] += vector
                    tem_count += 1
        if tem_count != word_num-4:
            print("Goes wrong when calculating UNK emb!")
        embeddings[self.UNK] = embeddings[self.UNK] / word_num

        return embeddings

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x) for x in xs]
        return self._tag2id.get(xs)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)
