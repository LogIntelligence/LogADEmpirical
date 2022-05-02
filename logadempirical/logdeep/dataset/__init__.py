from .vocab import Vocab
import torch
from transformers import BertTokenizer, BertModel
import re
import string

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def clean(s):
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s


def bert_encoder(s, E):
    s = clean(s)
    if s in E.keys():
        return E[s]
    inputs = bert_tokenizer(s, return_tensors='pt', max_length=512, truncation=True)
    outputs = bert_model(**inputs)
    v = torch.mean(outputs.last_hidden_state, dim=1)
    E[s] = v[0].detach().numpy()
    return E[s]
