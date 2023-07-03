import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from typing import List
from time import time
import json

print("Loading word2vec model...")
st = time()
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./crawl-300d-2M.vec', binary=False)
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
print("Loaded word2vec model in {:.2f} seconds".format(time() - st))


# remove stop word and  punctuation, split by camel case
def clean_template(template: str, remove_stop_words: bool = True):
    template = " ".join([word.lower() if word.isupper() else word for word in template.strip().split()])
    template = re.sub('[A-Z]', lambda x: " " + x.group(0), template)  # camel case
    word_tokens = tokenizer.tokenize(template)  # tokenize
    word_tokens = [w for w in word_tokens if not w.isdigit()]  # remove digital
    if remove_stop_words:  # remove stop words, we can close this function
        filtered_sentence = [w.lower() for w in word_tokens if w not in stop_words]
    else:
        filtered_sentence = [w.lower() for w in word_tokens]

    template_clean = " ".join(filtered_sentence)
    return template_clean  # return string


# get word vec of words in log key, using weight
def log_key2vec(log_template: str, weight: List[float] = None):
    """
    Get word vec of words in log key, using weight
    Parameters
    ----------
    log_template
    weight

    Returns
    -------
    log_template_vec: list of word vec
    """
    words = log_template.strip().split()
    log_template_vec = []

    if not weight:  # if not weight, uniform weight
        weight = [1] * len(words)

    for index, word in enumerate(words):
        try:  # catch the exception when word not in pre-trained word vector dictionary
            log_template_vec.append(word2vec_model[word] * weight[index])
        except Exception as _:
            pass
    if len(log_template_vec) == 0:
        log_template_vec = np.zeros(300)
    return log_template_vec


def generate_embeddings_fasttext(templates: List[str], strategy: str = 'average') -> dict:
    """
    Generate embeddings for templates using fasttext
    Parameters
    ----------
    templates: list of templates
    strategy: average or tfidf

    Returns
    -------
    embeddings: dict of embeddings
    """
    clean_templates = [clean_template(template) for template in templates]
    templates = zip(clean_templates, templates)
    embeddings = {}
    if strategy == 'average':
        for template, k in templates:
            embeddings[k] = np.mean(log_key2vec(template), axis=0).tolist()
    elif strategy == 'tfidf':
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        X = vectorizer.fit_transform(clean_templates)
        tfidf = transformer.fit_transform(X)
        tfidf = tfidf.toarray()
        words = vectorizer.get_feature_names()
        single_weights = []
        for i, (template, k) in enumerate(templates):
            for word in template.strip().split():
                if word in words:
                    single_weights.append(tfidf[i][words.index(word)])
                else:
                    single_weights.append(0)
            embeddings[k] = np.mean(log_key2vec(template, single_weights), axis=0).tolist()
    else:
        raise ValueError('Invalid strategy')

    return embeddings


def load_embeddings_fasttext(embedding_path: str) -> dict:
    """
    Load embeddings for templates using fasttext
    Parameters
    ----------
    embedding_path: path to embeddings (json)

    Returns
    -------
    embeddings: dict of embeddings
    """
    with open(embedding_path, 'r') as f:
        embeddings = json.load(f)
    return embeddings


if __name__ == '__main__':
    for dataset in ['BGL', 'HDFS', 'Spirit', 'Thunderbird']:
        print(f'Generating embeddings for {dataset}...')
        template_df = pd.read_csv(f'./{dataset}/{dataset}.log_templates.csv')
        templates = template_df['EventTemplate'].tolist()
        embeddings_tfidf = generate_embeddings_fasttext(templates, strategy='tfidf')
        # embeddings_average = generate_embeddings_fasttext(templates, strategy='average')
        with open(f'./{dataset}/{dataset}.log_embeddings_tfidf.json', 'w') as f:
            json.dump(embeddings_tfidf, f)
        # with open(f'./{dataset}/{dataset}.log_embeddings_average.json', 'w') as f:
        #     json.dump(embeddings_average, f)
