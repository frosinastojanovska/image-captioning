from nltk.tokenize import word_tokenize
import numpy as np
import os
from gensim.models import KeyedVectors
import json
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
import nltk


def append_unknown_token(from_file_name, to_file_name):
    with open(from_file_name, 'r', encoding='utf-8') as f, open(to_file_name, 'w', encoding='utf-8') as t:
        t.write('<UNK> ')
        for i in range(100):
            if i < 99:
                t.write('0 ')
            else:
                t.write('0\n')
        line = f.readline()
        while line != '':
            t.write(line)
            line = f.readline()


def load_embeddings(file_name):
    embeddings = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            vals = np.array(parts[1:], dtype=np.float)
            embeddings[parts[0]] = vals
            line = doc.readline()
    return embeddings


def load_embeddings_model(glove_file, word2vec_file):
    if not os.path.exists(word2vec_file):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_file, word2vec_file)
    word_embeddings = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    return word_embeddings


def reduce_embeddings(vocabulary, original_glove_file, glove_file):
    with open(original_glove_file, 'r', encoding='utf-8') as f, open(glove_file, 'w', encoding='utf-8') as t:
        line = f.readline()
        while line != '':
            parts = line.split(' ')
            if parts[0] in vocabulary or parts[0] == '<UNK>':
                t.write(line)
            line = f.readline()


def tokenize_corpus(data_file, train, validation):
    with open(data_file, 'r', encoding='utf-8') as doc:
        data = json.loads(doc.read())
    corpus = list()
    for x in data:
        if x['id'] in train or x['id'] in validation:
            regs = x['regions']
            for reg in regs:
                caption = reg['phrase']
                tokens = word_tokenize(caption.lower())
                tags = nltk.pos_tag(tokens)
                for tag in tags:
                    if tag[1].startswith('NN'):
                        corpus.append(tag[0])
    frequencies = sorted(Counter(corpus).items(), key=lambda x: x[1], reverse=True)
    return set([x[0] for x in frequencies if x[1] >= 15 and
                x[0] not in punctuation and
                x[0] not in set(stopwords.words('english'))])


def encode_caption(caption, model, vocabulary):
    """ convert caption from string to word embedding """
    tokens = word_tokenize(caption.lower())
    vector = list()
    for token in tokens:
        encoded_token = encode_word(token, model, vocabulary)
        if len(encoded_token) != 0:
            vector.append(encoded_token)
    return np.array(vector)


def encode_word(word, model, vocabulary):
    """ convert word to its word embedding vector """
    if word in model.wv.vocab and word in vocabulary:
        vec = model[word]
    else:
        # vec = model['<UNK>']
        vec = []
    '''
    if word in embeddings.keys():
        vec = embeddings[word]
    else:
        # w = Word(word)
        # w = w.spellcheck()[0][0]
        # if w in embeddings.keys():
            # vec = embeddings[w]
        # else:
            # vec = np.zeros(100)
        vec = np.zeros(100)
    '''
    return vec


def decode_caption(vector, model):
    """ convert word embedding array to caption """
    caption = ''
    for v in vector:
        caption += decode_word(v, model) + ' '
    print(caption)
    return caption


def decode_word(vec, model):
    """ convert word embedding vector to the corresponding word """
    if np.all(vec == 0):
        word = '<UNK>'
    else:
        word = model.most_similar(positive=[vec], negative=[], topn=1)
        if word[0][1] < 0.5:
            word = '<UNK>'
        else:
            word = word[0][0]
    return word


if __name__ == '__main__':
    original_glove_file = '../dataset/original_glove.6B.100d.txt'
    glove_file = '../dataset/glove.6B.100d.txt'
    word2vec_file = '../dataset/glove.6B.100d.txt.word2vec'
    if not os.path.exists(glove_file):
        append_unknown_token(original_glove_file, glove_file)
    if not os.path.exists(word2vec_file):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_file, word2vec_file)
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    # word_embeddings = load_embeddings(glove_file)

    print('===Encode===')
    encoded_caption = encode_caption('Helloo world xxx', model)
    print(encoded_caption)
    print('===Decode===')
    decoded_caption = decode_caption(encoded_caption, model)
    print(decoded_caption)
