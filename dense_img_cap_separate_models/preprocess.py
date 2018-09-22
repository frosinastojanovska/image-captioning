import json
import numpy as np
from string import punctuation
from collections import Counter
from nltk.tokenize import word_tokenize


def load_corpus(tokens, embeddings, embeddings_dim):
    id_to_word = dict()
    word_to_id = dict()
    embedding_matrix = np.zeros((len(tokens) + 3, embeddings_dim))
    id_to_word[0] = '<unk>'
    word_to_id['<unk>'] = 0

    # Add <start> and <end> tokens and initialize them randomly in range [-0.5, 0.5]
    embedding_matrix[1, :] = np.random.mtrand._rand.rand(embeddings_dim) - 0.5
    id_to_word[1] = '<start>'
    word_to_id['<start>'] = 1
    embedding_matrix[2, :] = np.random.mtrand._rand.rand(embeddings_dim) - 0.5
    id_to_word[2] = '<end>'
    word_to_id['<end>'] = 2

    for i in range(len(tokens)):
        id_to_word[i + 3] = tokens[i]
        word_to_id[tokens[i]] = i + 3
        embedding_matrix[i + 3, :] = embeddings[tokens[i]]
    return word_to_id, id_to_word, embedding_matrix


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


def tokenize_corpus(data_file, train, embeddings):
    with open(data_file, 'r', encoding='utf-8') as doc:
        data = json.loads(doc.read())
    corpus = list()
    for data_point in data:
        if data_point['id'] in train:
            regs = data_point['regions']
            for reg in regs:
                caption = reg['phrase']
                tokens = word_tokenize(caption.lower())
                for tag in tokens:
                    corpus.append(tag)
    frequencies = sorted(Counter(corpus).items(), key=lambda x: x[1], reverse=True)
    return set([x[0] for x in frequencies if x[1] >= 15 and x[0] in embeddings and x[0] not in punctuation])


def encode_caption(caption, word_to_id):
    """ convert caption from string to word index array """
    tokens = word_tokenize(caption.lower())
    vector = list()
    for token in tokens:
        encoded_token = encode_word(token, word_to_id)
        if encoded_token != 0:
            vector.append(encoded_token)
    return np.array(vector)


def encode_caption_v2(caption, word_to_id):
    """ convert caption from string to one-hot array """
    tokens = word_tokenize(caption.lower())
    vector = list()
    for token in tokens:
        encoded_token = encode_word_v2(token, word_to_id)
        if len(encoded_token) != 0 and encoded_token[0] != 1:
            vector.append(encoded_token)
    return np.array(vector)


def encode_word(word, word_to_id):
    """ convert word to its index"""
    if word in word_to_id.keys():
        pos = word_to_id[word]
    else:
        pos = 0
    return pos


def encode_word_v2(word, word_to_id):
    """ convert word to its one-hot vector"""
    if word in word_to_id.keys():
        pos = word_to_id[word]
        vec = np.zeros(len(word_to_id))
        vec[pos] = 1
    else:
        pos = word_to_id['<UNK>']
        vec = np.zeros(len(word_to_id))
        vec[pos] = 1
    return vec


def decode_caption(vector, id_to_word):
    """ convert one-hot encoding array to caption """
    caption = ''
    for v in vector:
        caption += decode_word(v, id_to_word) + ' '
    return caption


def decode_word(vec, id_to_word):
    """ convert one-hot encoding vector to the corresponding word """
    inverted = np.argmax(vec)
    return id_to_word[inverted]
