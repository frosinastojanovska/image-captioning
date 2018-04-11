from nltk.tokenize import word_tokenize
from textblob import Word
import numpy as np
from scipy.spatial import distance


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


def encode_caption(caption, embeddings):
    """ convert caption from string to word embedding array """
    tokens = word_tokenize(caption.lower())
    vector = list()
    for token in tokens:
        encoded_token = encode_word(token, embeddings)
        vector.append(encoded_token)
    return np.array(vector)


def encode_word(word, embeddings):
    """ convert word to its word embedding vector """
    if word in embeddings.keys():
        vec = embeddings[word]
    else:
        '''
        w = Word(word)
        w = w.spellcheck()[0][0]
        if w in embeddings.keys():
            vec = embeddings[w]
        else:
            vec = np.zeros(100)
        '''
        vec = np.zeros(100)
    return vec


def decode_caption(vector, embeddings):
    """ convert word embedding array to caption """
    caption = ''
    for v in vector:
        caption += decode_word(v, embeddings) + ' '
    return caption


def decode_word(vec, embeddings):
    """ convert word embedding vector to the corresponding word """
    word = ''
    min_distance = 999
    for key in embeddings.keys():
        dist = distance.euclidean(embeddings[key], vec)
        if dist < min_distance:
            min_distance = dist
            word = key
    return word


if __name__ == '__main__':
    glove = 'dataset/glove.6B.100d.txt'
    word_embeddings = load_embeddings(glove)
    print('===Encode===')
    encoded_caption = encode_caption('Hello World', word_embeddings)
    print(encoded_caption)
    print('===Decode===')
    decoded_caption = decode_caption(encoded_caption, word_embeddings)
    print(decoded_caption)
