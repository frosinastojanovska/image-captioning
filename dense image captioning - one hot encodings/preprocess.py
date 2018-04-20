from nltk.tokenize import word_tokenize
from textblob import Word
import numpy as np
from scipy.spatial import distance
import os
from gensim.models import KeyedVectors
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import _pickle


def load_corpus(tokens):
    tokens = ['<UNK>'] + tokens
    word_to_vector = dict()
    id_to_word = dict()
    for i in range(len(tokens)):
        array = np.zeros(len(tokens))
        array[i] = 1
        word_to_vector[tokens[i]] = array
        id_to_word[i] = tokens[i]
    return word_to_vector, id_to_word


def encode_caption(caption, word_to_vector):
    """ convert caption from string to one-hot array """
    tokens = word_tokenize(caption.lower())
    vector = list()
    for token in tokens:
        encoded_token = encode_word(token, word_to_vector)
        vector.append(encoded_token)
    return np.array(vector)


def encode_word(word, word_to_vector):
    """ convert word to its one-hot vector"""
    if word in word_to_vector.keys():
        vec = word_to_vector[word]
    else:
        vec = word_to_vector['<UNK>']
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


if __name__ == '__main__':
    one_hot_encodings, word_mappings = load_corpus(['hello', 'the', 'it', 'she', 'he', 'world', 'name', 'hi'])
    with open('word_to_vector_pt1.pickle', 'wb') as handle:
        _pickle.dump(dict(list(one_hot_encodings.items())[:int(len(one_hot_encodings) / 2)]), handle, protocol=4)
    with open('word_to_vector_pt2.pickle', 'wb') as handle:
        _pickle.dump(dict(list(one_hot_encodings.items())[int(len(one_hot_encodings) / 2):]), handle, protocol=4)
    with open('id_to_word.pickle', 'wb') as handle:
        _pickle.dump(word_mappings, handle, protocol=4)
    one_hot_encodings_p1 = _pickle.load(open('word_to_vector_pt1.pickle', 'rb'))
    one_hot_encodings_p2 = _pickle.load(open('word_to_vector_pt2.pickle', 'rb'))
    one_hot_encodings = dict()
    one_hot_encodings.update(one_hot_encodings_p1)
    one_hot_encodings.update(one_hot_encodings_p2)
    word_mappings = _pickle.load(open('id_to_word.pickle', 'rb'))
    print('===Encode===')
    encoded_caption = encode_caption('Hello world xxx', one_hot_encodings)
    print(encoded_caption)
    print('===Decode===')
    decoded_caption = decode_caption(encoded_caption, word_mappings)
    print(decoded_caption)