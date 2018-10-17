import json
import numpy as np
from tqdm import tqdm
from nltk.tag import pos_tag
from string import punctuation
from collections import Counter
from nltk.tokenize import word_tokenize


def load_corpus(tokens):
    class_id_to_tag = dict()
    tag_to_class_id = dict()

    for i in range(len(tokens)):
        class_id_to_tag[i] = tokens[i]
        tag_to_class_id[tokens[i]] = i
    return tag_to_class_id, class_id_to_tag


def tokenize_corpus(data_file, train):
    with open(data_file, 'r', encoding='utf-8') as doc:
        data = json.loads(doc.read())
    corpus = list()
    for data_point in tqdm(data):
        if data_point['id'] in train:
            regs = data_point['regions']
            for reg in regs:
                caption = reg['phrase']
                tokens = word_tokenize(caption.lower())
                tags = pos_tag(tokens, tagset='universal', lang='eng')
                for token, tag in tags:
                    if tag in ['ADJ', 'NOUN']:
                        corpus.append(token)
    frequencies = sorted(Counter(corpus).items(), key=lambda x: x[1], reverse=True)
    return set([x[0] for x in frequencies if x[1] >= 15 and x[0] not in punctuation])


def encode_region_tags(caption, tag_to_class_id):
    """ convert caption from string to encoded tag classes vector """
    tokens = word_tokenize(caption.lower())
    tags = pos_tag(tokens, tagset='universal', lang='eng')
    vector = list()
    for token, tag in tags:
        if tag in ['ADJ', 'NOUN']:
            encoded_token = encode_tag(token, tag_to_class_id)
            if encoded_token != -1:
                vector.append(encoded_token)
    tag_vector = np.zeros(len(tag_to_class_id))
    tag_vector[vector] = 1
    return tag_vector


def encode_tag(tag, tag_to_class_id):
    """ convert word to its index"""
    if tag in tag_to_class_id.keys():
        pos = tag_to_class_id[tag]
    else:
        pos = -1
    return pos


def decode_tags(vector, class_id_to_tag):
    """ convert one-hot encoding array to tags """
    indices, = np.where(vector > 0)
    tags = [class_id_to_tag[ind] for ind in indices]
    return tags
