import pickle

import numpy as np

from vgg16 import load_image, load_vgg16
from feature_generation.generate_roi_features import generate_features, load_model
import json
from nltk.tokenize import word_tokenize

counter = 0

START = "__START__ "
END = " __END__"


def encode_image(model, image,
                 data_dir="/home/shagun/projects/Image-Caption-Generator/data/"):
    '''Method to encode the given image'''
    image_dir = data_dir + "flickr8k/"
    prediction = generate_features(image_dir + str(image), model)
    return prediction


def read_captions(dataset='flickr8k', data_dir='../dataset/', mode='all'):
    if dataset == 'flickr8k':
        return read_captions_flickr(data_dir, mode)
    else:
        return read_captions_mscoco(data_dir, mode)


def read_captions_mscoco(data_dir, mode):
    with open(data_dir + 'captions_' + mode + '2014.json', 'r', encoding='utf-8') as file:
        image_data = json.loads(file.read())
    image_map = dict()
    for data in image_data['images']:
        image_map[data['id']] = data['file_name']
    captions = dict()
    for data in image_data['annotations']:
        caption = START + ' '.join(word_tokenize(data['caption'].lower())) + END
        if image_map[data['image_id']] in captions.keys():
            captions[image_map[data['image_id']]].append(caption)
        else:
            captions[image_map[data['image_id']]] = [caption]
    return captions


def read_captions_flickr(data_dir, mode):
    '''Method to read the captions from the text file'''
    with open(data_dir + "Flickr8k.token.txt") as caption_file:
        image_caption_dict = {}
        captions = map(lambda x: x.strip(),
                       caption_file.read().split('\n'))
        for caption in captions:
            image_name = caption.split('#')[0].strip()
            caption_text = caption.split('\t')[1].strip()
            if image_name not in image_caption_dict:
                image_caption_dict[image_name] = []
            image_caption_dict[image_name].append(START + caption_text + END)
    if mode == "all":
        return image_caption_dict
    else:
        image_name_list = read_image_list(dataset='flickr8k', mode=mode, data_dir=data_dir)
        filtered_image_caption_list = {}
        for image_name in image_name_list:
            filtered_image_caption_list[image_name] = image_caption_dict[image_name]
        return filtered_image_caption_list


def read_image_list(dataset='flickr8k', mode='train', data_dir='../dataset/'):
    if dataset == 'flickr8k':
        return read_image_list_flickr(mode, data_dir)
    else:
        return read_image_list_mscoco(mode, data_dir)


def read_image_list_mscoco(mode='train', data_dir='../dataset/'):
    with open(data_dir + 'captions_' + mode + '2014.json', 'r', encoding='utf-8') as file:
        image_data = json.loads(file.read())
    image_ids_list = [data['file_name'] for data in image_data['images']]
    return image_ids_list


def read_image_list_flickr(mode='train', data_dir='../dataset/'):
    '''Method to read the list of images'''
    with open(data_dir + "Flickr_8k." + mode + "Images.txt", 'r') as images_file:
        images = list(map(lambda x: x.strip(),
                          images_file.read().split('\n')))
    return images


def prepare_image_dataset(data_dir="/home/shagun/projects/Image-Caption-Generator/data/",
                          mode_list=["train", "test", "debug"]):
    image_encoding_model = load_model()
    for mode in mode_list:
        images = read_image_list(mode=mode,
                                 data_dir=data_dir)
        image_encoding = {}
        for image in images:
            image_encoding[image] = encode_image(image_encoding_model,
                                                 image,
                                                 data_dir=data_dir)
            with open(data_dir + "model/" + mode + "_image_encoding.pkl", "wb") as image_encoding_file:
                pickle.dump(image_encoding, image_encoding_file)


if __name__ == "__main__":
    data_dir="../dataset/"
    # image_ids_list = read_image_list(dataset='mscoco', mode='train', data_dir=data_dir)
    image_caption_dict = read_captions(dataset='mscoco', data_dir=data_dir, mode='train')
    prepare_image_dataset(data_dir=data_dir,
                          mode_list=["debug", "dev", "test", "train"])
