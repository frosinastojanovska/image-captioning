import os
import time
import json
import numpy as np
import skimage.io as skiimage_io
import skimage.color as skimage_color
from keras.preprocessing.sequence import pad_sequences

import utils
from config import Config
import dense_model as modellib
from preprocess import encode_caption, load_corpus
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import _pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class DenseCapConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "dense image captioning"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

    # Padding size
    PADDING_SIZE = 15

    def __init__(self, vocab_size):
        super().__init__()
        # Vocabulary size
        self.VOCABULARY_SIZE = vocab_size


class VisualGenomeDataset(utils.Dataset):
    def __init__(self, word_to_vector, padding_size):
        super().__init__()
        self.word_to_vector = word_to_vector
        self.padding_size = padding_size

    def load_visual_genome(self, data_dir, image_ids, image_meta_file, data_file):
        with open(data_file, 'r', encoding='utf-8') as doc:
            img_data = json.loads(doc.read())
        data = dict()
        for x in img_data:
            data[x['id']] = x['regions']

        with open(image_meta_file, 'r', encoding='utf-8') as doc:
            img_meta = json.loads(doc.read())
        image_meta = dict()
        for x in img_meta:
            image_meta[x['image_id']] = x

        # Add images
        for i in image_ids:
            captions = [[d['phrase']] for d in data[i]]
            self.add_image(
                "VisualGenome", image_id=i,
                path=os.path.join(data_dir, '{}.jpg'.format(i)),
                width=image_meta[i]['width'],
                height=image_meta[i]['height'],
                rois=[[d['y'], d['x'], d['y'] + d['height'], d['x'] + d['width']] for d in data[i]],
                captions=captions
            )

    def image_reference(self, image_id):
        """Return a link to the image in the Stanford Website."""
        info = self.image_info[image_id]
        return "https://cs.stanford.edu/people/rak248/VG_100K/{}.jpg".format(info["id"])

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skiimage_io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage_color.gray2rgb(image)
        return image

    def load_captions_and_rois(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        image_info = self.image_info[image_id]
        rois = np.array(image_info['rois'])
        captions = image_info['captions']
        caps = []
        for caption in captions:
            caps.append(self.encode_region_caption(caption[0], self.word_to_vector))
        captions = pad_sequences(caps, maxlen=self.padding_size, padding='post', dtype='float').astype(np.float32)
        return rois, captions

    def encode_region_caption(self, caption, word_to_vector):
        """ Convert caption to word embedding vector """
        return encode_caption(caption, word_to_vector)


def tokenize_corpus(data_file, train, validation):
    with open(data_file, 'r', encoding='utf-8') as doc:
        data = json.loads(doc.read())
    corpus = set()
    for x in data:
        if x['id'] in train or x['id'] in validation:
            regs = x['regions']
            for reg in regs:
                caption = reg['phrase']
                tokens = word_tokenize(caption.lower())
                for token in tokens:
                    corpus.add(token)
    return corpus


if __name__ == '__main__':
    '''
    import keras.backend as K
    K.clear_session()
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True  # allocate dynamically
    sess = tf.Session(config=config)
    '''

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs_dense_img_cap")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../rcnn_coco.h5")

    data_directory = '../dataset/visual genome/'
    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    # image_ids = [int(s.split('.')[0]) for s in os.listdir(data_directory)]

    train_image_ids = image_ids_list[:90000]
    val_image_ids = image_ids_list[90000:]  # image_ids_list[5:6]
    test_image_ids = image_ids_list[6:8]

    # load one-hot encodings
    word_to_vector_file_pt1 = '../dataset/word_to_vector_pt1.pickle'
    word_to_vector_file_pt2 = '../dataset/word_to_vector_pt2.pickle'
    id_to_word_file = '../dataset/id_to_word.pickle'
    if not os.path.exists(word_to_vector_file_pt1) \
            or not os.path.exists(word_to_vector_file_pt2) \
            or not os.path.exists(id_to_word_file):
        tokens = tokenize_corpus(data_file_path, train_image_ids, val_image_ids)
        word_to_vector, id_to_word = load_corpus(list(tokens))
        with open(word_to_vector_file_pt1, 'wb') as handle:
           _pickle.dump(dict(list(word_to_vector.items())[:int(len(word_to_vector) / 2)]), handle, protocol=4)
        with open(word_to_vector_file_pt2, 'wb') as handle:
           _pickle.dump(dict(list(word_to_vector.items())[int(len(word_to_vector) / 2):]), handle, protocol=4)
        with open(id_to_word_file, 'wb') as handle:
           _pickle.dump(id_to_word, handle, protocol=4)
    else:
        word_to_vector_pt1 = _pickle.load(open(word_to_vector_file_pt1, 'rb'))
        word_to_vector_pt2 = _pickle.load(open(word_to_vector_file_pt2, 'rb'))
        word_to_vector = dict()
        word_to_vector.update(word_to_vector_pt1)
        word_to_vector.update(word_to_vector_pt2)
        id_to_word = _pickle.load(open(id_to_word_file, 'rb'))
    config = DenseCapConfig(len(word_to_vector))
    config.display()

    # Training dataset
    dataset_train = VisualGenomeDataset(word_to_vector, config.PADDING_SIZE)
    dataset_train.load_visual_genome(data_directory, train_image_ids,
                                     image_meta_file_path, data_file_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VisualGenomeDataset(word_to_vector, config.PADDING_SIZE)
    dataset_val.load_visual_genome(data_directory, val_image_ids,
                                   image_meta_file_path, data_file_path)
    dataset_val.prepare()

    init_with = 'coco'

    # Create model in training mode
    model = modellib.DenseImageCapRCNN(mode="training", config=config,
                                       model_dir=MODEL_DIR)

    if init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else:
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different
        # TODO: correct this!!!!!!!!!!!!!
        model.load_weights(COCO_MODEL_PATH, by_name=True)

    start_time = time.time()
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=50,
                layers="4+")

    end_time = time.time()
    print(end_time - start_time)
