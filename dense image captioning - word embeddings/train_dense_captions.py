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
from preprocess import encode_caption, load_embeddings, load_embeddings_model
from gensim.models import KeyedVectors

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

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50

    # Embedding size
    EMBEDDING_SIZE = 100

    # Padding size
    PADDING_SIZE = 15


class VisualGenomeDataset(utils.Dataset):
    def __init__(self, embeddings, padding_size):
        super().__init__()
        self.word_embeddings = embeddings
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
            caps.append(self.encode_region_caption(caption[0], self.word_embeddings))
        captions = pad_sequences(caps, maxlen=self.padding_size, padding='post', dtype='float').astype(np.float32)
        return rois, captions

    def encode_region_caption(self, caption, embeddings):
        """ Convert caption to word embedding vector """
        return encode_caption(caption, embeddings)


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
    config = DenseCapConfig()
    config.display()

    data_directory = '../dataset/visual genome/'
    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'
    glove_file = '../dataset/glove.6B.100d.txt'
    word2vec_file = '../dataset/glove.6B.100d.txt.word2vec'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    # image_ids = [int(s.split('.')[0]) for s in os.listdir(data_directory)]

    train_image_ids = image_ids_list[:90000]
    val_image_ids = image_ids_list[90000:100000]
    test_image_ids = image_ids_list[100000:]

    # load word embeddings
    # word_embeddings = load_embeddings(glove_file)
    word_embeddings = load_embeddings_model(glove_file, word2vec_file)

    # Training dataset
    dataset_train = VisualGenomeDataset(word_embeddings, config.PADDING_SIZE)
    dataset_train.load_visual_genome(data_directory, train_image_ids,
                                     image_meta_file_path, data_file_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VisualGenomeDataset(word_embeddings, config.PADDING_SIZE)
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
                epochs=200,
                layers="lstm_only")

    end_time = time.time()
    print(end_time - start_time)
