import os
import time
import json
import pickle
import numpy as np
import skimage.io as skiimage_io
import skimage.color as skimage_color
from keras.preprocessing.sequence import pad_sequences

from utils import Dataset
from config import Config
from dense_model import DenseImageCapRCNN
from preprocess import encode_caption, load_corpus, tokenize_corpus, load_embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class DenseCapConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "dense image captioning"

    # Train on 1 GPU and n images per GPU. Batch size is n (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50

    # Padding size
    PADDING_SIZE = 15

    def __init__(self, vocab_size, embedding_weights):
        super(DenseCapConfig, self).__init__()
        # Vocabulary size
        self.VOCABULARY_SIZE = vocab_size
        self.EMBEDDING_WEIGHTS = embedding_weights
        self.EMBEDDING_SIZE = embedding_weights.shape[1]


class VisualGenomeDataset(Dataset):
    def __init__(self, words_to_ids, padding_size):
        super(VisualGenomeDataset, self).__init__()
        self.word_to_id = words_to_ids
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
        """ Return a link to the image in the Stanford Website. """
        info = self.image_info[image_id]
        return "https://cs.stanford.edu/people/rak248/VG_100K/{}.jpg".format(info["id"])

    def load_image(self, image_id):
        """ Load the specified image and return a [H,W,3] Numpy array. """
        # Load image
        image = skiimage_io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage_color.gray2rgb(image)
        return image

    def load_captions_and_rois(self, image_id):
        """ Generate instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = []
        caps = []
        for roi, caption in zip(image_info['rois'], image_info['captions']):
            cap = self.encode_region_caption(caption[0])
            if cap.size != 0:
                rois.append(roi)
                caps.append(cap)
        captions = pad_sequences(caps, maxlen=self.padding_size, padding='post', dtype='float').astype(np.float32)
        rois = np.array(rois)
        return rois, captions

    def load_original_captions_and_rois(self, image_id):
        """ Returns string instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = np.array(image_info['rois'])
        captions = image_info['captions']
        return rois, captions

    def encode_region_caption(self, caption):
        """ Convert caption to word embedding vector """
        return encode_caption(caption, self.word_to_id)


if __name__ == '__main__':
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs_dense_img_cap")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")

    embeddings_file_path = '../dataset/glove.6B.300d.txt'

    data_directory = '../dataset/visual genome/'
    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    train_image_ids = image_ids_list[:90000]
    val_image_ids = image_ids_list[90000:100000]
    test_image_ids = image_ids_list[100000:]

    # load word ids
    id_to_word_file = '../dataset/dense_img_cap/id_to_word.pickle'
    word_to_id_file = '../dataset/dense_img_cap/word_to_id.pickle'
    embedding_matrix_file = '../dataset/dense_img_cap/embedding_matrix.pickle'

    if not os.path.exists(id_to_word_file) or not os.path.exists(word_to_id_file) \
            or not os.path.exists(embedding_matrix_file):
        embeddings = load_embeddings(embeddings_file_path)
        tokens = tokenize_corpus(data_file_path, train_image_ids, val_image_ids, embeddings)
        word_to_id, id_to_word, embedding_matrix = load_corpus(list(tokens), embeddings, 300)

        with open(id_to_word_file, 'wb') as f:
            pickle.dump(id_to_word, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(word_to_id_file, 'wb') as f:
            pickle.dump(word_to_id, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(embedding_matrix_file, 'wb') as f:
            pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(id_to_word_file, 'rb') as f:
            id_to_word = pickle.load(f)
        with open(word_to_id_file, 'rb') as f:
            word_to_id = pickle.load(f)
        with open(embedding_matrix_file, 'rb') as f:
            embedding_matrix = pickle.load(f)

    config = DenseCapConfig(len(id_to_word), embedding_matrix)
    config.display()

    # Training dataset
    dataset_train = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
    dataset_train.load_visual_genome(data_directory, train_image_ids,
                                     image_meta_file_path, data_file_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
    dataset_val.load_visual_genome(data_directory, val_image_ids,
                                   image_meta_file_path, data_file_path)
    dataset_val.prepare()

    init_with = 'coco'

    # Create model in training mode
    model = DenseImageCapRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    if init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else:
        # Load weights trained on MS COCO, but skip layers that
        # are different
        model.load_weights(COCO_MODEL_PATH, by_name=True)

    print(model.keras_model.summary())

    start_time = time.time()
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=250,
                layers="caption_only")

    end_time = time.time()
    print(end_time - start_time)
