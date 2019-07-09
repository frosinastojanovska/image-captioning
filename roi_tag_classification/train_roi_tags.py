import os
import time
import json
import pickle
import numpy as np
import skimage.io as skiimage_io
import skimage.color as skimage_color

from utils import Dataset
from config import Config
from model import ROITagRCNN
from preprocess import encode_region_tags, decode_tags, load_corpus, tokenize_corpus

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class RoiTagConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "roitag_rcnn"

    # Train on 1 GPU and n images per GPU. Batch size is n (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    LEARNING_RATE = 0.001

    def __init__(self, num_classes):
        super(RoiTagConfig, self).__init__()
        # Vocabulary size
        self.NUM_CLASSES = num_classes


class VisualGenomeDataset(Dataset):
    def __init__(self, tags_to_class_id, class_ids_to_tag):
        super(VisualGenomeDataset, self).__init__()
        self.tag_to_class_id = tags_to_class_id
        self.class_id_to_tag = class_ids_to_tag

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

    def load_rois_and_tags(self, image_id):
        """ Generate instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = []
        tags = []
        for roi, caption in zip(image_info['rois'], image_info['captions']):
            tag_vec = self.encode_region_tags(caption[0])
            tags.append(tag_vec)
            rois.append(roi)
        tags = np.array(tags)
        rois = np.array(rois)
        return rois, tags

    def load_original_rois_and_tags(self, image_id):
        """ Returns string instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = np.array(image_info['rois'])
        captions = image_info['captions']
        tags = []
        for caption in captions:
            tag_vec = self.encode_region_tags(caption[0])
            tags_current = decode_tags(tag_vec, self.class_id_to_tag)
            tags.append(', '.join(tags_current))
        return rois, tags

    def encode_region_tags(self, caption):
        """ Convert caption to encoded tag vector """
        return encode_region_tags(caption, self.tag_to_class_id)


if __name__ == '__main__':
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs_dense_img_cap")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")

    data_directory = '../dataset/visual genome/'
    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    train_image_ids = image_ids_list[:90000]
    val_image_ids = image_ids_list[90000:100000]
    test_image_ids = image_ids_list[100000:]

    # load tags ids
    class_id_to_tag_file = '../dataset/class_id_to_tag.pickle'
    tag_to_class_id_file = '../dataset/tag_to_class_id.pickle'

    if not os.path.exists(class_id_to_tag_file) or not os.path.exists(tag_to_class_id_file):
        tokens = tokenize_corpus(data_file_path, train_image_ids)
        tag_to_class_id, class_id_to_tag = load_corpus(list(tokens))

        with open(class_id_to_tag_file, 'wb') as f:
            pickle.dump(class_id_to_tag, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tag_to_class_id_file, 'wb') as f:
            pickle.dump(tag_to_class_id, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(class_id_to_tag_file, 'rb') as f:
            class_id_to_tag = pickle.load(f)
        with open(tag_to_class_id_file, 'rb') as f:
            tag_to_class_id = pickle.load(f)

    config = RoiTagConfig(len(class_id_to_tag))
    config.display()

    # Training dataset
    dataset_train = VisualGenomeDataset(tag_to_class_id, class_id_to_tag)
    dataset_train.load_visual_genome(data_directory, train_image_ids,
                                     image_meta_file_path, data_file_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VisualGenomeDataset(tag_to_class_id, class_id_to_tag)
    dataset_val.load_visual_genome(data_directory, val_image_ids,
                                   image_meta_file_path, data_file_path)
    dataset_val.prepare()

    init_with = 'coco'

    # Create model in training mode
    model = ROITagRCNN(mode="training", config=config,
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
                epochs=200,
                layers="3+")

    end_time = time.time()
    print(end_time - start_time)
