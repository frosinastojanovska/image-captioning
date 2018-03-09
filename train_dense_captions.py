import os
import time
import json
import skimage.io as skiimage_io
import skimage.color as skimage_color

from config import Config
import utils
import dense_model as modellib


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
    IMAGES_PER_GPU = 8

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32


class VisualGenomeDataset(utils.Dataset):
    def load_visual_genome(self, data_dir, image_ids, image_meta_file, data_file):
        with open(data_file, 'r', encoding='utf-8') as doc:
            data = json.loads(doc.read())

        with open(image_meta_file, 'r', encoding='utf-8') as doc:
            image_meta = json.loads(doc.read())

        # Add images
        for i in image_ids:
            self.add_image(
                "VisualGenome", image_id=i,
                path=os.path.join(data_dir, '{}.jpg'.format(i)),
                width=image_meta[i-1]['width'],
                height=image_meta[i-1]['height'],
                rois=[[d['x'], d['y'], d['width'], d['height']] for d in data[i-1]['regions']],
                captions=[[d['phrase']] for d in data[i-1]['regions']]
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
        rois = image_info['rois']
        captions = image_info['captions']
        return rois, captions


if __name__ == '__main__':
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs_dense_img_cap")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "rcnn_coco.h5")
    config = DenseCapConfig()
    config.display()

    data_directory = 'dataset/visual genome/'
    image_meta_file_path = 'dataset/image_data.json'
    data_file_path = 'dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    train_image_ids = image_ids_list[54:56]
    val_image_ids = [62, 65]  # image_ids_list[5:6]
    test_image_ids = image_ids_list[6:8]

    # Training dataset
    dataset_train = VisualGenomeDataset()
    dataset_train.load_visual_genome(data_directory, train_image_ids, image_meta_file_path, data_file_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VisualGenomeDataset()
    dataset_val.load_visual_genome(data_directory, val_image_ids, image_meta_file_path, data_file_path)
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
    # TODO: CHANGE TO TRAIN THE LAYERS C4+ (or something else)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")

    end_time = time.time()
    print(end_time - start_time)
