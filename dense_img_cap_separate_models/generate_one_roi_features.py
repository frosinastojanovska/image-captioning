import os
import numpy as np

from modified_dense_model import DenseImageCapRCNN
from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../dataset/visual genome")


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
    PADDING_SIZE = 5

    # Reduce word embeddings
    REDUCE_EMBEDDINGS = True


class InferenceConfig(DenseCapConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()


def load_model():
    # Create model object in inference mode.
    model = DenseImageCapRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on Visual Genome dense image captioning
    model.load_weights(MODEL_PATH, by_name=True)

    return model


def generate_features(dataset, image_id, model):
    # Run detection
    image = dataset.load_image(image_id)
    rois, _ = dataset.load_captions_and_rois(image_id)
    rois = np.expand_dims(rois, axis=0)
    results = model.generate_captions([image], rois, verbose=0)
    features = results[0]['features']
    return features
