import os
import utils
import model as modellib
from config import Config


class VisualGenomeConfig(Config):
    """

    """
    NAME = "VisualGenome"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 0


def load_trained_rcnn_model():
    # Root directory of the project
    root_dir = os.getcwd()

    # Directory to save logs and trained model
    model_dir = os.path.join(root_dir, "logs")

    # Local path to trained weights file
    coco_model_path = os.path.join(root_dir, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(coco_model_path):
        utils.download_trained_weights(coco_model_path)

    config = VisualGenomeConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="training", model_dir=model_dir, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(coco_model_path, by_name=True)


def train_model(model, config):
    # Root directory of the project
    root_dir = os.getcwd()
    # Directory to save logs and trained model
    model_dir = os.path.join(root_dir, "logs")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='all')

    model_path = os.path.join(model_dir, "mask_rcnn_visual_genome.h5")
    model.keras_model.save_weights(model_path)
