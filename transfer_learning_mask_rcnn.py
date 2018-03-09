import os
import utils
import mask_rcnn_model as modellib
from config import Config


class InitialConfig(Config):
    NAME = "initial"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2


def load_trained_model():
    # Root directory of the project
    root_dir = os.getcwd()

    # Directory to save logs and trained model
    model_dir = os.path.join(root_dir, "logs")

    # Local path to trained weights file
    coco_model_path = os.path.join(root_dir, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(coco_model_path):
        utils.download_trained_weights(coco_model_path)

    config = InitialConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_dir)

    # Load weights trained on MS-COCO
    model.load_weights(coco_model_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"]
                       )

    return model


def change_and_save_model(model):
    for i in range(42):
        model.keras_model.layers.pop()

    print(model.keras_model.summary())
    model.keras_model.save_weights('rcnn_coco.h5', overwrite=True)


if __name__ == '__main__':
    rcnn_model = load_trained_model()
    change_and_save_model(rcnn_model)
