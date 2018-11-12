import os
import json
import pickle

from model import ROITagRCNN
from preprocess import decode_tags
from visualize import draw_boxes_and_tags
from train_roi_tags import RoiTagConfig, VisualGenomeDataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "roitag_rcnn.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../dataset/visual genome")


class InferenceConfig(RoiTagConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_NMS_THRESHOLD = 0.7

    DETECTION_MAX_INSTANCES = 10

    def __init__(self, num_classes):
        super(InferenceConfig, self).__init__(num_classes)


# load word to indices mappings and embeddings
class_id_to_tag_file = '../dataset/class_id_to_tag.pickle'
tag_to_class_id_file = '../dataset/tag_to_class_id.pickle'
with open(class_id_to_tag_file, 'rb') as f1, open(tag_to_class_id_file, 'rb') as f2:
    class_id_to_tag = pickle.load(f1)
    tag_to_class_id = pickle.load(f2)

config = InferenceConfig(len(class_id_to_tag))
config.display()

data_directory = '../dataset/visual genome/'
image_meta_file_path = '../dataset/image_data.json'
data_file_path = '../dataset/region_descriptions.json'

# Load a random image from the images folder
with open(image_meta_file_path, 'r', encoding='utf-8') as file:
    image_meta_data = json.loads(file.read())
image_ids_list = [meta['image_id'] for meta in image_meta_data]
image_ids = image_ids_list[100000:]

# Validation dataset
dataset_test = VisualGenomeDataset(tag_to_class_id, class_id_to_tag)
dataset_test.load_visual_genome(data_directory, image_ids,
                                image_meta_file_path, data_file_path)
dataset_test.prepare()

image_id = dataset_test._image_ids[0]

# Create model object in inference mode.
model = ROITagRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on Visual Genome dense image captioning
model.load_weights(MODEL_PATH, by_name=True)

# Run detection
image = dataset_test.load_image(image_id)
results = model.generate_roi_tags([image], verbose=1)

# Visualize results
r = results[0]
tags = []
for tag_vector in r['tags']:
    tag_vector[tag_vector < config.DETECTION_MIN_CONFIDENCE] = 0
    tags.append(', '.join(decode_tags(tag_vector, class_id_to_tag)))

draw_boxes_and_tags(image, r['rois'], tags, title=f'Image {str(image_ids[image_id])}',
                    file_path='predicted.png')

ground_truth_rois, ground_truth_tags = dataset_test.load_original_rois_and_tags(image_id)
draw_boxes_and_tags(image, ground_truth_rois, ground_truth_tags, title=f'Image {str(image_ids[image_id])}',
                    file_path='ground_truth.png')
