import os
import json
import random
import pickle
import skimage.io
import numpy as np

from dense_model import DenseImageCapRCNN
from preprocess import decode_caption
from visualize import draw_boxes_and_captions
from train_dense_captions import DenseCapConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "img_cap_dense.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../dataset/visual genome")


class InferenceConfig(DenseCapConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_NMS_THRESHOLD = 0.7
    
    # Padding size
    PADDING_SIZE = 10

    def __init__(self, vocab_size, embedding_weights):
        super(InferenceConfig, self).__init__(vocab_size, embedding_weights)


# load word to indices mappings and embeddings
id_to_word_file = '../dataset/id_to_word.pickle'
word_to_id_file = '../dataset/word_to_id.pickle'
embedding_matrix_file = '../dataset/embedding_matrix.pickle'
id_to_word = pickle.load(open(id_to_word_file, 'rb'))
word_to_id = pickle.load(open(word_to_id_file, 'rb'))
embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'))

config = InferenceConfig(len(word_to_id), embedding_matrix)
config.display()

# Create model object in inference mode.
model = DenseImageCapRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on Visual Genome dense image captioning
model.load_weights(MODEL_PATH, by_name=True)

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
file_name = random.choice(file_names)
# file_name = '56.jpg'
image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

data_file_path = '../dataset/region_descriptions.json'

with open(data_file_path, 'r', encoding='utf-8') as doc:
    img_data = json.loads(doc.read())

data = []
i = int(file_name.split('.jpg')[0])

for x in img_data:
    if x['id'] == i:
        data = x['regions']
        break

ground_truth_captions = [[d['phrase']] for d in data]
ground_truth_rois = np.array([[d['y'], d['x'], d['y'] + d['height'], d['x'] + d['width']] for d in data])

# Run detection
results = model.generate_captions([image], verbose=1)

# Visualize results
r = results[0]
captions = []
for caption in r['captions']:
    captions.append(decode_caption(caption, id_to_word))
# draw_boxes_and_captions(image, ground_truth_rois, ground_truth_captions, file_name)
draw_boxes_and_captions(image, r['rois'], captions, file_name)
