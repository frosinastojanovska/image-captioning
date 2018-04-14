import os
import random
import skimage.io

import dense_model as modellib
from train_dense_captions import DenseCapConfig
from preprocess import decode_caption
from gensim.models import KeyedVectors
import _pickle

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

    def __init__(self, vocab_size):
        super().__init__(vocab_size)


# load one-hot encodings
word_to_vector_file = '../dataset/word_to_vector.pickle'
id_to_word_file = '../dataset/id_to_word.pickle'
word_to_vector = _pickle.load(open(word_to_vector_file, 'rb'))
id_to_word = _pickle.load(open(id_to_word_file, 'rb'))

config = InferenceConfig(len(word_to_vector))
config.display()

# Create model object in inference mode.
model = modellib.DenseImageCapRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on Visual Genome dense image captioning
model.load_weights(MODEL_PATH, by_name=True)

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
# test_image_ids = image_ids_list[6:8]
file_name = random.choice(file_names)
image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

# Run detection
results = model.generate_captions([image], verbose=1)

# Visualize results
print(file_name)
print(results)
r = results[0]
print(r['rois'])
for caption in r['captions']:
    print(decode_caption(caption, id_to_word))
