import os
import random
import skimage.io

import dense_model as modellib
from train_dense_captions import DenseCapConfig
from preprocess import decode_caption, load_embeddings, load_embeddings_model
from gensim.models import KeyedVectors

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


config = InferenceConfig()
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

# load word embeddings
glove_file = '../dataset/glove.6B.100d.txt'
word2vec_file = '../dataset/glove.6B.100d.txt.word2vec'
# word_embeddings = load_embeddings(glove_file)
word_embeddings = load_embeddings_model(glove_file, word2vec_file)

# Visualize results
print(file_name)
print(results)
r = results[0]
print(r['rois'])
for caption in r['captions']:
    print(decode_caption(caption, word_embeddings))
