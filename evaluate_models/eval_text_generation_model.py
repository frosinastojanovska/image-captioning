import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import skimage.io as skiimage_io
import skimage.color as skimage_color
from keras.preprocessing.sequence import pad_sequences

from config import Config
from utils import Dataset
from eval.bleu.bleu import Bleu
from eval.spice.spice import Spice
from eval.rouge.rouge import Rouge
from eval.cider.cider import Cider
from eval.meteor.meteor import Meteor
from eval.tokenizer.ptbtokenizer import PTBTokenizer
from dense_img_cap_separate_models.text_generation_model import build_lstm_model
from dense_img_cap_separate_models.preprocess import encode_caption, decode_caption
from dense_img_cap_separate_models.generate_one_roi_features import generate_features, load_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class DenseCapConfig(Config):
    """Configuration for training on the dense caption dataset.
    Derives from the base Config class and overrides values specific
    to the caption dataset.
    """
    # Give the configuration a recognizable name
    NAME = "dense image captioning"

    # Train on 1 GPU and n images per GPU. Batch size is n (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BATCH_SIZE = 10

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50

    # Padding size
    PADDING_SIZE = 10

    def __init__(self, vocab_size, embedding_weights, batch_size):
        super(DenseCapConfig, self).__init__()
        # Vocabulary size
        self.VOCABULARY_SIZE = vocab_size
        self.EMBEDDING_WEIGHTS = embedding_weights
        self.EMBEDDING_SIZE = embedding_weights.shape[1]
        self.BATCH_SIZE = batch_size


class VisualGenomeDataset(Dataset):
    def __init__(self, words_to_ids, padding_size):
        super(VisualGenomeDataset, self).__init__()
        self.word_to_id = words_to_ids
        self.padding_size = padding_size
        self.rois = None

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
                # add <start> and <end> tokens
                cap = np.hstack((np.array(1), cap, np.array(2))) if len(cap) < (self.padding_size - 2) \
                    else np.hstack((np.array(1), cap[:(self.padding_size - 2)], np.array(2)))
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


def generate_predictions(model, model_name, dataset, features_model, config, id_to_word):
    if os.path.exists('dataset/' + model_name + '_predictions.pickle'):
        with open('dataset/' + model_name + '_predictions.pickle', 'rb') as f:
            predictions = pickle.load(f)
    else:
        predictions = []
        separator = ' <end>'
        print('Generating predictions...')

        for image_id in tqdm(dataset.image_ids):
            features = generate_features(dataset, image_id, features_model)
            _, captions = dataset.load_original_captions_and_rois(image_id)
            steps = features.shape[0]
            for i in range(steps):
                f = features[i]
                c = captions[i]
                res = model.predict(np.array([f]), batch_size=config.BATCH_SIZE)
                predicted_caption = decode_caption(res[0], id_to_word).split(separator, 1)[0]
                real_caption = c[0]
                predictions.append({'p': predicted_caption, 'r': real_caption})

        with open('dataset/' + model_name + '_predictions.pickle', 'wb') as f:
            pickle.dump(predictions, f, protocol=2)

    return predictions


def score_predictions(predictions, model_name):
    print('Scoring predictions...')
    tokenizer = PTBTokenizer()
    gens = {}
    refs = {}
    for p, i in zip(predictions, range(len(predictions))):
        gens[str(i)] = [p['p']]
        refs[str(i)] = [p['r']]
    gens = tokenizer.tokenize(gens)
    refs = tokenizer.tokenize(refs)
    refs.update({key: [] for key in gens if key not in refs})

    if not os.path.exists('dataset/' + model_name + '_spice_scores.pickle'):
        print('Calculating SPICE...')
        spice_scorer = Spice()
        spice_avg_score, spice_all_scores = spice_scorer.compute_score(refs, gens)
        spice_all_scores = np.array([s['All']['f'] for s in spice_all_scores])
        with open('dataset/' + model_name + '_spice_scores.pickle', 'wb') as f:
            pickle.dump((spice_avg_score, spice_all_scores), f, protocol=2)

    if not os.path.exists('dataset/' + model_name + '_rouge_scores.pickle'):
        print('Calculating ROUGE...')
        rouge_scorer = Rouge()
        rouge_avg_score, rouge_all_scores = rouge_scorer.compute_score(refs, gens)
        with open('dataset/' + model_name + '_rouge_scores.pickle', 'wb') as f:
            pickle.dump((rouge_avg_score, rouge_all_scores), f, protocol=2)

    if not os.path.exists('dataset/' + model_name + '_meteor_scores.pickle'):
        print('Calculating METEOR...')
        meteor_scorer = Meteor()
        meteor_avg_score, meteor_all_scores = meteor_scorer.compute_score(refs, gens)
        with open('dataset/' + model_name + '_meteor_scores.pickle', 'wb') as f:
            pickle.dump((meteor_avg_score, meteor_all_scores), f, protocol=2)

    if not os.path.exists('dataset/' + model_name + '_cider_scores.pickle'):
        print('Calculating CIDER...')
        cider_scorer = Cider()
        cider_avg_score, cider_all_scores = cider_scorer.compute_score(refs, gens)
        with open('dataset/' + model_name + '_cider_scores.pickle', 'wb') as f:
            pickle.dump((cider_avg_score, cider_all_scores), f, protocol=2)

    if not os.path.exists('dataset/' + model_name + '_bleu1_scores.pickle'):
        print('Calculating BLEU...')
        bleu_scorer = Bleu(4)
        bleu_avg_score, bleu_all_scores = bleu_scorer.compute_score(refs, gens)
        for n, bleu_n_avg_score, bleu_n_all_scores in zip(range(len(bleu_avg_score)), bleu_avg_score, bleu_all_scores):
            with open('dataset/' + model_name + '_bleu' + str(n + 1) + '_scores.pickle', 'wb') as f:
                pickle.dump((bleu_n_avg_score, bleu_n_all_scores), f, protocol=2)


if __name__ == '__main__':
    embeddings_file_path = 'dataset/glove.6B.300d.txt'

    data_directory = 'dataset/visual genome/'
    image_meta_file_path = 'dataset/image_data.json'
    data_file_path = 'dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    train_val_image_ids = image_ids_list[:100000]
    test_image_ids = image_ids_list[100000:]

    # load word ids
    id_to_word_file = 'dataset/dense_img_cap/id_to_word.pickle'
    word_to_id_file = 'dataset/dense_img_cap/word_to_id.pickle'
    embedding_matrix_file = 'dataset/dense_img_cap/embedding_matrix.pickle'

    with open(id_to_word_file, 'rb') as f:
        id_to_word = pickle.load(f)
    with open(word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)
    with open(embedding_matrix_file, 'rb') as f:
        embedding_matrix = pickle.load(f)

    features_model = load_model()

    config = DenseCapConfig(len(id_to_word), embedding_matrix, 1)
    config.display()

    model = build_lstm_model([7, 7, 256], config, 512, 'inference')

    # Test dataset
    dataset_test = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
    dataset_test.load_visual_genome(data_directory, test_image_ids, image_meta_file_path,
                                    data_file_path)
    dataset_test.prepare()
    model.load_weights('evaluate_models/models/model3-47-1.74.h5')
    model_name = 'm3-47'

    preds = generate_predictions(model, model_name, dataset_test, features_model, config, id_to_word)
    score_predictions(preds, model_name)
