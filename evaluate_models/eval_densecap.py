import os
import json
import pickle
import numpy as np
import skimage.io as skiimage_io
import skimage.color as skimage_color
from tqdm import tqdm

from utils import Dataset, compute_overlaps
from preprocess import decode_caption, encode_caption_v2
from dense_model import DenseImageCapRCNN
from config import Config


from eval.spice.spice import Spice
from eval.rouge.rouge import Rouge
from eval.meteor.meteor import Meteor
from eval.cider.cider import Cider
from eval.bleu.bleu import Bleu
from eval.tokenizer.ptbtokenizer import PTBTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class DenseCapConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "dense image captioning"

    # Train on 1 GPU and n images per GPU. Batch size is n (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BATCH_SIZE = 64

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50

    # Padding size
    PADDING_SIZE = 10

    def __init__(self, vocab_size, embedding_weights):
        super(DenseCapConfig, self).__init__()
        # Vocabulary size
        self.VOCABULARY_SIZE = vocab_size
        self.EMBEDDING_WEIGHTS = embedding_weights
        self.EMBEDDING_SIZE = embedding_weights.shape[1]


class VisualGenomeDataset(Dataset):
    def __init__(self, words_to_ids, padding_size):
        super(VisualGenomeDataset, self).__init__()
        self.word_to_id = words_to_ids
        self.padding_size = padding_size

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

    def add_sequences(self, sequences):
        self.sequences = sequences

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
                caps.append(cap)
        # captions = pad_sequences(caps, maxlen=self.padding_size, padding='post', dtype='float').astype(np.float32)
        rois = np.array(rois)
        captions = np.array(caps)
        return rois, captions

    def load_original_captions_and_rois(self, image_id):
        """ Returns string instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = np.array(image_info['rois'])
        captions = image_info['captions']
        return rois, captions

    def encode_region_caption(self, caption):
        """ Convert caption to word embedding vector """
        return encode_caption_v2(caption, self.word_to_id)


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


class DenseCaptioningEvaluator:
    def __init__(self, model, dataset, id_to_word):
        """
        :param model: Dense Captioning keras model in inference mode
        :type model: keras model object
        :param text_metrics: language metrics, one of {'SPICE', 'METEOR'}
        :type text_metrics: str
        :param dataset: the test dataset
        :type dataset: VisualGenomeDataset object
        :param id_to_word: mapping from id to words from the vocabulary
        :type id_to_word: dict
        """
        self.min_overlaps = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.min_scores = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.model = model
        self.dataset = dataset
        self.id_to_word = id_to_word
        self.predictions = None
        self.ground_truths = None

    @staticmethod
    def merge_boxes(boxes, captions, thresh):
        """ The ground truth boxes are merged together if they overlap with IoU >= thresh.
        With this approach we group many overlapping boxes into one, with multiple caption
        references, instead of the original ground truth annotations that can be on top of
        each other.

        :param boxes: boxes representing the ROIs
        :type boxes: numpy.array
        :param captions: captions for each box
        :type captions: list(list(str))
        :param thresh: IoU threshold for overlapping boxes
        :type thresh: float
        :return: merged boxes, merged captions
        :rtype: (numpy.array, list(list(str)))
        """
        assert (thresh > 0)
        pairwise_iou = compute_overlaps(boxes, boxes)

        ix = []
        while True:
            good = (pairwise_iou > thresh).astype(int)
            good_sum = np.sum(good, axis=0)
            top_num = np.amax(good_sum, axis=0)
            if top_num == 0:
                break
            top_ix = np.argmax(good_sum, axis=0)
            merge_ix = np.nonzero(good[top_ix])
            ix.append(merge_ix)
            pairwise_iou[merge_ix] = 0
            pairwise_iou[:, merge_ix] = 0

        n = len(ix)
        new_boxes = np.zeros((n, 4))
        new_captions = []
        old_captions = np.array(captions)
        for i in range(n):
            ix_i = ix[i]
            if boxes[ix_i].shape[0] > 1:
                new_box = np.mean(boxes[ix_i], axis=0).astype(int)
            else:
                new_box = boxes[ix_i][0]
            new_boxes[i] = new_box
            new_captions.append(old_captions[ix_i].tolist())

        return new_boxes, new_captions

    def get_gt_captions(self, image_ids):
        """ Get ground truth dense captions for images with corresponding image ids

        :param image_ids: ids to the images from the dataset
        :type image_ids: list(int)
        :return: images, ground truth boxes, ground truth captions
        :rtype: list(numpy.array), list(numpy.array), list(list(str))
        """
        images = []
        gt_boxes = []
        gt_captions = []

        for image_id in image_ids:
            # Load image
            image = self.dataset.load_image(image_id)
            boxes, captions = self.dataset.load_original_captions_and_rois(image_id)

            # Merge ground truth boxes that overlap by >= 0.7 into a
            # single box with multiple reference captions
            merged_boxes, merged_captions = self.merge_boxes(boxes, captions, 0.7)

            images.append(image)
            gt_boxes.append(merged_boxes)
            gt_captions.append(merged_captions)

        return images, gt_boxes, gt_captions

    def get_generated_captions(self):
        """ Returns the generated dense captions from the model for the given images

        :param num_images: number of images
        :type num_images: int
        :param images: the images
        :type images: list(numpy.array)
        :return: detected boxes, generated captions, log probabilities for each captions (box)
        :rtype: (list(numpy.array), list(list(str)), list(numpy.array))
        """
        boxes = []
        captions = []
        log_probs = []
        for image_id in self.dataset._image_ids:
            image_info = self.dataset.image_info[image_id]
            with open(f'../dataset/densecap/{image_info["id"]}.json', 'r', encoding='utf-8') as file:
                predictions = json.loads(file.read())
            captions.append([[cap['caption']] for cap in predictions['output']['captions']])
            boxes.append(np.array([cap['bounding_box'] for cap in predictions['output']['captions']]))
            log_probs.append(np.array([cap['confidence'] for cap in predictions['output']['captions']]))

        return boxes, captions, log_probs

    @staticmethod
    def assign_detections_to_ground_truth(num_images, gt_boxes, gt_captions, boxes, captions, log_probs):
        """ Assign detected boxes to ground truth boxes. The detected boxes are assigned to one of
        the ground truth boxes according to the maximum IoU overlapping score, and if two detected
        boxes have the same score, then the detected box with greater generation probability is
        assigned to the corresponding ground truth box.

        :param num_images: number of images
        :type num_images: int
        :param gt_boxes: ground truth boxes
        :type gt_boxes: list(numpy.array)
        :param gt_captions: ground truth captions
        :type gt_captions: list(list(str))
        :param boxes: detected boxes
        :type boxes: list(numpy.array)
        :param captions: generated captions for each box
        :type captions: list(list(str))
        :param log_probs: log probabilities for each captions (box)
        :type log_probs: list(numpy.array)
        :return: {'ok': ok for the ground truth, 'ov': overlap score,
                  'candidate': generated caption, 'references': reference captions}
        :rtype: dict
        """
        results = []
        for i in range(num_images):
            indices = np.argsort(log_probs[i])[::-1]

            num_detections = log_probs[i].shape[0]
            overlaps = compute_overlaps(boxes[i], gt_boxes[i])
            assignments = np.argmax(overlaps, axis=1)
            overlaps_maxs = np.amax(overlaps, axis=1)
            used = set()
            records = []
            for d in range(num_detections):
                ind = indices[d]
                ov = overlaps_maxs[ind]
                ok = 0
                assignment = assignments[ind]
                if assignment not in used:
                    used.add(assignment)
                    ok = 1
                if ov > 0:
                    references = gt_captions[i][assignment]
                else:
                    references = []

                records.append({'ok': ok, 'ov': ov, 'candidate': captions[i][ind], 'references': references})
            results.append(records)

        return results

    def score_captions(self, records, method):
        """ Scores generated captions using the SPICE metrics

        :param records: records for aligned detections and generations with ground truths
        :type records: list(dict)
        :return: score for each caption (caption for each box)
        """
        assert (method in {'SPICE', 'ROUGE', 'METEOR', 'CIDER', 'BLEU'})
        if method == 'SPICE':
            scorer = Spice()
        elif method == 'ROUGE':
            scorer = Rouge()
        elif method == 'METEOR':
            scorer = Meteor()
        elif method == 'CIDER':
            scorer = Cider()
        else:
            scorer = Bleu(4)
        tokenizer = PTBTokenizer()
        scores = []
        gens = {}
        refs = {}
        i = 0
        for r in records:
            r1 = [x for x in r if len(x['references']) > 0]
            for rec in r1:
                gens[str(i)] = rec['candidate']
                refs[str(i)] = [sent[0] for sent in rec['references']]
                i += 1
        gens = tokenizer.tokenize(gens)
        refs = tokenizer.tokenize(refs)
        refs.update({key: [] for key in gens if key not in refs})
        avg_score, all_scores = scorer.compute_score(refs, gens)
        if method == 'SPICE':
            all_scores = [s['All']['f'] for s in all_scores]
        scores.append(all_scores)

        return scores

    def evaluate_captions(self):
        """
        Evaluate generated dense captions
        :return: {'map': map_value, 'ap_breakdown': ap_results, 'detmap': det_map, 'det_breakdown': det_results}
        :rtype: dict

        """
        num_images = self.dataset.num_images
        image_ids = self.dataset._image_ids

        _, gt_boxes, gt_captions = self.get_gt_captions(image_ids)
        boxes, captions, log_probs = self.get_generated_captions()

        records = self.assign_detections_to_ground_truth(num_images, gt_boxes, gt_captions,
                                                         boxes, captions, log_probs)
        # text scores
        if not os.path.exists('../dataset/densecap_scores_spice.pkl'):
            print('Evaluating with SPICE...')
            scores = self.score_captions(records, 'SPICE')
            scores = [item for sublist in scores for item in sublist]
            with open('../dataset/densecap_scores_spice.pkl', 'wb') as f:
                pickle.dump(scores, f)
        if not os.path.exists('../dataset/densecap_scores_rouge.pkl'):
            print('Evaluating with ROUGE...')
            scores = self.score_captions(records, 'ROUGE')
            scores = [item for sublist in scores for item in sublist]
            with open('../dataset/densecap_scores_rouge.pkl', 'wb') as f:
                pickle.dump(scores, f)
        if not os.path.exists('../dataset/densecap_scores_meteor.pkl'):
            print('Evaluating with METEOR...')
            scores = self.score_captions(records, 'METEOR')
            scores = [item for sublist in scores for item in sublist]
            with open('../dataset/densecap_scores_meteor.pkl', 'wb') as f:
                pickle.dump(scores, f)
        if not os.path.exists('../dataset/densecap_scores_cider.pkl'):
            print('Evaluating with CIDER...')
            scores = self.score_captions(records, 'CIDER')
            scores = [item for sublist in scores for item in sublist]
            with open('../dataset/densecap_scores_cider.pkl', 'wb') as f:
                pickle.dump(scores, f)
        if not os.path.exists('../dataset/densecap_scores_bleu_1.pkl'):
            print('Evaluating with BLEU...')
            scores = self.score_captions(records, 'BLEU')
            scores_1 = []
            scores_2 = []
            scores_3 = []
            scores_4 = []
            for score in scores:
                scores_1.append(score[0])
                scores_2.append(score[1])
                scores_3.append(score[2])
                scores_4.append(score[3])
            scores_1 = [item for sublist in scores_1 for item in sublist]
            scores_2 = [item for sublist in scores_2 for item in sublist]
            scores_3 = [item for sublist in scores_3 for item in sublist]
            scores_4 = [item for sublist in scores_4 for item in sublist]
            with open('../dataset/densecap_scores_bleu_1.pkl', 'wb') as f:
                pickle.dump(scores_1, f)
            with open('../dataset/densecap_scores_bleu_2.pkl', 'wb') as f:
                pickle.dump(scores_2, f)
            with open('../dataset/densecap_scores_bleu_3.pkl', 'wb') as f:
                pickle.dump(scores_3, f)
            with open('../dataset/densecap_scores_bleu_4.pkl', 'wb') as f:
                pickle.dump(scores_4, f)

    def evaluate(self, method):
        """
        Evaluate generated dense captions
        :return: {'map': map_value, 'ap_breakdown': ap_results, 'detmap': det_map, 'det_breakdown': det_results}
        :rtype: dict

        """
        num_images = self.dataset.num_images
        image_ids = self.dataset._image_ids

        _, gt_boxes, gt_captions = self.get_gt_captions(image_ids)
        boxes, captions, log_probs = self.get_generated_captions()

        records = self.assign_detections_to_ground_truth(num_images, gt_boxes, gt_captions,
                                                         boxes, captions, log_probs)
        # text scores
        scores = self.score_captions(records, method)

        # flatten everything
        records = [item for sublist in records for item in sublist]
        scores = [item for sublist in scores for item in sublist]

        npos = np.sum([box.shape[0] for box in gt_boxes])
        ap_results = {}
        det_results = {}
        for min_overlap in self.min_overlaps:
            for min_score in self.min_scores:
                # go down the list and build tp, fp arrays
                n = len(records)
                tp = np.zeros(n)
                fp = np.zeros(n)
                for i in range(n):
                    r = records[i]
                    if len(r['references']) == 0:
                        # nothing aligned to this predicted box in the ground truth
                        fp[i] = 1
                    else:
                        # ok something aligned. Lets check if it aligned enough, and correctly enough
                        score = scores[i]
                        if r['ov'] >= min_overlap and r['ok'] == 1 and score > min_score:
                            tp[i] = 1
                        else:
                            fp[i] = 1

                fp = np.cumsum(fp, axis=0)
                tp = np.cumsum(tp, axis=0)
                rec = tp / npos
                prec = tp / (tp + fp)

                # compute max-interpolated average precision
                ap = 0
                apn = 0
                for t in [n / 100 for n in range(0, 101)]:
                    mask = rec >= t
                    prec_masked = prec[mask]
                    if prec_masked.shape[0] > 0:
                        p = np.amax(prec_masked)
                    else:
                        p = 0
                    ap = ap + p
                    apn = apn + 1

                ap = ap / apn

                # store it
                if min_score == -1:
                    det_results['ov{}'.format(min_overlap)] = ap
                else:
                    ap_results['ov{}_score{}'.format(min_overlap, min_score)] = ap

        map_value = np.mean(list(ap_results.values()))
        det_map = np.mean(list(det_results.values()))
        return {'map': map_value,
                'ap_breakdown': ap_results,
                'detmap': det_map,
                'det_breakdown': det_results}


def evaluate_test_captions():
    """ Evaluates test set captions

    :param results_file_path: file path to save the results
    :type results_file_path: str
    :return: None
    """
    # Root directory of the project
    root_dir = os.getcwd()

    # Directory to save logs and trained model
    model_dir = os.path.join(root_dir, "logs")

    # Local path to trained weights file
    model_path = os.path.join(root_dir, "img_cap_dense.h5")

    # Directory of images to run detection on
    image_dir = os.path.join(root_dir, "../dataset/visual genome")

    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    # load one-hot encodings
    id_to_word_file = '../dataset/id_to_word.pickle'
    word_to_id_file = '../dataset/word_to_id.pickle'
    embedding_matrix_file = '../dataset/embedding_matrix.pickle'
    id_to_word = pickle.load(open(id_to_word_file, 'rb'))
    word_to_id = pickle.load(open(word_to_id_file, 'rb'))
    embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'))

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    test_image_ids = image_ids_list[100000:]
    # test_image_ids = [62, 65]
    config = InferenceConfig(len(word_to_id), embedding_matrix)
    config.display()

    # Testing dataset
    dataset_test = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
    dataset_test.load_visual_genome(image_dir, test_image_ids,
                                    image_meta_file_path, data_file_path)
    dataset_test.prepare()

    #  Create model object in inference mode.
    model = DenseImageCapRCNN(mode="inference", model_dir=model_dir, config=config)

    # Load weights trained on Visual Genome dense image captioning
    model.load_weights(model_path, by_name=True)

    evaluator = DenseCaptioningEvaluator(model, dataset_test, id_to_word)
    evaluator.evaluate_captions()


if __name__ == '__main__':
    evaluate_test_captions()
