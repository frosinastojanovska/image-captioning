import os
import json
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

from utils import compute_overlaps, non_max_suppression, resize_image

from eval.spice.spice import Spice
from eval.rouge.rouge import Rouge
from eval.meteor.meteor import Meteor
from eval.cider.cider import Cider
from eval.bleu.bleu import Bleu
from eval.tokenizer.ptbtokenizer import PTBTokenizer
from generate_one_roi_features import load_model
from preprocess import decode_word, load_embeddings, tokenize_corpus, load_corpus
from eval_text_generation_model_v2 import build_model, DenseCapConfig, VisualGenomeDataset
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class InferenceConfig(DenseCapConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    POST_NMS_ROIS_INFERENCE = 200
    DETECTION_NMS_THRESHOLD = 0.5
    DETECTION_MAX_INSTANCES = 100

    # Padding size
    PADDING_SIZE = 15

    def __init__(self, vocab_size, embedding_weights):
        super(InferenceConfig, self).__init__(vocab_size, embedding_weights)


def generate_features(image, dataset, image_id, model):
    # Run detection
    rois, _ = dataset.load_captions_and_rois(image_id)
    rois = np.expand_dims(rois, axis=0)
    results = model.generate_captions([image], rois, verbose=0)
    boxes = results[0]['rois']
    features = results[0]['features']
    return boxes, features


class DenseCaptioningEvaluator:
    def __init__(self, model, feature_model, text_metrics, dataset, id_to_word, word_to_id, config, model_name):
        """
        :param model: Dense Captioning keras model in inference mode
        :type model: keras model object
        :param feature_model: ROI features model
        :type feature_model: keras model object
        :param text_metrics: language metrics, one of {'SPICE', 'METEOR'}
        :type text_metrics: str
        :param dataset: the test dataset
        :type dataset: VisualGenomeDataset object
        :param id_to_word: mapping from id to words from the vocabulary
        :type id_to_word: dict
        :param word_to_id: mapping from words to ids from the vocabulary
        :type word_to_id: dict
        :param config: configuration object
        :type config: DenseCapConfig object
        :param model_name: name of the model
        :type model_name: str
        """
        assert (text_metrics in {'SPICE', 'METEOR'})
        self.min_overlaps = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.min_scores = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.model = model
        self.feature_model = feature_model
        self.method = text_metrics
        self.dataset = dataset
        self.id_to_word = id_to_word
        self.word_to_id = word_to_id
        self.config = config
        self.model_name = model_name
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

    def get_generated_captions(self, num_images, images):
        """ Returns the generated dense captions from the model for the given images

        :param num_images: number of images
        :type num_images: int
        :param images: the images
        :type images: list(numpy.array)
        :return: detected boxes, generated captions, log probabilities for each captions (box)
        :rtype: (list(numpy.array), list(list(str)), list(numpy.array))
        """
        boxes_file = '../dataset/' + self.model_name + '_predictions_boxes.pickle'
        captions_file = '../dataset/' + self.model_name + '_predictions_caps.pickle'
        log_probs_file = '../dataset/' + self.model_name + '_predictions_probs.pickle'
        if os.path.exists(boxes_file) and os.path.exists(captions_file) and os.path.exists(log_probs_file):
            with open(boxes_file, 'rb') as f1, open(captions_file, 'rb') as f2, open(log_probs_file, 'rb') as f3:
                boxes = pickle.load(f1)
                captions = pickle.load(f2)
                log_probs = pickle.load(f3)
        else:
            boxes = []
            captions = []
            log_probs = []
            for i in tqdm(list(range(num_images))):
                # Run detection
                im_id = self.dataset._image_ids[i]
                _, window, _, _ = resize_image(images[i],
                                               min_dim=self.config.IMAGE_MIN_DIM,
                                               max_dim=self.config.IMAGE_MAX_DIM,
                                               padding=self.config.IMAGE_PADDING)
                img_boxes, img_features = generate_features(images[i], self.dataset, im_id, self.feature_model)
                caps = []
                for j in range(img_boxes.shape[0]):
                    f = img_features[j]
                    start_word = np.zeros(self.config.VOCABULARY_SIZE)
                    prev = [start_word]
                    for k in range(self.config.PADDING_SIZE - 1):
                        res = self.model.predict([np.array([f]),
                                                  np.array([pad_sequences([[np.argmax(cap) for cap in prev]],
                                                                          self.config.PADDING_SIZE)[0]])])
                        prev.append(res[0])
                    caps.append(np.vstack(prev[1:]))
                rois, img_captions = self.refine_generations(img_boxes, np.array(caps), window, self.config)
                image_captions = []
                for cap in img_captions.tolist():
                    cap = ' '.join([decode_word(c, self.id_to_word) for c in cap])
                    cap = cap.split(' .', maxsplit=1)[0]
                    image_captions.append(cap)
                boxes.append(rois)
                word_prob_log = np.log(np.max(img_captions, axis=2))
                log_prob = np.sum(word_prob_log, axis=1)
                log_probs.append(log_prob)
                captions.append(image_captions)
            with open(boxes_file, 'wb') as f1, open(captions_file, 'wb') as f2, open(log_probs_file, 'wb') as f3:
                pickle.dump(boxes, f1)
                pickle.dump(captions, f2)
                pickle.dump(log_probs, f3)

        return boxes, captions, log_probs

    def refine_generations(self, rois, captions, window, config):
        """Refine classified proposals and filter overlaps and return final
        generations.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            captions: [N, embeddings]. Captions embeddings
            window: (y1, x1, y2, x2) in image coordinates. The part of the image
                that contains the image excluding the padding.

        Returns generations shaped: [N, (y1, x1, y2, x2, embeddings)]
        """
        # Captions word IDs per ROI
        word_probs = np.max(captions, axis=2)
        word_probs_log = np.log(word_probs)
        captions_scores = np.sum(word_probs_log, axis=1)
        # Convert coordinates to image domain
        # TODO: better to keep them normalized until later
        # height, width = config.IMAGE_SHAPE[:2]
        # rois *= np.array([height, width, height, width])
        # # Clip boxes to image window
        # refined_rois = self.clip_to_window(window, rois)
        # # Round and cast to int since we're dealing with pixels now
        # refined_rois = np.rint(refined_rois).astype(np.int32)

        refined_rois = rois

        # TODO: Filter out boxes with zero area

        keep = non_max_suppression(rois, captions_scores,
                                   config.DETECTION_NMS_THRESHOLD)

        # Keep top generations
        roi_count = config.DETECTION_MAX_INSTANCES
        top_ids = np.argsort(captions_scores[keep])[::-1][:roi_count]
        keep = keep[top_ids]

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are in image domain.
        result = (refined_rois[keep], captions[keep])
        return result

    def clip_to_window(self, window, boxes):
        """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
        """
        boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
        boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
        boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
        boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
        return boxes

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
        """ Scores generated captions using text metrics

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
        for r in records:
            i = 0
            r1 = [x for x in r if len(x['references']) > 0]
            for rec in r1:
                gens[str(i)] = [rec['candidate']]
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
    
    def evaluate_captions(self, img_ids):
        """
            Evaluate generated dense captions
        """
        num_images = self.dataset.num_images
        image_ids = self.dataset._image_ids

        images, gt_boxes, gt_captions = self.get_gt_captions(num_images)
        boxes, captions, log_probs = self.get_generated_captions(num_images, images)

        records = self.assign_detections_to_ground_truth(num_images, gt_boxes, gt_captions,
                                                         boxes, captions, log_probs)
        # text scores
        if not os.path.exists('../dataset/m1_i_scores_spice.csv'):
            print('Evaluating with SPICE...')
            scores = self.score_captions(records, 'SPICE')
            scores = [np.average(s) for s in scores]
            values = dict()
            for i, s in zip(img_ids, scores):
                values[i] = s
            pd.DataFrame().from_dict(values, orient='index').to_csv('../dataset/m1_i_scores_spice.csv')
        if not os.path.exists('../dataset/m1_i_scores_rouge.csv'):
            print('Evaluating with ROUGE...')
            scores = self.score_captions(records, 'ROUGE')
            scores = [np.average(s) for s in scores]
            values = dict()
            for i, s in zip(img_ids, scores):
                values[i] = s
            pd.DataFrame().from_dict(values, orient='index').to_csv('../dataset/m1_i_scores_rouge.csv')
        if not os.path.exists('../dataset/m1_i_scores_meteor.csv'):
            print('Evaluating with METEOR...')
            scores = self.score_captions(records, 'METEOR')
            scores = [np.average(s) for s in scores]
            values = dict()
            for i, s in zip(img_ids, scores):
                values[i] = s
            pd.DataFrame().from_dict(values, orient='index').to_csv('../dataset/m1_i_scores_meteor.csv')
        if not os.path.exists('../dataset/m1_i_scores_cider.csv'):
            print('Evaluating with CIDER...')
            scores = self.score_captions(records, 'CIDER')
            scores = [np.average(s) for s in scores]
            values = dict()
            for i, s in zip(img_ids, scores):
                values[i] = s
            pd.DataFrame().from_dict(values, orient='index').to_csv('../dataset/m1_i_scores_cider.csv')
        if not os.path.exists('../dataset/m1_i_scores_bleu_1.csv'):
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
            scores_1 = [np.average(s) for s in scores_1]
            values_1 = dict()
            for i, s in zip(img_ids, scores_1):
                values_1[i] = s
            pd.DataFrame().from_dict(values_1, orient='index').to_csv('../dataset/m1_i_scores_bleu_1.csv')
            scores_2 = [np.average(s) for s in scores_2]
            values_2 = dict()
            for i, s in zip(img_ids, scores_2):
                values_2[i] = s
            pd.DataFrame().from_dict(values_2, orient='index').to_csv('../dataset/m1_i_scores_bleu_2.csv')
            scores_3 = [np.average(s) for s in scores_3]
            values_3 = dict()
            for i, s in zip(img_ids, scores_3):
                values_3[i] = s
            pd.DataFrame().from_dict(values_3, orient='index').to_csv('../dataset/m1_i_scores_bleu_3.csv')
            scores_4 = [np.average(s) for s in scores_4]
            values_4 = dict()
            for i, s in zip(img_ids, scores_4):
                values_4[i] = s
            pd.DataFrame().from_dict(values_4, orient='index').to_csv('../dataset/m1_i_scores_bleu_4.csv')

    def evaluate(self, img_ids):
        """ Evaluate generated dense captions

        :return: {'map': map_value, 'ap_breakdown': ap_results, 'detmap': det_map, 'det_breakdown': det_results}
        :rtype: dict

        """
        num_images = self.dataset.num_images
        image_ids = self.dataset._image_ids

        images, gt_boxes, gt_captions = self.get_gt_captions(image_ids)
        boxes, captions, log_probs = self.get_generated_captions(num_images, images)

        records = self.assign_detections_to_ground_truth(num_images, gt_boxes, gt_captions,
                                                         boxes, captions, log_probs)
        dice_scores = dict()
        for record, im_id in zip(records, img_ids):
            dice_scores[im_id] = np.average([r['ov'] for r in record])
        pd.DataFrame().from_dict(dice_scores, orient='index').to_csv('../dataset/m1_dice_scores.csv')


def evaluate_test_captions():
    """ Evaluates test set captions

    :param results_file_path: file path to save the results
    :type results_file_path: str
    :return: None
    """
    inject = True
    embeddings_file_path = '../dataset/glove.6B.300d.txt'

    data_directory = '../dataset/visual genome/'
    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    train_val_image_ids = image_ids_list[:100000]
    # test_image_ids = image_ids_list[100000:102000]
    test_image_ids = image_ids_list[100000:]

    id_to_word_file = '../dataset/id_to_word_v2.pickle'
    word_to_id_file = '../dataset/word_to_id_v2.pickle'
    embedding_matrix_file = '../dataset/embedding_matrix_v2.pickle'

    if not os.path.exists(id_to_word_file) or not os.path.exists(word_to_id_file) \
            or not os.path.exists(embedding_matrix_file):
        embeddings = load_embeddings(embeddings_file_path)
        tokens = tokenize_corpus(data_file_path, train_val_image_ids, embeddings)
        word_to_id, id_to_word, embedding_matrix = load_corpus(list(tokens), embeddings, 300)

        with open(id_to_word_file, 'wb') as f:
            pickle.dump(id_to_word, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(word_to_id_file, 'wb') as f:
            pickle.dump(word_to_id, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(embedding_matrix_file, 'wb') as f:
            pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(id_to_word_file, 'rb') as f:
            id_to_word = pickle.load(f)
        with open(word_to_id_file, 'rb') as f:
            word_to_id = pickle.load(f)
        with open(embedding_matrix_file, 'rb') as f:
            embedding_matrix = pickle.load(f)

    config = DenseCapConfig(len(id_to_word), embedding_matrix)
    config.display()

    features_model = load_model(use_generated_rois=True)

    model = build_model((7, 7, 256), (10,), config, 256, inject)
    model.load_weights('mask_rcnn_coco.h5', by_name=True)
    print(model.summary())

    dataset_test = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
    dataset_test.load_visual_genome(data_directory, test_image_ids, image_meta_file_path,
                                    data_file_path)
    dataset_test.prepare()
    if inject:
        model.load_weights('models/model1-85-3.69.h5')
        model_name = 'm1'
    else:
        model.load_weights('models/model2-85-4.01.h5')
        model_name = 'm2'

    evaluator = DenseCaptioningEvaluator(model, features_model, 'METEOR', dataset_test,
                                         id_to_word, word_to_id, config, model_name)
    evaluator.evaluate_captions(test_image_ids)
    evaluator.evaluate(test_image_ids)


if __name__ == '__main__':
    evaluate_test_captions()
