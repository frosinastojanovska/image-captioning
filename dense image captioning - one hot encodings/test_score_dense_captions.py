import os
import json
import pickle
import numpy as np

import utils
import dense_model as modellib
from preprocess import decode_caption
from eval.meteor.meteor import Meteor
from train_dense_captions import DenseCapConfig
from train_dense_captions import VisualGenomeDataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class InferenceConfig(DenseCapConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    def __init__(self, vocab_size):
        super().__init__(vocab_size)


class DenseCaptioningEvaluator:
    def __init__(self, model, dataset, id_to_word):
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
        :type boxes: np.array
        :param captions: captions for each box
        :type captions: list(list(str))
        :param thresh: IoU threshold for overlapping boxes
        :type thresh: float
        :return: merged boxes, merged captions
        :rtype: (np.array, list(list(str)))
        """
        assert (thresh > 0)
        pairwise_iou = utils.compute_overlaps(boxes, boxes)

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
        :rtype: list(np.array), list(np.array), list(list(str))
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
        :type images: list(np.array)
        :return: detected boxes, generated captions, log probabilities for each captions (box)
        :rtype: (list(np.array), list(list(str)), list(np.array))
        """
        boxes = []
        captions = []
        log_probs = []
        for i in range(num_images):
            # Run detection
            results = self.model.generate_captions([images[i]], verbose=0)
            boxes.append(results[0]['rois'])
            word_prob_log = np.log(np.max(results[0]['captions'], axis=2))
            log_prob = np.sum(word_prob_log, axis=1)
            log_probs.append(log_prob)
            image_captions = []
            for caption in results[0]['captions']:
                image_captions.append(decode_caption(caption, self.id_to_word))
            captions.append(image_captions)

        return boxes, captions, log_probs

    @staticmethod
    def assign_detections_to_ground_truth(num_images, gt_boxes, gt_captions, boxes, captions, log_probs):
        """

        :param num_images:
        :param gt_boxes:
        :param gt_captions:
        :param boxes:
        :param captions:
        :param log_probs:
        :return:
        """
        results = []
        for i in range(num_images):
            indices = np.argsort(log_probs[i])[::-1]

            num_detections = log_probs[i].shape[0]
            overlaps = utils.compute_overlaps(boxes[i], gt_boxes[i])
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

    @staticmethod
    def score_captions(records):
        m = Meteor()
        scores = []
        for r in records:
            gens = {}
            refs = {}
            for rec, i in zip(r, range(len(r))):
                gens[str(i)] = [rec['candidate']]
                refs[str(i)] = [sent[0] for sent in rec['references']]
                # score = m.compute_score(rec['candidate'], [sent[0] for sent in rec['references']])
            print('entering Meteor')
            score, scoress = m.compute_score(refs, gens)
            print(score)
            # print(scoress)
            scores.append(score)

        return scores

    def evaluate(self):
        """
        Evaluate generated dense captions
        :return: {'map': map_value, 'ap_breakdown': ap_results, 'detmap': det_map, 'det_breakdown': det_results}
        :rtype: dict

        """
        num_images = self.dataset.num_images
        image_ids = self.dataset._image_ids

        images, gt_boxes, gt_captions = self.get_gt_captions(image_ids)
        boxes, captions, log_probs = self.get_generated_captions(num_images, images)

        records = self.assign_detections_to_ground_truth(num_images, gt_boxes, gt_captions,
                                                         boxes, captions, log_probs)
        # METEOR
        scores = self.score_captions(records)
        print(scores)

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
                    mask = rec > t
                    prec_masked = prec[mask]
                    p = np.amax(prec_masked)
                    ap = ap + p
                    apn = apn + 1

                ap = ap / apn

                # store it
                if min_score == -1:
                    det_results['ov{}'.format(min_overlap)] = ap
                else:
                    ap_results['ov{}_score{}'.format(min_overlap, min_score)] = ap

        map_value = np.mean(ap_results.values())
        det_map = np.mean(det_results.values())
        return {'map': map_value, 'ap_breakdown': ap_results, 'detmap': det_map, 'det_breakdown': det_results}


def evaluate_test_captions(results_file_path):
    """ Evaluates test set captions

    :param results_file_path: file path to save the results
    :type results_file_path: str
    :return: None
    """
    # Root directory of the project
    root_dir = os.getcwd()

    # Local path to trained weights file
    model_path = os.path.join(root_dir, "img_cap_dense.h5")

    # Directory of images to run detection on
    image_dir = os.path.join(root_dir, "../dataset/visual genome")

    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    # load one-hot encodings
    id_to_word_file = '../dataset/id_to_word.pickle'
    word_to_id_file = '../dataset/word_to_id.pickle'
    id_to_word = pickle.load(open(id_to_word_file, 'rb'))
    word_to_id = pickle.load(open(word_to_id_file, 'rb'))

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    test_image_ids = image_ids_list[100000:]
    test_image_ids = [62, 65]
    config = InferenceConfig(len(word_to_id))
    config.display()

    # Testing dataset
    dataset_test = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
    dataset_test.load_visual_genome(image_dir, test_image_ids,
                                    image_meta_file_path, data_file_path)
    dataset_test.prepare()

    #  Create model object in inference mode.
    model = modellib.DenseImageCapRCNN(mode="inference", model_dir=model_path, config=config)

    evaluator = DenseCaptioningEvaluator(model, dataset_test, id_to_word)
    results = evaluator.evaluate()
    with open(results_file_path, 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    evaluate_test_captions('results_test.p')
